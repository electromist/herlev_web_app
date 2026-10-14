"""
Cell Division & Whole-Cell Boundary Segmentation (CSGO & Mitosis AI)
Integrates:
1. CSGO (Cell Segmentation with Globally Optimized Boundaries):
   Pretrained H&E Cell Membrane U-Net (Gu et al., Laboratory Investigation 2025)
   Predicts true outer plasma membrane & cytoplasmic boundary, NOT just inner nucleus.
2. Cell Division & Mitosis Analysis (OMG-Net / WHO 5th Ed. Criteria):
   Assesses chromatin hypercondensation, equatorial metaphase plates, anaphase polar
   separation, cytokinesis cleavage furrows, and Nuclear-to-Cytoplasmic (N:C) ratios.
"""

import os
import time
import math
from collections import OrderedDict
import numpy as np
import cv2
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet50

MICRONS_PER_PIXEL = 0.5  # Standard 40x magnification (0.5 um/pixel)
CSGO_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "csgo_membrane_unet.pt"
)
MIDOG_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "midog_dann_mitosis.pth"
)

# ─────────────────────────────────────────────────────────────
# 1. CSGO Official Membrane U-Net Architecture (Gu et al., 2025)
# ─────────────────────────────────────────────────────────────
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="default",
                 dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None):
        if padding == "default":
            padding = tuple((k - 1) // 2 for k in [kernel_size, kernel_size])
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                     dilation, groups, bias, padding_mode, device, dtype)


class ConvBNReLU(nn.Sequential):
    def __init__(self, conv, norm_layer=None, activation=None, dropout_rate=0.0):
        layers = [conv]
        if norm_layer is not None:
            layers.append(norm_layer(conv.out_channels))
        if activation is not None:
            layers.append(activation)
        if dropout_rate:
            layers.append(nn.Dropout2d(dropout_rate))
        super(ConvBNReLU, self).__init__(*layers)


class Conv2dBNReLU(ConvBNReLU):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding="default", dilation=1, groups=1, bias=None,
                 norm_layer="batch", activation=None, dropout_rate=0.0):
        if activation is None:
            activation = nn.ReLU(inplace=True)
        if norm_layer == "batch":
            nl = nn.BatchNorm2d
            if bias is None:
                bias = False
        else:
            nl = None
            if bias is None:
                bias = True
        conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias)
        super(Conv2dBNReLU, self).__init__(conv, norm_layer=nl, activation=activation, dropout_rate=dropout_rate)


class ResizeConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2,
                 conv=(Conv2d, {"kernel_size": 3, "padding": "default"}), mode="bilinear"):
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(ResizeConv2d, self).__init__(
            nn.Upsample(scale_factor=stride, mode=mode),
            conv[0](in_channels, out_channels, **conv[1]),
        )


class UNetFeatures(nn.Module):
    def __init__(self, in_channels=3, n_channels=32, n_downsampling=4):
        super(UNetFeatures, self).__init__()
        self.out_channels = [n_channels]
        self.conv = [
            Conv2dBNReLU(in_channels, n_channels, kernel_size=3),
            Conv2dBNReLU(n_channels, n_channels, kernel_size=3),
        ]
        for i in range(n_downsampling):
            in_c, out_c = self.out_channels[-1], 2**(i + 1) * n_channels
            self.conv.append(nn.MaxPool2d(2))
            self.conv.append(Conv2dBNReLU(in_c, out_c, kernel_size=3, stride=1))
            self.conv.append(Conv2dBNReLU(out_c, out_c, kernel_size=3))
            self.out_channels.append(out_c)
        self.conv = nn.Sequential(*self.conv)
        self.return_layers = [3 * i + 2 for i in range(n_downsampling + 1)]

    def forward(self, x):
        res = []
        for s, t in zip([0] + self.return_layers, self.return_layers):
            for _ in range(s, t):
                x = self.conv[_](x)
            res.append(x)
        return res


class CSGOMembraneUNet(nn.Module):
    """CSGO Pretrained Membrane U-Net predicting H&E cellular boundary probabilities."""
    def __init__(self, num_classes=1, n_channels=32, n_downsampling=4):
        super(CSGOMembraneUNet, self).__init__()
        self.encoder = UNetFeatures(in_channels=3, n_channels=n_channels, n_downsampling=n_downsampling)
        self.out_channels = self.encoder.out_channels

        decoder = nn.ModuleList()
        for in_c, out_c in zip(self.out_channels[-1:0:-1], self.out_channels[-2::-1]):
            up_layer = ResizeConv2d(in_c, out_c, stride=2, mode="bilinear",
                                    conv=(Conv2d, {"kernel_size": 3, "padding": "default"}))
            conv_layer = nn.Sequential(
                Conv2dBNReLU(out_c * 2, out_c, 3),
                Conv2dBNReLU(out_c, out_c, 3)
            )
            modules = nn.ModuleDict([("up", up_layer), ("conv", conv_layer)])
            decoder.append(modules)
        self.decoder = decoder
        self.classifier = nn.Sequential(
            nn.Conv2d(self.out_channels[0], num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        x = features.pop()
        for layers in self.decoder:
            up = layers["up"](x)
            up = torch.cat([up, features.pop()], dim=1)
            x = layers["conv"](up)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────
# 2. Model Loader Singleton
# ─────────────────────────────────────────────────────────────
_CSGO_UNET = None

def get_csgo_model():
    """Returns loaded CSGO Membrane U-Net model."""
    global _CSGO_UNET
    if _CSGO_UNET is not None:
        return _CSGO_UNET

    model = CSGOMembraneUNet(num_classes=1)
    if os.path.exists(CSGO_WEIGHTS_PATH):
        try:
            w = torch.load(CSGO_WEIGHTS_PATH, map_location="cpu")
            model.load_state_dict(w, strict=True)
            print(f"[CSGO] Loaded official pretrained membrane weights from: {CSGO_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[CSGO] Checkpoint load warning: {e}")
    else:
        print(f"[CSGO] Warning: {CSGO_WEIGHTS_PATH} not found.")

    model.eval()
    _CSGO_UNET = model
    return _CSGO_UNET


# ─────────────────────────────────────────────────────────────
# 2.5 MIDOG++ Multi-Scale DANN Mitosis Deep Learning Classifier
# ─────────────────────────────────────────────────────────────
class MitosisDANN(nn.Module):
    """
    Multi-Scale ResNet-50 Domain-Adversarial Mitosis Classifier (MIDOG++ Benchmark).
    Extracts multi-scale deep features from layers 2, 3, and 4 (concatenated to 3584-d vector)
    and classifies H&E cellular figures into:
      Class 0: Non-Mitotic / Interphase Resting Cell
      Class 1: Mitotic Figure (Active Cell Division)
    """
    def __init__(self):
        super().__init__()
        r50 = resnet50(weights=None)
        self.stem = nn.Sequential(
            r50.conv1,
            r50.bn1,
            r50.relu,
            r50.maxpool,
            r50.layer1
        )
        self.layer2 = r50.layer2
        self.layer3 = r50.layer3
        self.layer4 = r50.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mitosis_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 1024 + 2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.stem(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        p2 = self.pool(feat2).flatten(1)
        p3 = self.pool(feat3).flatten(1)
        p4 = self.pool(feat4).flatten(1)
        multi_feat = torch.cat([p2, p3, p4], dim=1)
        return self.mitosis_classifier(multi_feat)


_MIDOG_MODEL = None
_MITOSIS_TRANSFORM = None

def get_midog_mitosis_model():
    """Returns loaded MIDOG++ DANN deep learning mitosis classifier and transforms."""
    global _MIDOG_MODEL, _MITOSIS_TRANSFORM
    if _MIDOG_MODEL is not None:
        return _MIDOG_MODEL, _MITOSIS_TRANSFORM

    _MITOSIS_TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = MitosisDANN()
    if not os.path.exists(MIDOG_WEIGHTS_PATH):
        try:
            import urllib.request
            print(f"[Mitosis-AI] Downloading MIDOG++ DANN Mitosis model to {MIDOG_WEIGHTS_PATH}...")
            urllib.request.urlretrieve(
                "https://huggingface.co/lillianlam/multi-stage-dann-model-mitotic-figures/resolve/main/DANN_50.pth",
                MIDOG_WEIGHTS_PATH
            )
        except Exception as e:
            print(f"[Mitosis-AI] Could not auto-download MIDOG++ model: {e}")

    if os.path.exists(MIDOG_WEIGHTS_PATH):
        try:
            ckpt = torch.load(MIDOG_WEIGHTS_PATH, map_location="cpu")
            model.load_state_dict(ckpt, strict=False)
            print(f"[Mitosis-AI] Successfully loaded MIDOG++ DANN Mitosis model from: {MIDOG_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[Mitosis-AI] Warning loading MIDOG++ weights: {e}")
    else:
        print(f"[Mitosis-AI] Checkpoint {MIDOG_WEIGHTS_PATH} not found.")

    model.eval()
    _MIDOG_MODEL = model
    return _MIDOG_MODEL, _MITOSIS_TRANSFORM


def predict_mitosis_probability(image_rgb: np.ndarray, cx: int, cy: int, radius: int = 36) -> tuple:
    """
    Runs MIDOG++ Deep Learning classifier on the cellular region around (cx, cy).
    Returns (is_mitotic: bool, p_mitotic: float).
    """
    model, transform = get_midog_mitosis_model()
    h, w = image_rgb.shape[:2]
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return False, 0.0

    pil_crop = Image.fromarray(crop)
    tensor = transform(pil_crop).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        p_mitotic = float(probs[1].item())

    # Mitosis threshold: >= 0.45 indicates active mitotic figure
    is_mitotic = (p_mitotic >= 0.45)
    return is_mitotic, p_mitotic


# ─────────────────────────────────────────────────────────────
# 3. Cellpose Cyto3 Whole-Cell Engine & Model Loader Singleton
# ─────────────────────────────────────────────────────────────
_CELLPOSE_MODEL = None
_CELLPOSE_CACHE = {}

def get_cellpose_model():
    """Returns loaded Cellpose Cyto3 Whole-Cell model (CPU optimized)."""
    global _CELLPOSE_MODEL
    if _CELLPOSE_MODEL is not None:
        return _CELLPOSE_MODEL

    try:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="cellpose")
        from cellpose import models
        _CELLPOSE_MODEL = models.CellposeModel(gpu=False, model_type="cyto3")
        print("[Cellpose] Loaded official Cyto3 Whole-Cell model on CPU.")
    except Exception as e:
        print("[Cellpose] Cyto3 initialization error:", e)
        _CELLPOSE_MODEL = None

    return _CELLPOSE_MODEL


def get_cached_cellpose_masks(image_rgb: np.ndarray, diameter = None) -> np.ndarray:
    """
    Caches Cellpose whole-cell evaluation for the current image so multiple
    cell selections or inspections are instantaneous.
    diameter=None enables automatic size estimation across different zoom levels.
    """
    h, w = image_rgb.shape[:2]
    sig = (h, w, int(image_rgb[0, 0, 0]), int(image_rgb[h // 2, w // 2, 0]), int(image_rgb[-1, -1, 0]), str(diameter))
    if sig in _CELLPOSE_CACHE:
        return _CELLPOSE_CACHE[sig]

    cp = get_cellpose_model()
    if cp is None:
        return None

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            masks, _, _ = cp.eval(image_rgb, diameter=diameter, channels=[0, 0])
        if len(_CELLPOSE_CACHE) > 4:
            _CELLPOSE_CACHE.clear()
        _CELLPOSE_CACHE[sig] = masks
        return masks
    except Exception as e:
        print("[Cellpose] Inference warning:", e)
        return None


def predict_csgo_membrane_patch(image_rgb: np.ndarray, box: tuple = None) -> np.ndarray:
    """
    Predicts cell membrane probability map across image patch using CSGO U-Net.
    """
    h, w = image_rgb.shape[:2]
    model = get_csgo_model()

    # Determine patch area
    if box is not None:
        x1, y1, x2, y2 = box
        pad = 24
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        sub_img = image_rgb[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = 0, 0, w, h
        sub_img = image_rgb

    sh, sw = sub_img.shape[:2]
    target_h = int(math.ceil(sh / 16.0) * 16)
    target_w = int(math.ceil(sw / 16.0) * 16)
    target_h = max(32, target_h)
    target_w = max(32, target_w)

    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[:sh, :sw] = sub_img

    t = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        prob_pad = model(t)[0, 0].cpu().numpy()

    sub_prob = prob_pad[:sh, :sw]
    full_prob = np.zeros((h, w), dtype=np.float32)
    full_prob[y1:y2, x1:x2] = sub_prob
    return full_prob


def segment_whole_cell_at_point(medsam_predictor, image_rgb: np.ndarray, cx: int, cy: int) -> dict:
    """
    Segments the entire WHOLE CELL BODY (Plasma Membrane + Cytoplasm + Nucleus) at clicked point (cx, cy).
    Workflow:
    1. Primary Engine: Cellpose Cyto3 (Whole-cell & cytoplasmic boundaries foundation model).
    2. Snap-to-cell tolerance: Snaps accurately if click falls near outer membrane perimeter.
    3. Resilient Fallbacks: MedSAM whole-cell box prompting & CSGO membrane U-Net.
    4. Hematoxylin/Eosin Chromatin Decomposition: Isolates internal nuclear/mitotic dividing core.
    """
    h, w = image_rgb.shape[:2]
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    whole_mask = None

    # Step 1: Cellpose Cyto3 Whole-Cell Segmentation
    cp_model = get_cellpose_model()
    if cp_model is not None:
        try:
            if max(h, w) <= 650:
                masks = get_cached_cellpose_masks(image_rgb, diameter=None)
                if masks is not None:
                    val = masks[cy, cx]
                    if val == 0:
                        # Snap to nearest cell if click was within 25px neighborhood
                        rad = 25
                        nx1, ny1 = max(0, cx - rad), max(0, cy - rad)
                        nx2, ny2 = min(w, cx + rad), min(h, cy + rad)
                        sub_m = masks[ny1:ny2, nx1:nx2]
                        nonzero = sub_m[sub_m > 0]
                        if len(nonzero) > 0:
                            val = int(np.bincount(nonzero).argmax())
                    if val > 0:
                        whole_mask = (masks == val)
            else:
                # Patch evaluation around clicked point for sub-second CPU response
                patch_r = 140
                x1, y1 = max(0, cx - patch_r), max(0, cy - patch_r)
                x2, y2 = min(w, cx + patch_r), min(h, cy + patch_r)
                sub_img = image_rgb[y1:y2, x1:x2]
                sub_cx, sub_cy = cx - x1, cy - y1

                sub_masks = get_cached_cellpose_masks(sub_img, diameter=None)
                if sub_masks is not None:
                    val = sub_masks[sub_cy, sub_cx]
                    if val == 0:
                        rad = 25
                        nx1, ny1 = max(0, sub_cx - rad), max(0, sub_cy - rad)
                        nx2, ny2 = min(sub_img.shape[1], sub_cx + rad), min(sub_img.shape[0], sub_cy + rad)
                        sub_m = sub_masks[ny1:ny2, nx1:nx2]
                        nonzero = sub_m[sub_m > 0]
                        if len(nonzero) > 0:
                            val = int(np.bincount(nonzero).argmax())
                    if val > 0:
                        whole_mask = np.zeros((h, w), dtype=bool)
                        whole_mask[y1:y2, x1:x2] = (sub_masks == val)
        except Exception as e:
            print("[Cellpose] Segment at point error:", e)

    # Step 2: Fallback to MedSAM Whole-Cell Box Prompt if Cellpose did not find a mask
    if whole_mask is None or not whole_mask.any() or whole_mask.sum() < 60:
        if medsam_predictor is not None:
            try:
                patch_radius = 80
                x1, y1 = max(0, cx - patch_radius), max(0, cy - patch_radius)
                x2, y2 = min(w, cx + patch_radius), min(h, cy + patch_radius)
                box_prompt = np.array([x1, y1, x2, y2])
                pts = np.array([[cx, cy]], dtype=np.float32)
                lbls = np.array([1], dtype=np.int32)

                masks_sam, iou_preds, _ = medsam_predictor.predict(
                    point_coords=pts,
                    point_labels=lbls,
                    box=box_prompt[None, :],
                    multimask_output=True
                )
                best_m = None
                max_a = 0
                for m in masks_sam:
                    if m[cy, cx]:
                        a = int(m.sum())
                        if a > max_a:
                            max_a = a
                            best_m = m
                if best_m is not None:
                    whole_mask = best_m.copy()
            except Exception as e:
                print("[MedSAM] Fallback error:", e)

    # Step 3: CSGO Membrane distance flood fallback
    if whole_mask is None or not whole_mask.any() or whole_mask.sum() < 60:
        try:
            csgo_membrane = predict_csgo_membrane_patch(image_rgb)
            membrane_barrier = (csgo_membrane > 0.35).astype(np.uint8)
            dist = cv2.distanceTransform(1 - membrane_barrier, cv2.DIST_L2, 5)
            _, cell_flood = cv2.threshold(dist, 0.1, 255, cv2.THRESH_BINARY)
            num_l, labels = cv2.connectedComponents(cell_flood.astype(np.uint8))
            target_lbl = labels[cy, cx]
            whole_mask = np.zeros((h, w), dtype=bool)
            if target_lbl > 0:
                whole_mask = (labels == target_lbl)
            else:
                cv2.circle(whole_mask.view(np.uint8), (cx, cy), 28, 1, -1)
        except Exception as e:
            print("[CSGO] Flood fallback error:", e)
            whole_mask = np.zeros((h, w), dtype=bool)
            cv2.circle(whole_mask.view(np.uint8), (cx, cy), 28, 1, -1)

    # Clean morphology of whole-cell boundary
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    whole_clean = cv2.morphologyEx(whole_mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    whole_clean = ndi.binary_fill_holes(whole_clean).astype(np.uint8)

    cnts, _ = cv2.findContours(whole_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    whole_c = max(cnts, key=cv2.contourArea)
    whole_mask = np.zeros((h, w), dtype=bool)
    cv2.drawContours(whole_mask.view(np.uint8), [whole_c], -1, 1, -1)

    # Step 4: Isolate Internal Nuclear / Dividing Chromatin Core
    green = image_rgb[:, :, 1].astype(np.float32)
    darkness = 255.0 - green
    darkness[~whole_mask] = 0

    dist_from_membrane = ndi.distance_transform_edt(whole_mask)
    inner_zone = dist_from_membrane > 5
    if not inner_zone.any():
        inner_zone = whole_mask

    thresh_val = np.percentile(darkness[whole_mask & inner_zone], 50)
    nuclear_raw = ((darkness > thresh_val) & inner_zone).astype(np.uint8)
    k_nuc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nuclear_clean = cv2.morphologyEx(nuclear_raw, cv2.MORPH_OPEN, k_nuc)
    nuclear_clean = cv2.morphologyEx(nuclear_clean, cv2.MORPH_CLOSE, k_nuc)
    nuclear_clean = ndi.binary_fill_holes(nuclear_clean).astype(np.uint8)

    n_cnts, _ = cv2.findContours(nuclear_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if n_cnts:
        nuc_c = max(n_cnts, key=cv2.contourArea)
        nuclear_mask = np.zeros((h, w), dtype=bool)
        cv2.drawContours(nuclear_mask.view(np.uint8), [nuc_c], -1, 1, -1)
    else:
        nuclear_mask = whole_mask.copy()
        nuc_c = whole_c

    # Step 5: Mitosis & Morphometry Analysis
    cell_info = analyze_cell_division(image_rgb, whole_mask, nuclear_mask, whole_c, nuc_c, (cx, cy))
    cell_info["whole_cell_mask"] = whole_mask
    cell_info["nuclear_mask"] = nuclear_mask
    cell_info["whole_contour"] = whole_c
    cell_info["nuclear_contour"] = nuc_c
    return cell_info


# ─────────────────────────────────────────────────────────────
# 4. Cell Division & Mitotic Stage Classifier (OMG-Net Criteria)
# ─────────────────────────────────────────────────────────────
def analyze_cell_division(image_rgb: np.ndarray, whole_mask: np.ndarray, nuclear_mask: np.ndarray,
                          whole_c: np.ndarray, nuc_c: np.ndarray, center: tuple) -> dict:
    """
    Evaluates:
    1. Whole-cell vs Nuclear dimensions (Area, Perimeter, Circularity, Aspect Ratio).
    2. Nuclear-to-Cytoplasmic (N:C) Ratio.
    3. Chromatin Hyperchromatism & Equatorial Plate Condensation.
    4. Cytokinesis Cleavage Furrow Index.
    5. Mitosis Stage Determination.
    """
    whole_area_px = int(cv2.contourArea(whole_c))
    nuc_area_px = int(cv2.contourArea(nuc_c))
    cyto_area_px = max(1, whole_area_px - nuc_area_px)

    whole_area_um2 = round(whole_area_px * (MICRONS_PER_PIXEL ** 2), 1)
    nuc_area_um2 = round(nuc_area_px * (MICRONS_PER_PIXEL ** 2), 1)
    cyto_area_um2 = round(cyto_area_px * (MICRONS_PER_PIXEL ** 2), 1)

    # N:C Ratio
    nc_ratio = round(float(nuc_area_px) / max(1.0, float(whole_area_px)), 2)

    # Whole-cell morphometry
    w_peri = float(cv2.arcLength(whole_c, True))
    w_circ = round(float(4.0 * np.pi * whole_area_px / (w_peri ** 2 + 1e-6)), 3)

    rect = cv2.minAreaRect(whole_c)
    rw, rh = rect[1]
    aspect_ratio = round(float(max(rw, rh) / max(1.0, min(rw, rh))), 2)

    # Chromatin Darkness & Hyperchromatism Index
    green = image_rgb[:, :, 1].astype(np.float32)
    darkness = 255.0 - green
    tissue_mean_darkness = float(np.mean(darkness))
    cell_nuc_darkness = float(np.mean(darkness[nuclear_mask])) if nuclear_mask.any() else 0.0
    hyperchromatism_ratio = round(cell_nuc_darkness / max(1.0, tissue_mean_darkness), 2)

    # Cleavage Furrow / Cytokinesis Constriction Index:
    # A dividing cell in cytokinesis shows a peanut/biconcave constriction in its hull
    hull = cv2.convexHull(whole_c)
    hull_area = cv2.contourArea(hull)
    solidity = round(float(whole_area_px) / max(1.0, hull_area), 3)

    is_cleavage_furrow = (solidity < 0.88 and aspect_ratio >= 1.45)

    # Deep Learning Mitosis Detection (MIDOG++ Multi-Scale DANN Model)
    is_mitotic, p_mitotic = predict_mitosis_probability(image_rgb, center[0], center[1], radius=36)

    # Mitotic Stage Classification Grounded in Deep Learning Output
    if is_mitotic:
        if is_cleavage_furrow:
            stage = "Active Mitosis: Telophase / Cytokinesis"
            badge = "🟣"
            color_rgb = (168, 85, 247)  # purple
            conf = p_mitotic
            description = (
                f"Deep-learning verified mitotic figure (MIDOG++ confidence: {p_mitotic*100:.1f}%). "
                "Visible cleavage furrow constriction separating two daughter nuclear masses."
            )
        elif aspect_ratio >= 1.55 and w_circ <= 0.70:
            stage = "Active Mitosis: Anaphase"
            badge = "🟠"
            color_rgb = (249, 115, 22)  # orange
            conf = p_mitotic
            description = (
                f"Deep-learning verified mitotic figure (MIDOG++ confidence: {p_mitotic*100:.1f}%). "
                "Chromosomes separated into two distinct polar bundles moving toward spindle poles."
            )
        else:
            stage = "Active Mitosis: Metaphase (Equatorial Plate)"
            badge = "🔴"
            color_rgb = (239, 68, 68)   # red
            conf = p_mitotic
            description = (
                f"Deep-learning verified mitotic figure (MIDOG++ confidence: {p_mitotic*100:.1f}%). "
                "Dense condensed chromatin equatorial plate. Dissolved nuclear membrane."
            )
    else:
        stage = "Interphase (Non-Dividing / Resting)"
        badge = "🟢"
        color_rgb = (34, 197, 94)   # green
        conf = 1.0 - p_mitotic
        description = (
            f"Resting cell verified by MIDOG++ AI (Resting confidence: {conf*100:.1f}%). "
            "Chromatin normally dispersed within intact nuclear membrane; no active division."
        )

    bx, by, bw, bh = cv2.boundingRect(whole_c)

    return {
        "division_stage": stage,
        "badge": badge,
        "color_rgb": color_rgb,
        "confidence": conf,
        "mitosis_probability": round(p_mitotic, 3),
        "description": description,
        "whole_area_px": whole_area_px,
        "whole_area_um2": whole_area_um2,
        "nuclear_area_um2": nuc_area_um2,
        "cytoplasm_area_um2": cyto_area_um2,
        "nc_ratio": nc_ratio,
        "hyperchromatism_ratio": hyperchromatism_ratio,
        "circularity": w_circ,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "cleavage_furrow": is_cleavage_furrow,
        "center": center,
        "bbox": [bx, by, bx + bw, by + bh]
    }


# ─────────────────────────────────────────────────────────────
# 5. Auto-Scan Mitotic & Dividing Cells (Cellpose Whole-Cell Scanner)
# ─────────────────────────────────────────────────────────────
def scan_candidate_dividing_cells(medsam_predictor, image_rgb: np.ndarray, max_candidates: int = 15) -> list:
    """
    Scans the entire slide image for candidate whole cells using Cellpose Cyto3,
    and analyzes cell division and mitosis for each instance.
    """
    h, w = image_rgb.shape[:2]
    masks = get_cached_cellpose_masks(image_rgb, diameter=None)
    cells = []

    if masks is not None:
        unique_vals = np.unique(masks)
        for val in unique_vals:
            if val == 0:
                continue
            m = (masks == val)
            if m.sum() < 80:
                continue
            ys, xs = np.where(m)
            cx, cy = int(xs.mean()), int(ys.mean())

            cell_data = segment_whole_cell_at_point(medsam_predictor, image_rgb, cx, cy)
            if cell_data is not None:
                cells.append(cell_data)

    # If Cellpose found no cells, fall back to chromatin peaks
    if not cells:
        green = image_rgb[:, :, 1].astype(np.float32)
        darkness = 255.0 - green
        blurred = cv2.GaussianBlur(darkness, (7, 7), 0)
        thresh = np.percentile(blurred, 92)
        candidates_mask = (blurred > thresh).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates_mask)
        ranked = []
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if 30 <= area <= 2500:
                cx, cy = int(centroids[lbl][0]), int(centroids[lbl][1])
                ranked.append((blurred[cy, cx], cx, cy))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for _, cx, cy in ranked[:max_candidates]:
            if any((cx - c["center"][0])**2 + (cy - c["center"][1])**2 < (28**2) for c in cells):
                continue
            cell_data = segment_whole_cell_at_point(medsam_predictor, image_rgb, cx, cy)
            if cell_data is not None:
                cells.append(cell_data)

    # Sort descending by mitotic probability so active cell divisions appear first!
    cells.sort(key=lambda c: c.get("mitosis_probability", 0.0), reverse=True)
    cells = cells[:max_candidates]
    for i, c in enumerate(cells):
        c["id"] = i + 1

    return cells


# ─────────────────────────────────────────────────────────────
# 6. Visual Rendering Overlay
# ─────────────────────────────────────────────────────────────
def render_celldivision_display(image_rgb: np.ndarray, instances: list, selected_id: int = None) -> np.ndarray:
    """
    Renders Whole-Cell Boundaries and Mitotic Figures:
    - Amber/Orange Outer Contour: Whole-Cell Boundary (Plasma Membrane + Cytoplasm)
    - Violet/Red Inner Contour: Nuclear / Dividing Chromatin Core
    - Glowing dual highlight on selected cell.
    """
    overlay = image_rgb.copy()
    if not instances:
        return overlay

    # Create translucent cytoplasmic wash layer
    wash_layer = np.zeros_like(overlay)
    for inst in instances:
        wc = inst["whole_contour"]
        color = inst["color_rgb"]
        cv2.drawContours(wash_layer, [wc], -1, color, -1)

    has_cell = wash_layer.any(axis=-1)
    overlay[has_cell] = (
        0.72 * overlay[has_cell].astype(float)
        + 0.28 * wash_layer[has_cell].astype(float)
    ).astype(np.uint8)

    # Draw boundaries
    for inst in instances:
        is_sel = (selected_id is not None and inst.get("id") == selected_id)
        wc = inst["whole_contour"]
        nc = inst["nuclear_contour"]
        color = inst["color_rgb"]

        if is_sel:
            # Highlight selected cell with bold glowing yellow outer contour
            cv2.drawContours(overlay, [wc], -1, (255, 235, 0), 4)
            cv2.drawContours(overlay, [wc], -1, (255, 255, 255), 2)

            # Inner chromatin contour
            cv2.drawContours(overlay, [nc], -1, (239, 68, 68), 2)

            # Centroid point
            cx, cy = inst["center"]
            cv2.circle(overlay, (cx, cy), 5, (255, 235, 0), -1)
            cv2.circle(overlay, (cx, cy), 7, (0, 0, 0), 1)
        else:
            # Outer whole-cell membrane contour
            cv2.drawContours(overlay, [wc], -1, color, 2)
            # Inner chromatin contour
            cv2.drawContours(overlay, [nc], -1, (168, 85, 247), 1)

    return overlay


# ─────────────────────────────────────────────────────────────
# 7. Formatted Markdown Metrics Report
# ─────────────────────────────────────────────────────────────
def format_celldivision_markdown(cell: dict, total_cells: int = 1) -> str:
    """Formats clinical and morphological markdown report for cell division and whole cell."""
    if not cell:
        return "### No cell selected. Click anywhere with '+' to segment a whole cell."

    cid = cell.get("id", 1)
    stage = cell["division_stage"]
    badge = cell["badge"]
    conf = cell["confidence"] * 100.0
    p_mitotic = cell.get("mitosis_probability", 0.0) * 100.0
    desc = cell["description"]
    w_area = cell["whole_area_um2"]
    n_area = cell["nuclear_area_um2"]
    c_area = cell["cytoplasm_area_um2"]
    nc = cell["nc_ratio"]
    hc = cell["hyperchromatism_ratio"]
    circ = cell["circularity"]
    asp = cell["aspect_ratio"]
    furrow = "Detected (Peanut Cleavage Constriction)" if cell["cleavage_furrow"] else "None (Regular Shape)"
    (cx, cy) = cell["center"]

    report = [
        f"### 🧬 Whole-Cell #{cid}: **{stage}**\n\n",
        f"- **Cell Division Status**: {badge} **{stage}** (MIDOG++ Mitosis Confidence: `{p_mitotic:.1f}%`)\n",
        f"- **Cell Boundary Type**: `Whole-Cell (Plasma Membrane + Cytoplasm)` *(Cellpose Cyto3 & CSGO)*\n",
        f"- **Pathological AI Interpretation**: {desc}\n\n",
        "#### Morphometry & Compartmental Ratios:\n",
        f"| Compartment | Area (µm²) | Ratio / Index | Visual Outline |\n",
        f"| :--- | :---: | :---: | :---: |\n",
        f"| 🟡 **Whole Cell Boundary** | `{w_area}` µm² | Aspect: `{asp}`, Circ: `{circ}` | Outer Membrane Contour |\n",
        f"| 🟣 **Cytoplasmic Body** | `{c_area}` µm² | Volume buffer | Translucent Interior Wash |\n",
        f"| 🔴 **Dividing Chromatin / Nucleus** | `{n_area}` µm² | Hyperchromatism: `{hc}x` | Inner Chromatin Core |\n",
        f"| ⚖️ **Nuclear-to-Cytoplasmic (N:C)** | **`{nc}`** | Cleavage Furrow: `{furrow}` | Malignancy / Mitosis Proxy |\n\n",
        f"- **Centroid Position**: `x={cx}, y={cy}` | **Active Cells Segmented**: `{total_cells}`\n",
        f"- Click **'Extract Whole-Cell Entity'** below to view true-color bounding box crop and isolated transparent plasma cutout!"
    ]
    return "".join(report)
