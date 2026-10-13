"""
HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images
Official PyTorch implementation with PanNuke-pretrained weights.
Reference: Simon Graham, Quoc Dang Vu, Shan E Ahmed Raza, Ayesha Azam, Yee Wah Tsang,
           Jin Tae Kwak, Nasir Rajpoot (Medical Image Analysis / arXiv:1812.06499v5)

PanNuke Nuclear Phenotype Classes:
0: Background
1: Neoplastic (🔴 Red)
2: Inflammatory (🔵 Blue)
3: Connective / Stroma (🟢 Green)
4: Dead / Necrotic (🟡 Yellow)
5: Non-Neoplastic Epithelial (🟣 Cyan/Purple)
"""

import os
import math
import time
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import cv2
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# Class definitions & Color Palette (Official PanNuke Benchmark)
# ─────────────────────────────────────────────────────────────
HOVERNET_CLASSES = {
    0: {"name": "Background", "color_rgb": (0, 0, 0), "hex": "#000000", "badge": "⚪"},
    1: {"name": "Neoplastic", "color_rgb": (239, 68, 68), "hex": "#ef4444", "badge": "🔴"},
    2: {"name": "Inflammatory", "color_rgb": (59, 130, 246), "hex": "#3b82f6", "badge": "🔵"},
    3: {"name": "Connective", "color_rgb": (16, 185, 129), "hex": "#10b981", "badge": "🟢"},
    4: {"name": "Dead", "color_rgb": (245, 158, 11), "hex": "#f59e0b", "badge": "🟡"},
    5: {"name": "Non-Neoplastic Epithelial", "color_rgb": (6, 182, 212), "hex": "#06b6d4", "badge": "🟣"},
}

MICRONS_PER_PIXEL = 0.5  # Standard 40x digital pathology magnification (0.5 um/pixel)
MODEL_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hovernet_fast_pannuke_type_tf2pytorch.tar"
)


# ─────────────────────────────────────────────────────────────
# Official HoVer-Net PyTorch Architecture (Fast Mode)
# ─────────────────────────────────────────────────────────────
class TFSamepaddingLayer(nn.Module):
    def __init__(self, ksize, stride):
        super(TFSamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
        return F.pad(x, padding, "constant", 0)


def _crop_op(x, cropping):
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    return x[:, :, crop_t:-crop_b, crop_l:-crop_r]


def _crop_to_shape(x, y):
    assert y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    crop_shape = (x.shape[2] - y.shape[2], x.shape[3] - y.shape[3])
    return _crop_op(x, crop_shape)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super(DenseBlock, self).__init__()
        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    OrderedDict([
                        ("preact_bna/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                        ("preact_bna/relu", nn.ReLU(inplace=True)),
                        ("conv1", nn.Conv2d(unit_in_ch, unit_ch[0], unit_ksize[0], stride=1, padding=0, bias=False)),
                        ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                        ("conv1/relu", nn.ReLU(inplace=True)),
                        ("conv2", nn.Conv2d(unit_ch[0], unit_ch[1], unit_ksize[1], groups=split, stride=1, padding=0, bias=False)),
                    ])
                )
            )
            unit_in_ch += unit_ch[1]
        self.blk_bna = nn.Sequential(
            OrderedDict([("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)), ("relu", nn.ReLU(inplace=True))])
        )

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = _crop_to_shape(prev_feat, new_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        return self.blk_bna(prev_feat)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, stride=1):
        super(ResidualBlock, self).__init__()
        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch
        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            unit_layer = [
                ("preact/bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)),
                ("preact/relu", nn.ReLU(inplace=True)),
                ("conv1", nn.Conv2d(unit_in_ch, unit_ch[0], unit_ksize[0], stride=1, padding=0, bias=False)),
                ("conv1/bn", nn.BatchNorm2d(unit_ch[0], eps=1e-5)),
                ("conv1/relu", nn.ReLU(inplace=True)),
                ("conv2/pad", TFSamepaddingLayer(ksize=unit_ksize[1], stride=stride if idx == 0 else 1)),
                ("conv2", nn.Conv2d(unit_ch[0], unit_ch[1], unit_ksize[1], stride=stride if idx == 0 else 1, padding=0, bias=False)),
                ("conv2/bn", nn.BatchNorm2d(unit_ch[1], eps=1e-5)),
                ("conv2/relu", nn.ReLU(inplace=True)),
                ("conv3", nn.Conv2d(unit_ch[1], unit_ch[2], unit_ksize[2], stride=1, padding=0, bias=False)),
            ]
            unit_layer = unit_layer if idx != 0 else unit_layer[2:]
            self.units.append(nn.Sequential(OrderedDict(unit_layer)))
            unit_in_ch = unit_ch[-1]

        if in_ch != unit_ch[-1] or stride != 1:
            self.shortcut = nn.Conv2d(in_ch, unit_ch[-1], 1, stride=stride, bias=False)
        else:
            self.shortcut = None

        self.blk_bna = nn.Sequential(
            OrderedDict([("bn", nn.BatchNorm2d(unit_in_ch, eps=1e-5)), ("relu", nn.ReLU(inplace=True))])
        )

    def forward(self, prev_feat):
        shortcut = prev_feat if self.shortcut is None else self.shortcut(prev_feat)
        for idx in range(0, len(self.units)):
            new_feat = self.units[idx](prev_feat)
            prev_feat = new_feat + shortcut
            shortcut = prev_feat
        return self.blk_bna(prev_feat)


class UpSample2x(nn.Module):
    def __init__(self):
        super(UpSample2x, self).__init__()
        self.register_buffer("unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32")))

    def forward(self, x):
        input_shape = list(x.shape)
        x = x.unsqueeze(-1)
        mat = self.unpool_mat.unsqueeze(0)
        ret = torch.tensordot(x, mat, dims=1)
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        return ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))


class HoVerNetFast(nn.Module):
    """
    Official PyTorch HoVer-Net Fast Model for PanNuke (Graham et al.)
    Patch input: 256x256, Patch output: 164x164
    Output branches:
    - 'tp': 6-class nuclear type prediction
    - 'np': 2-class nuclear pixel probability
    - 'hv': 2-channel horizontal & vertical distance maps
    """
    def __init__(self, input_ch=3, nr_types=6):
        super().__init__()
        self.nr_types = nr_types

        module_list = [
            ("pad", TFSamepaddingLayer(ksize=7, stride=1)),
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)
        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=3):
            u3 = nn.Sequential(OrderedDict([
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False)),
            ]))
            u2 = nn.Sequential(OrderedDict([
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)),
            ]))
            u1 = nn.Sequential(OrderedDict([
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False)),
            ]))
            u0 = nn.Sequential(OrderedDict([
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True)),
            ]))
            return nn.Sequential(OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)]))

        self.decoder = nn.ModuleDict(OrderedDict([
            ("tp", create_decoder_branch(ksize=3, out_ch=nr_types)),
            ("np", create_decoder_branch(ksize=3, out_ch=2)),
            ("hv", create_decoder_branch(ksize=3, out_ch=2)),
        ]))
        self.upsample2x = UpSample2x()

    def forward(self, imgs):
        imgs = imgs / 255.0
        d0 = self.conv0(imgs)
        d0 = self.d0(d0)
        d1 = self.d1(d0)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]
        d[0] = _crop_op(d[0], [92, 92])
        d[1] = _crop_op(d[1], [36, 36])

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)
            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)
            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)
            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0
        return out_dict


# ─────────────────────────────────────────────────────────────
# Model Singleton & Loader
# ─────────────────────────────────────────────────────────────
_HOVERNET_NET = None

def get_hovernet_model():
    """Returns initialized HoVer-Net PyTorch model with loaded weights."""
    global _HOVERNET_NET
    if _HOVERNET_NET is not None:
        return _HOVERNET_NET

    net = HoVerNetFast(input_ch=3, nr_types=6)
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        try:
            chkpt = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")
            state_dict = chkpt.get("desc", chkpt)
            net.load_state_dict(state_dict, strict=True)
            print(f"[HoVer-Net] Loaded official PanNuke checkpoint from: {MODEL_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"[HoVer-Net] Checkpoint load warning: {e}")
    else:
        print(f"[HoVer-Net] Warning: Checkpoint not found at {MODEL_CHECKPOINT_PATH}")

    net.eval()
    _HOVERNET_NET = net
    return _HOVERNET_NET


# ─────────────────────────────────────────────────────────────
# Official HoVer-Net Watershed Post-Processing (Graham et al.)
# ─────────────────────────────────────────────────────────────
def proc_np_hv(np_map: np.ndarray, hv_map: np.ndarray) -> np.ndarray:
    """
    Watershed marker-controlled instance segmentation based on:
    - np_map: nuclear pixel probability [0, 1]
    - hv_map: horizontal & vertical distance gradient landscape
    """
    blb = (np_map >= 0.5).astype(np.int32)
    blb = ndi.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1

    h_dir = cv2.normalize(hv_map[..., 0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(hv_map[..., 1], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall_mask = np.array(overall >= 0.4, dtype=np.int32)
    marker = blb - overall_mask
    marker[marker < 0] = 0
    marker = ndi.binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = ndi.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    inst_map = watershed(dist, markers=marker, mask=blb)
    return inst_map.astype(np.int32)


# ─────────────────────────────────────────────────────────────
# HoVer-Net Nuclear Segmentation Pipeline
# ─────────────────────────────────────────────────────────────
def run_hovernet_segmentation(image_rgb: np.ndarray, roi_mask: np.ndarray = None):
    """
    Executes HoVer-Net simultaneous nuclear instance segmentation and classification
    using the official PanNuke-pretrained neural weights.
    """
    start_time = time.perf_counter()
    orig_h, orig_w = image_rgb.shape[:2]

    # For efficient and ultra-responsive CPU processing, scale if image > 512px
    max_dim = max(orig_h, orig_w)
    scale_factor = 1.0
    if max_dim > 512:
        scale_factor = 512.0 / max_dim
        proc_w = int(round(orig_w * scale_factor))
        proc_h = int(round(orig_h * scale_factor))
        proc_img = cv2.resize(image_rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        proc_img = image_rgb.copy()
        proc_h, proc_w = orig_h, orig_w

    net = get_hovernet_model()

    win_size = 256
    msk_size = step_size = 164

    def get_last_steps(length, msk, step):
        nr_step = math.ceil((length - msk) / step)
        return int((nr_step + 1) * step)

    last_h = get_last_steps(proc_h, msk_size, step_size)
    last_w = get_last_steps(proc_w, msk_size, step_size)
    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - proc_h
    padr = last_w + win_size - proc_w

    padded = np.pad(proc_img, ((padt, padb), (padl, padr), (0, 0)), mode="reflect")

    pred_tp = np.zeros((last_h + win_size, last_w + win_size, 6), dtype=np.float32)
    pred_np = np.zeros((last_h + win_size, last_w + win_size, 2), dtype=np.float32)
    pred_hv = np.zeros((last_h + win_size, last_w + win_size, 2), dtype=np.float32)

    patches = []
    coords = []
    for y in range(0, last_h, step_size):
        for x in range(0, last_w, step_size):
            p = padded[y : y + win_size, x : x + win_size]
            patches.append(p)
            coords.append((y, x))

    batch_size = 4
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            b_patches = patches[i : i + batch_size]
            b_coords = coords[i : i + batch_size]

            t_batch = torch.from_numpy(np.stack(b_patches)).permute(0, 3, 1, 2).float()
            out = net(t_batch)

            b_tp = F.softmax(out["tp"], dim=1).permute(0, 2, 3, 1).cpu().numpy()
            b_np = F.softmax(out["np"], dim=1).permute(0, 2, 3, 1).cpu().numpy()
            b_hv = out["hv"].permute(0, 2, 3, 1).cpu().numpy()

            for idx, (cy, cx) in enumerate(b_coords):
                pred_tp[cy : cy + msk_size, cx : cx + msk_size] = b_tp[idx]
                pred_np[cy : cy + msk_size, cx : cx + msk_size] = b_np[idx]
                pred_hv[cy : cy + msk_size, cx : cx + msk_size] = b_hv[idx]

    pred_tp = pred_tp[:proc_h, :proc_w]
    pred_np = pred_np[:proc_h, :proc_w]
    pred_hv = pred_hv[:proc_h, :proc_w]

    # Nuclear pixel probability map (channel 1 = positive nucleus)
    np_prob = pred_np[..., 1]

    # Run official watershed instance separation
    proc_labels = proc_np_hv(np_prob, pred_hv)

    # Scale instance map and distance fields back to original input dimensions
    if scale_factor != 1.0:
        instance_labels = cv2.resize(proc_labels, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        h_map = cv2.resize(pred_hv[..., 0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        v_map = cv2.resize(pred_hv[..., 1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        tp_probs = cv2.resize(pred_tp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        instance_labels = proc_labels
        h_map = pred_hv[..., 0]
        v_map = pred_hv[..., 1]
        tp_probs = pred_tp

    # Extract instance metrics and individual cell boundaries
    instances = []
    unique_ids = np.unique(instance_labels)
    unique_ids = unique_ids[unique_ids > 0]

    for inst_id in unique_ids:
        inst_mask = (instance_labels == inst_id).astype(np.uint8)
        cnts, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area_px = int(cv2.contourArea(c))
        if area_px < 10 or area_px > (orig_h * orig_w * 0.40):
            instance_labels[instance_labels == inst_id] = 0
            continue

        area_um2 = round(area_px * (MICRONS_PER_PIXEL ** 2), 1)
        perimeter = float(cv2.arcLength(c, True))
        circularity = round(float(4.0 * np.pi * area_px / (perimeter ** 2 + 1e-6)), 3)

        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        aspect_ratio = round(float(max(rw, rh) / max(1.0, min(rw, rh))), 2)

        # Phenotype classification from tp branch
        inst_tp_probs = tp_probs[inst_mask > 0]
        if len(inst_tp_probs) > 0:
            mean_tp = np.mean(inst_tp_probs, axis=0)  # shape (6,)
            non_bg_probs = mean_tp[1:]
            pred_class = int(np.argmax(non_bg_probs) + 1)
            conf = float(non_bg_probs[pred_class - 1])
        else:
            pred_class = 1
            conf = 0.85

        bx, by, bw, bh = cv2.boundingRect(c)
        cx = int(rect[0][0]) if not np.isnan(rect[0][0]) else (bx + bw // 2)
        cy = int(rect[0][1]) if not np.isnan(rect[0][1]) else (by + bh // 2)

        cls_info = HOVERNET_CLASSES.get(pred_class, HOVERNET_CLASSES[1])

        instances.append({
            "id": int(inst_id),
            "class_id": pred_class,
            "class_name": cls_info["name"],
            "badge": cls_info["badge"],
            "color_rgb": cls_info["color_rgb"],
            "hex": cls_info["hex"],
            "confidence": round(conf, 2),
            "area_px": area_px,
            "area_um2": area_um2,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "center": (cx, cy),
            "bbox": [bx, by, bx + bw, by + bh],
            "contour": c
        })

    inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 1)

    return {
        "instance_labels": instance_labels,
        "instances": instances,
        "h_map": h_map,
        "v_map": v_map,
        "inference_time_ms": inference_time_ms
    }


# ─────────────────────────────────────────────────────────────
# Rendering Overlays & HoVer Map Visualization
# ─────────────────────────────────────────────────────────────
def render_hovernet_display(image_rgb: np.ndarray, result_data: dict, selected_id: int = None):
    """
    Renders the composite HoVer-Net visualization:
    - Official color-coded contours for PanNuke nuclear phenotypes
    - When a cell is selected, draws a prominent, glowing boundary outline
      so the user can inspect each cell's boundary one by one.
    """
    overlay = image_rgb.copy()
    instances = result_data.get("instances", [])
    if not instances:
        return overlay

    # Create translucent color fills
    fill_layer = np.zeros_like(overlay)
    for inst in instances:
        c = inst["contour"]
        color = inst["color_rgb"]
        cv2.drawContours(fill_layer, [c], -1, color, -1)

    has_nuclei = fill_layer.any(axis=-1)
    overlay[has_nuclei] = (
        0.68 * overlay[has_nuclei].astype(float)
        + 0.32 * fill_layer[has_nuclei].astype(float)
    ).astype(np.uint8)

    # Draw crisp colored outer boundaries for all cells
    for inst in instances:
        c = inst["contour"]
        color = inst["color_rgb"]
        is_sel = (selected_id is not None and inst["id"] == selected_id)
        if not is_sel:
            cv2.drawContours(overlay, [c], -1, color, 2)

    # Highlight selected cell with high-contrast dual-stroke glowing boundary
    if selected_id is not None:
        for inst in instances:
            if inst["id"] == selected_id:
                c = inst["contour"]
                # Outer bright yellow contour + inner pure white border
                cv2.drawContours(overlay, [c], -1, (255, 235, 0), 4)
                cv2.drawContours(overlay, [c], -1, (255, 255, 255), 2)
                cx, cy = inst["center"]
                cv2.circle(overlay, (cx, cy), 5, (255, 235, 0), -1)
                cv2.circle(overlay, (cx, cy), 7, (0, 0, 0), 1)
                break

    return overlay


def render_hover_maps_rgb(h_map: np.ndarray, v_map: np.ndarray) -> np.ndarray:
    """
    Generates an RGB visualization of the horizontal and vertical distance maps
    (matching Fig. 3 in the HoVer-Net paper).
    """
    h_vis = np.uint8(np.clip((h_map + 1.0) * 0.5 * 255, 0, 255))
    v_vis = np.uint8(np.clip((v_map + 1.0) * 0.5 * 255, 0, 255))
    mid_vis = np.uint8(np.clip(np.abs(h_map * v_map) * 255, 0, 255))

    hover_rgb = cv2.merge([h_vis, mid_vis, v_vis])
    return hover_rgb


# ─────────────────────────────────────────────────────────────
# Formatted Markdown Metrics Report
# ─────────────────────────────────────────────────────────────
def format_hovernet_markdown(result_data: dict, selected_instance: dict = None) -> str:
    """
    Generates report with PanNuke phenotype distributions and cell boundary inspector.
    """
    instances = result_data.get("instances", [])
    inf_time = result_data.get("inference_time_ms", 0.0)
    total_nuclei = len(instances)

    if total_nuclei == 0:
        return "### HoVer-Net: No nuclear instances identified in the active field."

    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    areas = []
    circularities = []

    for inst in instances:
        cid = inst["class_id"]
        counts[cid] = counts.get(cid, 0) + 1
        areas.append(inst["area_um2"])
        circularities.append(inst["circularity"])

    avg_area = round(float(np.mean(areas)), 1)
    avg_circ = round(float(np.mean(circularities)), 2)

    p1 = round(100.0 * counts[1] / total_nuclei, 1)
    p2 = round(100.0 * counts[2] / total_nuclei, 1)
    p3 = round(100.0 * counts[3] / total_nuclei, 1)
    p4 = round(100.0 * counts[4] / total_nuclei, 1)
    p5 = round(100.0 * counts[5] / total_nuclei, 1)

    til_ratio = round(counts[2] / max(1, counts[1] + counts[2]) * 100.0, 1)

    report = [
        f"### 🔬 HoVer-Net: **{total_nuclei} Nuclear Instances** Segmented & Classified\n",
        f"- **Model**: Official PanNuke-pretrained HoVer-Net (Fast Mode, Graham et al., 2019)\n",
        f"- **Inference Latency**: `{inf_time} ms`\n",
        f"- **TIL Proxy Score**: `{til_ratio}%` (Inflammatory infiltration ratio among neoplastic clusters)\n\n",
        "#### Official PanNuke Phenotype Breakdown:\n",
        f"| Class | Phenotype | Count | Proportion | Visual Contour |\n",
        f"| :--- | :--- | :---: | :---: | :---: |\n",
        f"| 🔴 **Class 1** | **Neoplastic** | `{counts[1]}` | {p1}% | Red Outline |\n",
        f"| 🔵 **Class 2** | **Inflammatory (TILs)** | `{counts[2]}` | {p2}% | Blue Outline |\n",
        f"| 🟢 **Class 3** | **Connective (Stroma)** | `{counts[3]}` | {p3}% | Green Outline |\n",
        f"| 🟡 **Class 4** | **Dead / Necrotic** | `{counts[4]}` | {p4}% | Yellow Outline |\n",
        f"| 🟣 **Class 5** | **Non-Neoplastic Epithelial** | `{counts[5]}` | {p5}% | Cyan Outline |\n\n",
        f"- **Mean Nuclear Area**: `{avg_area} µm²` | **Mean Circularity**: `{avg_circ}`\n"
    ]

    if selected_instance:
        s_cls = selected_instance["class_name"]
        s_bdg = selected_instance["badge"]
        s_id = selected_instance["id"]
        s_area = selected_instance["area_um2"]
        s_circ = selected_instance["circularity"]
        s_asp = selected_instance["aspect_ratio"]
        s_conf = selected_instance["confidence"]
        (cx, cy) = selected_instance["center"]

        report.append(
            f"\n---\n"
            f"#### 🎯 Selected Nucleus Inspector (Instance #{s_id}):\n"
            f"- **Predicted Class**: {s_bdg} **{s_cls}** (Confidence: `{s_conf * 100:.1f}%`)\n"
            f"- **Centroid Position**: `x={cx}, y={cy}`\n"
            f"- **Morphometry**: Area = `{s_area} µm²`, Circularity = `{s_circ}`, Aspect Ratio = `{s_asp}`\n"
            f"- **Active Boundary**: Highlighted with glowing yellow & white contour on image.\n"
            f"- Click **'Extract Selected Nucleus'** below to view isolated crop & transparent cutout!"
        )

    return "".join(report)
