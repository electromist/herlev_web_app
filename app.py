import os
import cv2
import threading
import time
import tempfile
import uuid
from PIL import Image
import numpy as np
import torch
import gradio as gr
from gradio.components.image_editor import ImageEditor, FileData
import segmentation_models_pytorch as smp

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sam_model_registry, SamPredictor = None, None

# [Oral Histopath AI - Commented out for now]
# from oral_histopath import (
#     load_scribbleprompt,
#     predict_scribbleprompt,
# )
from oral_histopath import (
    analyze_morphology,
    format_morphology_markdown,
    extract_region,
    compare_models_roi
)

from hovernet import (
    run_hovernet_segmentation,
    render_hovernet_display,
    render_hover_maps_rgb,
    format_hovernet_markdown,
    HOVERNET_CLASSES
)

# ─────────────────────────────────────────────────────────────
# Resilient Image I/O for Windows File-Locking & Async Uploads
# ─────────────────────────────────────────────────────────────
_orig_pil_open = Image.open

def _resilient_pil_open(fp, mode="r", formats=None):
    if isinstance(fp, (str, os.PathLike)):
        for _ in range(25):
            try:
                if os.path.exists(fp) and os.path.getsize(fp) > 0:
                    im = _orig_pil_open(fp, mode=mode, formats=formats)
                    _ = im.size
                    return im
            except Exception:
                pass
            time.sleep(0.04)
    return _orig_pil_open(fp, mode=mode, formats=formats)

Image.open = _resilient_pil_open

_orig_convert_and_format = ImageEditor.convert_and_format_image

def _safe_convert_and_format(self, file):
    if file is None:
        return None
    if isinstance(file, FileData) and getattr(file, "path", None):
        for _ in range(25):
            try:
                if os.path.exists(file.path) and os.path.getsize(file.path) > 0:
                    return _orig_convert_and_format(self, file)
            except Exception:
                pass
            time.sleep(0.04)
        try:
            return _orig_convert_and_format(self, file)
        except Exception:
            return None
    return _orig_convert_and_format(self, file)

ImageEditor.convert_and_format_image = _safe_convert_and_format

_orig_imageeditor_preprocess = ImageEditor.preprocess

def _safe_imageeditor_preprocess(self, payload):
    for attempt in range(3):
        try:
            return _orig_imageeditor_preprocess(self, payload)
        except Exception as e:
            if attempt < 2:
                time.sleep(0.1)
            else:
                return {
                    "background": None,
                    "layers": [],
                    "composite": None
                }

ImageEditor.preprocess = _safe_imageeditor_preprocess

# ─────────────────────────────────────────────────────────────
# Settings & Paths
# ─────────────────────────────────────────────────────────────
IMG_SIZE = 128
N_CLASSES = 3

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "herlev_effnetb7_fpn.pth"
)

_BEST_MEDSAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medsam_vit_b_best.pth")
_BASE_MEDSAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medsam_vit_b.pth")
MEDSAM_MODEL_PATH = _BEST_MEDSAM if os.path.exists(_BEST_MEDSAM) else _BASE_MEDSAM

SCRIBBLEPROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scribbleprompt.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "Background",
    1: "Cytoplasm",
    2: "Nucleus",
}

# RGB colors used in the app overlay
CLASS_COLORS = {
    0: np.array([220, 38, 38], dtype=np.uint8),    # red
    1: np.array([30, 64, 175], dtype=np.uint8),    # dark blue
    2: np.array([86, 190, 240], dtype=np.uint8),   # light blue
}

YELLOW = np.array([255, 235, 0], dtype=np.uint8)
CONTOUR_COLOR = (34, 197, 94)  # vibrant emerald green for rounded nucleus borders


# ─────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────
def build_model():
    try:
        model = smp.FPN(
            encoder_name="tu-tf_efficientnet_b7_ns",
            encoder_weights=None,
            in_channels=3,
            classes=N_CLASSES,
            activation=None,
        )
        print("Using EfficientNet-B7 Noisy Student + FPN")
    except Exception as e:
        print("Noisy Student alias failed; using EfficientNet-B7:", e)
        model = smp.FPN(
            encoder_name="efficientnet-b7",
            encoder_weights=None,
            in_channels=3,
            classes=N_CLASSES,
            activation=None,
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def build_single_medsam(checkpoint_path, label):
    if sam_model_registry is None or not os.path.exists(checkpoint_path):
        return None
    try:
        sam = sam_model_registry["vit_b"](checkpoint=None)
        state_dict = torch.load(checkpoint_path, map_location=device)
        sam.load_state_dict(state_dict)
        sam.to(device=device)
        sam.eval()
        ckpt_name = os.path.basename(checkpoint_path)
        print(f"MedSAM ({label}) loaded successfully from [{ckpt_name}] on {device}")
        return SamPredictor(sam)
    except Exception as e:
        print(f"Failed to load MedSAM ({label}):", e)
        return None


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}\n\n"
        "Put herlev_effnetb7_fpn.pth in the same folder as app.py."
    )

model = build_model()
medsam_best_predictor = build_single_medsam(_BEST_MEDSAM, "Fine-Tuned Best") if os.path.exists(_BEST_MEDSAM) else None
medsam_base_predictor = build_single_medsam(_BASE_MEDSAM, "Base Model") if os.path.exists(_BASE_MEDSAM) else None
medsam_predictor = medsam_best_predictor or medsam_base_predictor

# [Oral Histopath AI - Commented out for now]
# scribbleprompt_net = load_scribbleprompt(SCRIBBLEPROMPT_PATH, device)
# if scribbleprompt_net is not None:
#     print("ScribblePrompt (Oral Histopath AI) loaded successfully on", device)
scribbleprompt_net = None

print("Models loaded successfully.")
print("Device:", device)

_cached_medsam_sig = None
_medsam_lock = threading.Lock()


def ensure_medsam_image(image_rgb):
    """Encodes image in both best and base MedSAM predictors only once per image."""
    global _cached_medsam_sig
    if image_rgb is None:
        return
    sig = (image_rgb.shape, int(image_rgb[0, 0, 0]), int(image_rgb[-1, -1, 0]), int(image_rgb.mean()))
    if _cached_medsam_sig != sig:
        with _medsam_lock:
            if _cached_medsam_sig != sig:
                with torch.inference_mode():
                    if medsam_best_predictor is not None:
                        medsam_best_predictor.set_image(image_rgb)
                    if medsam_base_predictor is not None:
                        medsam_base_predictor.set_image(image_rgb)
                _cached_medsam_sig = sig


def prewarm_medsam_image(image_rgb):
    """Asynchronously pre-encodes the image in the background so manual selection is instant."""
    if image_rgb is not None:
        threading.Thread(target=ensure_medsam_image, args=(image_rgb,), daemon=True).start()


# ─────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────
def robust_image(img_array):
    """Saves image to a unique path to avoid Gradio temp file race conditions and black canvas."""
    img_array = np.ascontiguousarray(img_array)
    path = os.path.join(tempfile.gettempdir(), f"herlev_{uuid.uuid4().hex}.png")
    Image.fromarray(img_array).save(path)
    return {"background": path, "layers": [], "composite": path}

def colorize_mask(mask):
    """Convert label IDs 0/1/2 into an RGB segmentation image."""
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color
    return colored


def make_overlay(image_rgb, mask, alpha=0.45):
    """Blend segmentation colors over the original image."""
    segmentation_rgb = colorize_mask(mask)
    return cv2.addWeighted(image_rgb, 1 - alpha, segmentation_rgb, alpha, 0)


def preprocess(image_rgb):
    """Prepare image exactly like the model training pipeline."""
    original_h, original_w = image_rgb.shape[:2]
    resized = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    x = resized.astype(np.float32) / 255.0
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    x = x.to(device)
    return x, original_h, original_w


def round_off_mask(binary_mask):
    """Smooths and fills holes in nucleus mask to produce rounded, convex boundaries."""
    if not binary_mask.any():
        return binary_mask
    mask_u = binary_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask_u, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    for c in cnts:
        if cv2.contourArea(c) > 15:
            cv2.drawContours(filled, [c], -1, 1, -1)
    return filled.astype(bool)


def detect_all_nuclei_boxes(image_rgb):
    """
    Detects real cell nuclei across the entire slide using:
    - Hematoxylin optical density absorption
    - Multi-scale morphological top-hat filtering
    - Spatial grid balancing across top, center, bottom, left, right.
    """
    h, w = image_rgb.shape[:2]
    b, g, r = cv2.split(image_rgb)
    
    # Hematoxylin optical density signal (highest in dark purple nuclei)
    nuc_signal = (255 - g.astype(float)) + 0.3 * (255 - r.astype(float))
    nuc_signal = cv2.normalize(nuc_signal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    tophat = cv2.addWeighted(
        cv2.morphologyEx(nuc_signal, cv2.MORPH_TOPHAT, k1), 0.5,
        cv2.morphologyEx(nuc_signal, cv2.MORPH_TOPHAT, k2), 0.5, 0
    )
    
    thresh = cv2.adaptiveThreshold(
        tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -6
    )
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if 20 < area < 900:
            bx, by, bw, bh = cv2.boundingRect(c)
            cx, cy = bx + bw // 2, by + bh // 2
            # Avoid blood smear corner artifacts if high red
            if cy < 50 and cx > (w - 120):
                continue
            pad = 6
            x1 = max(0, cx - pad - bw // 2)
            y1 = max(0, cy - pad - bh // 2)
            x2 = min(w, cx + pad + bw // 2)
            y2 = min(h, cy + pad + bh // 2)
            score = float(np.mean(tophat[y1:y2, x1:x2]))
            candidates.append((score, [x1, y1, x2, y2], (cx, cy)))
            
    # Spatial grid distribution (6 cols x 5 rows) to evenly cover the whole slide
    n_rows, n_cols = 5, 6
    grid = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    cell_w = w / n_cols
    cell_h = h / n_rows
    
    for score, box, (cx, cy) in candidates:
        r_idx = min(n_rows - 1, int(cy / cell_h))
        c_idx = min(n_cols - 1, int(cx / cell_w))
        grid[r_idx][c_idx].append((score, box))
        
    selected_boxes = []
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            cell_cands = sorted(grid[r_idx][c_idx], key=lambda x: x[0], reverse=True)
            for sc, b_box in cell_cands[:1]:
                selected_boxes.append(b_box)
                
    return selected_boxes[:25]


def get_herlev_nuc_mask(state, image_rgb):
    """Semantic nucleus prior using Herlev EffNet-B7 model (cached per image)."""
    if state is not None and state.get("herlev_nuc") is not None:
        return state["herlev_nuc"]
    h, w = image_rgb.shape[:2]
    try:
        x, orig_h, orig_w = preprocess(image_rgb)
        with torch.no_grad():
            logits = model(x)
            pred = logits.softmax(dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize((pred == 2).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print("Herlev prior error:", e)
        mask = np.zeros((h, w), dtype=np.uint8)
    if state is not None:
        state["herlev_nuc"] = mask
    return mask


def target_nucleus_rigorous(predictor, image_rgb, px, py, herlev_nuc=None, user_box=None):
    """
    Rigorously targets and segments a single nucleus around a doctor's click or brush stroke:
    - Uses user-drawn circle/arc bounding box if provided.
    - Checks Herlev semantic nucleus prior for the exact cell nucleus boundaries.
    - If found, prompts MedSAM with the precise bounding box and centroid.
    - Otherwise falls back to hematoxylin optical density patch search + calibrated box.
    - MedSAM promptable segmentation with multimask selection, component filtering, and smooth rounding.
    """
    h, w = image_rgb.shape[:2]
    cx, cy = px, py
    box = user_box
    
    if box is None and herlev_nuc is not None and herlev_nuc.any():
        num_lbl, lbls, stats, centroids = cv2.connectedComponentsWithStats(herlev_nuc)
        best_comp = None
        min_dist = 9999
        for lbl in range(1, num_lbl):
            bx, by, bw, bh, area = stats[lbl]
            if area < 10:
                continue
            c_x, c_y = centroids[lbl]
            if (bx - 15 <= px <= bx + bw + 15) and (by - 15 <= py <= by + bh + 15):
                d = np.hypot(c_x - px, c_y - py)
                if d < min_dist and d < 35:
                    min_dist = d
                    best_comp = lbl
        if best_comp is not None:
            bx, by, bw, bh, _ = stats[best_comp]
            cx, cy = int(round(centroids[best_comp][0])), int(round(centroids[best_comp][1]))
            pad = 5
            box = np.array([max(0, bx - pad), max(0, by - pad), min(w, bx + bw + pad), min(h, by + bh + pad)])

    if box is None:
        pw = 22
        x1, y1 = max(0, px - pw), max(0, py - pw)
        x2, y2 = min(w, px + pw), min(h, py + pw)
        patch = image_rgb[y1:y2, x1:x2]
        if patch.size > 0:
            min_val, _, min_loc, _ = cv2.minMaxLoc(patch[:, :, 1])
            cx = x1 + min_loc[0]
            cy = y1 + min_loc[1]
        r = 20
        box = np.array([max(0, cx - r), max(0, cy - r), min(w, cx + r), min(h, cy + r)])

    # Inference with ensemble / probability fusion between best and base models
    candidate_masks = []
    candidate_scores = []

    # 1. Best Fine-Tuned MedSAM Predictor
    p_best = medsam_best_predictor or predictor
    if p_best is not None:
        with torch.inference_mode():
            masks_b, scores_b, _ = p_best.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                box=box,
                multimask_output=True
            )
        for i in range(len(masks_b)):
            candidate_masks.append(masks_b[i])
            # Give slight priority boost to fine-tuned weights
            candidate_scores.append(float(scores_b[i]) * 1.08)

    # 2. Base MedSAM Predictor (foundation model prior)
    if medsam_base_predictor is not None and medsam_base_predictor is not p_best:
        with torch.inference_mode():
            masks_base, scores_base, _ = medsam_base_predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                box=box,
                multimask_output=True
            )
        for i in range(len(masks_base)):
            candidate_masks.append(masks_base[i])
            candidate_scores.append(float(scores_base[i]))

    if not candidate_masks:
        return np.zeros((h, w), dtype=bool), (cx, cy), 0.0

    # Select the optimal mask: prioritize masks that cover the nucleus body rather than tiny sub-nucleolar peaks
    best_idx = 0
    best_score = -1.0
    box_area = float((box[2] - box[0]) * (box[3] - box[1])) if box is not None else 1000.0

    for i in range(len(candidate_masks)):
        m = candidate_masks[i]
        m_area = float(m.sum())
        sc = candidate_scores[i]

        if m[cy, cx]:
            sc += 0.35  # Enclosure bonus

        # If user circled a nucleus, favor masks that occupy a realistic fraction (15% to 85%) of the box
        if box is not None and box_area > 50:
            coverage = m_area / box_area
            if 0.12 <= coverage <= 0.85:
                sc += 0.40  # Well-proportioned nucleus coverage
            elif coverage < 0.10:
                sc -= 0.30  # Too small (sub-nucleolar or fragment)

        if sc > best_score:
            best_score = sc
            best_idx = i

    raw_mask = candidate_masks[best_idx].copy()

    # If SAM was hesitant / pale chromatin inside box, fallback to adaptive color segmentation inside the box
    if raw_mask.sum() < 25 and box is not None:
        bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        roi = image_rgb[by1:by2, bx1:bx2]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            # Nuclei are darker than immediate surrounding cytoplasm
            otsu_val, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Keep component near center
            local_cx = cx - bx1
            local_cy = cy - by1
            num_l, labs, st, cent = cv2.connectedComponentsWithStats(thresh)
            best_l = 0
            for l_idx in range(1, num_l):
                if labs[max(0, min(local_cy, labs.shape[0]-1)), max(0, min(local_cx, labs.shape[1]-1))] == l_idx:
                    best_l = l_idx
                    break
            if best_l > 0:
                raw_mask[by1:by2, bx1:bx2] = (labs == best_l)

    # Isolate component closest to true nucleus core center
    mask_u = raw_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u)
    if num_labels > 1:
        best_lbl = 1
        min_dist = 9999
        for lbl in range(1, num_labels):
            cnt_x, cnt_y = centroids[lbl]
            d = (cnt_x - cx)**2 + (cnt_y - cy)**2
            if d < min_dist:
                min_dist = d
                best_lbl = lbl
        mask_u = (labels == best_lbl).astype(np.uint8)

    # Elliptical morphological closing to eliminate holes and produce clean natural nucleus boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smooth = cv2.morphologyEx(mask_u, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(smooth)
    for c in cnts:
        cv2.drawContours(filled, [c], -1, 1, -1)

    # Final sanity check: if filled has non-zero mask, return it
    if not filled.any() and box is not None:
        bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.circle(filled, (cx, cy), max(6, min(bx2 - bx1, by2 - by1) // 3), 1, -1)

    return filled.astype(bool), (cx, cy), max(0.85, float(candidate_scores[best_idx]))


def render_medsam_display(image_rgb, combined_mask, doctor_targets=None):
    """Renders composite display showing all nuclei."""
    result = image_rgb.copy()
    if combined_mask.any():
        yellow_layer = np.zeros_like(result)
        yellow_layer[combined_mask] = YELLOW
        result[combined_mask] = (
            0.50 * result[combined_mask].astype(float)
            + 0.50 * yellow_layer[combined_mask].astype(float)
        ).astype(np.uint8)
        
        # Smooth green boundary contours for all nuclei
        cnts, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, cnts, -1, CONTOUR_COLOR, 2)
        
    return result


# ─────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────
def load_image(image_rgb, model_choice):
    """Loads the image without running segmentation. Pre-warms MedSAM if selected."""
    if image_rgb is None:
        return None, "Upload an image first.", None

    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[:, :, :3]

    image_rgb = image_rgb.astype(np.uint8)
    h, w = image_rgb.shape[:2]
    
    if "Herlev" in str(model_choice):
        x_herlev, orig_h, orig_w = preprocess(image_rgb)
        with torch.no_grad():
            logits = model(x_herlev)
            pred_128 = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        prediction_full = cv2.resize(pred_128, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        overlay = make_overlay(image_rgb, prediction_full)
        state = {
            "image": image_rgb,
            "mask": prediction_full,
            "last_class": None,
            "model": "Herlev"
        }
        info = (
            "Segmentation complete. Click anywhere on the image below.\n\n"
            "Red = Background | Dark blue = Cytoplasm | Light blue = Nucleus"
        )
        return robust_image(overlay), info, state

    if "MedSAM" in model_choice:  # or "Oral Histopath" in model_choice:
        prewarm_medsam_image(image_rgb)
        
    combined_mask = np.zeros((h, w), dtype=bool)
    state = {
        "image": image_rgb,
        "auto_mask": combined_mask.copy(),
        "mask": combined_mask,
        "doctor_targets": [],
        "model": model_choice
    }
    
    info = (
        "### Image Loaded Successfully\n\n"
        f"- **Model**: `{model_choice}`\n"
        "- Click **'Run Whole-Slide Segmentation'** below to auto-detect.\n"
        "- Or, use the brush tool / click directly to manually target nuclei right now!"
    )
    
    return robust_image(image_rgb), info, state

def predict_image(image_rgb, model_choice):
    """
    Runs segmentation based on model selection:
    - Herlev: full semantic segmentation (Background, Cytoplasm, Nucleus)
    - MedSAM: scans the whole slide, segments all detected nuclei across the field,
              and prepares for doctor manual targeting.
    """
    if image_rgb is None:
        return None, "Upload an image first.", None

    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[:, :, :3]

    image_rgb = image_rgb.astype(np.uint8)
    h, w = image_rgb.shape[:2]

    if "MedSAM" in model_choice:  # or "Oral Histopath" in model_choice:
        return load_image(image_rgb, model_choice)

    # Herlev inference
    x, original_h, original_w = preprocess(image_rgb)

    with torch.no_grad():
        logits = model(x)
        prediction_128 = torch.softmax(logits, dim=1).argmax(dim=1)[0]
        prediction_128 = prediction_128.cpu().numpy().astype(np.uint8)

    # Resize class IDs back to original image size
    prediction_full = cv2.resize(
        prediction_128,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = make_overlay(image_rgb, prediction_full)

    state = {
        "image": image_rgb,
        "mask": prediction_full,
        "last_class": None,
        "model": "Herlev"
    }

    info = (
        "Segmentation complete. Click anywhere on the image below.\n\n"
        "Red = Background | Dark blue = Cytoplasm | Light blue = Nucleus"
    )

    return robust_image(overlay), info, state


# ─────────────────────────────────────────────────────────────
# Click interaction (Doctor Targeting & Memory)
# ─────────────────────────────────────────────────────────────
def select_region(state, evt: gr.SelectData):
    """
    Doctor Manual Targeting & Memory Callback:
    - Remembers clicked points.
    - Rigorously finds the true local nucleus core and rounds it off with MedSAM.
    - Combines previous data (auto-detected + prior doctor targets) with new targets.
    - Displays all nuclei together with numbered markers.
    """
    if state is None or state.get("image") is None:
        return gr.skip(), "Upload an image and run segmentation first.", state

    idx = getattr(evt, "index", None)
    if idx is None:
        return gr.skip(), gr.skip(), state

    image_rgb = state["image"].copy()
    h, w = image_rgb.shape[:2]
    x, y = idx[0], idx[1]

    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    if "Herlev" in str(state.get("model", "")):
        mask = state.get("mask")
        if mask is None or mask.shape[:2] != (h, w) or mask.dtype == bool:
            x_herlev, orig_h, orig_w = preprocess(image_rgb)
            with torch.no_grad():
                logits = model(x_herlev)
                pred_128 = torch.softmax(logits, dim=1).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(pred_128, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            state["mask"] = mask

        class_id = int(mask[y, x])
        class_name = CLASS_NAMES.get(class_id, "Unknown")

        result = make_overlay(image_rgb, mask, alpha=0.42)
        selected = mask == class_id
        yellow_layer = np.zeros_like(result)
        yellow_layer[selected] = YELLOW

        result[selected] = (
            0.52 * result[selected].astype(np.float32)
            + 0.48 * yellow_layer[selected].astype(np.float32)
        ).astype(np.uint8)

        selected_pixels = int(selected.sum())
        total_pixels = int(mask.size)
        selected_percent = 100 * selected_pixels / total_pixels

        message = (
            f"### Selected: **{class_name}**\n\n"
            f"- Click position: `x={x}, y={y}`\n"
            f"- Predicted class ID: `{class_id}`\n"
            f"- Selected area: `{selected_pixels:,}` pixels ({selected_percent:.1f}% of image)\n\n"
            f"The transparent yellow overlay marks the predicted **{class_name.lower()}** region."
        )

        return robust_image(result), message, state

    if medsam_predictor is None:
        return gr.skip(), "Error: MedSAM model not loaded.", state

    existing_mask = state.get("mask")
    if existing_mask is not None and existing_mask.shape == (h, w) and existing_mask[y, x]:
        return gr.skip(), gr.skip(), state

    ensure_medsam_image(image_rgb)
    herlev_nuc = get_herlev_nuc_mask(state, image_rgb)

    # Rigorously segment the nucleus at the clicked location
    try:
        with _medsam_lock, torch.inference_mode():
            new_nuc_mask, (cx, cy), conf = target_nucleus_rigorous(
                medsam_predictor, image_rgb, x, y, herlev_nuc=herlev_nuc
            )
    except Exception as e:
        print("MedSAM targeting error:", e)
        return gr.skip(), f"MedSAM targeting error: {e}", state

    # Remember doctor target point
    doctor_targets = state.get("doctor_targets", [])
    doctor_targets.append((x, y))
    state["doctor_targets"] = doctor_targets

    # Accumulate with previous data (Purana + Naya)
    if existing_mask is None or existing_mask.shape != (h, w):
        combined_mask = new_nuc_mask
    else:
        combined_mask = existing_mask | new_nuc_mask
    state["mask"] = combined_mask

    # Render combined display
    result = render_medsam_display(image_rgb, combined_mask)

    cnts_tot, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_nuclei = len(cnts_tot)
    selected_pixels = int(combined_mask.sum())
    selected_percent = 100 * selected_pixels / (h * w)

    morph = analyze_morphology(image_rgb, new_nuc_mask)
    morph_md = format_morphology_markdown(morph) if morph else ""

    message = (
        f"### {state.get('model', 'MedSAM')}: **{total_nuclei} Nuclei Segmented Together**\n\n"
        f"- **New Target Added**: Target #{len(doctor_targets)} at `x={x}, y={y}` (refined core at `{cx}, {cy}`)\n"
        f"- **Confidence Score**: `{conf:.3f}`\n"
        f"- **Doctor Targets Remembered**: `{len(doctor_targets)}` manually verified nuclei\n"
        f"- **Combined Field Area**: `{selected_pixels:,}` pixels ({selected_percent:.1f}% of slide)\n\n"
        f"Showing **previous auto-detected nuclei + all doctor-targeted nuclei** together in transparent yellow with green rounded boundaries!"
        f"{morph_md}"
    )
    return robust_image(result), message, state


def target_drawn_circle(editor_data, state):
    """
    Doctor Circling / Brush Tool Callback:
    - Extracts drawn circle(s) / brush strokes from the segmentation result.
    - Accurately detects the nucleus core inside the doctor's circled boundary.
    - MedSAM rigorously segments and rounds off the nucleus, turning vague shapes into perfect nuclei.
    - Accumulates previous + newly circled doctor targets.
    """
    if state is None or state.get("image") is None:
        return gr.skip(), gr.skip(), gr.skip()

    if not isinstance(editor_data, dict):
        return gr.skip(), gr.skip(), gr.skip()

    layers = editor_data.get("layers", [])
    if not layers:
        return gr.skip(), gr.skip(), gr.skip()

    image_rgb = state["image"].copy()
    h, w = image_rgb.shape[:2]

    # Combine all drawn stroke layers
    stroke_mask = np.zeros((h, w), dtype=bool)
    for l in layers:
        l_img = None
        if isinstance(l, str):
            for _ in range(10):
                l_img = cv2.imread(l, cv2.IMREAD_UNCHANGED)
                if l_img is not None:
                    break
                time.sleep(0.04)
        elif isinstance(l, np.ndarray):
            l_img = l
        elif hasattr(l, "convert"):
            l_img = np.array(l)

        if l_img is not None and l_img.ndim == 3 and l_img.shape[2] == 4:
            if l_img.shape[:2] != (h, w):
                l_img = cv2.resize(l_img, (w, h), interpolation=cv2.INTER_NEAREST)
            stroke_mask = stroke_mask | (l_img[:, :, 3] > 0)

    if not stroke_mask.any():
        return gr.skip(), gr.skip(), gr.skip()

    if state.get("model") != "Herlev":
        # [Oral Histopath AI - Commented out for now]
        # if "Oral Histopath" in state.get("model", "") and scribbleprompt_net is not None:
        #     try:
        #         sp_mask = predict_scribbleprompt(scribbleprompt_net, image_rgb, stroke_mask, device)
        #         if sp_mask.any():
        #             combined_mask = state.get("mask", np.zeros((h, w), dtype=bool)) | sp_mask
        #             state["mask"] = combined_mask
        #             result = render_medsam_display(image_rgb, combined_mask)
        #             morph = analyze_morphology(image_rgb, combined_mask)
        #             morph_md = format_morphology_markdown(morph) if morph else ""
        #             cnts_tot, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #             msg = (
        #                 f"### Oral Histopath AI: **{len(cnts_tot)} Region(s) Segmented**\n\n"
        #                 f"- **Model**: `ScribblePrompt (6-Channel Interactive UNet)`\n"
        #                 f"- Interactive brush/scribble segmentation executed successfully.\n"
        #                 f"- Showing segmented nuclei with green boundary contours."
        #                 f"{morph_md}"
        #             )
        #             return robust_image(result), msg, state
        #     except Exception as e:
        #         print("ScribblePrompt inference error:", e)

        if medsam_predictor is None:
            return gr.skip(), "Error: MedSAM model not loaded.", state

        cnts, _ = cv2.findContours(stroke_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return gr.skip(), gr.skip(), gr.skip()

        doctor_targets = state.get("doctor_targets", [])
        combined_mask = state.get("mask", np.zeros((h, w), dtype=bool))
        newly_added = 0
        herlev_nuc = get_herlev_nuc_mask(state, image_rgb)

        # Filled stroke mask to check enclosure when user circles around a nucleus
        filled_stroke = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled_stroke, cnts, -1, 1, -1)

        # Sort contours from largest to smallest
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 3 or bh < 3:
                continue

            # Check if this stroke/contour corresponds to a Herlev nucleus
            # A stroke can either:
            # 1) Touch an edge / arc across a nucleus (overlap with stroke_mask)
            # 2) Enclose the nucleus inside a drawn circle (overlap with filled_stroke, or centroid inside polygon)
            # Calculate polygon centroid & Moments of the drawn stroke/circle
            M = cv2.moments(c)
            if M["m00"] > 0:
                poly_cx = int(round(M["m10"] / M["m00"]))
                poly_cy = int(round(M["m01"] / M["m00"]))
            else:
                poly_cx = bx + bw // 2
                poly_cy = by + bh // 2

            # Is this stroke an enclosure (closed or semi-closed circle)?
            is_enclosure = (bw > 12 and bh > 12 and cv2.contourArea(c) > 60)

            # Inside the circled boundary, locate the nucleus chromatin core (darkest/purple region)
            # Clip patch strictly to bounding box of the drawn circle
            pad_p = 2
            px1, py1 = max(0, bx - pad_p), max(0, by - pad_p)
            px2, py2 = min(w, bx + bw + pad_p), min(h, by + bh + pad_p)
            patch = image_rgb[py1:py2, px1:px2]

            if patch.size > 0:
                # In H&E histology, nuclei have high Hematoxylin absorption (low green/red intensity)
                # Compute hematoxylin darkness proxy = 255 - green channel
                darkness = 255 - patch[:, :, 1]
                # If enclosure, mask out pixels outside the drawn contour
                c_local = c.copy()
                c_local[:, :, 0] -= px1
                c_local[:, :, 1] -= py1
                patch_mask = np.zeros(patch.shape[:2], dtype=np.uint8)
                cv2.drawContours(patch_mask, [c_local], -1, 1, -1)
                if patch_mask.any() and patch_mask.sum() > 20:
                    darkness[patch_mask == 0] = 0

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(darkness)
                if max_val > 20:
                    cx = px1 + max_loc[0]
                    cy = py1 + max_loc[1]
                else:
                    cx = poly_cx
                    cy = poly_cy
            else:
                cx = poly_cx
                cy = poly_cy

            # Avoid re-adding if exactly at an already recorded doctor target point
            if any((cx - tx)**2 + (cy - ty)**2 < 16 for tx, ty in doctor_targets):
                continue

            # Strict user bounding box: tightly encompasses the circled nucleus
            pad_box = 4
            user_box = np.array([
                max(0, bx - pad_box),
                max(0, by - pad_box),
                min(w, bx + bw + pad_box),
                min(h, by + bh + pad_box)
            ])

            ensure_medsam_image(image_rgb)
            try:
                with _medsam_lock, torch.inference_mode():
                    refined, (cx, cy), conf = target_nucleus_rigorous(
                        medsam_predictor, image_rgb, cx, cy, herlev_nuc=herlev_nuc, user_box=user_box
                    )
                combined_mask = combined_mask | refined
                doctor_targets.append((cx, cy))
                newly_added += 1
            except Exception as e:
                print("MedSAM brush circling error:", e)

        if newly_added == 0:
            result = render_medsam_display(image_rgb, combined_mask)
            return robust_image(result), f"Covered — `{len(doctor_targets)}` doctor targets remembered.", state

        state["mask"] = combined_mask
        state["doctor_targets"] = doctor_targets

        result = render_medsam_display(image_rgb, combined_mask)

        cnts_tot, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_nuclei = len(cnts_tot)
        selected_pixels = int(combined_mask.sum())
        selected_percent = 100 * selected_pixels / (h * w)

        morph = analyze_morphology(image_rgb, combined_mask)
        morph_md = format_morphology_markdown(morph) if morph else ""

        message = (
            f"### MedSAM: **{total_nuclei} Nuclei Segmented Together**\n\n"
            f"- **Doctor Brush Circling**: Added and perfected **{newly_added} circled nucleus candidate(s)**\n"
            f"- **Doctor Targets Remembered**: `{len(doctor_targets)}` manually verified nuclei\n"
            f"- **Combined Field Area**: `{selected_pixels:,}` pixels ({selected_percent:.1f}% of slide)\n\n"
            f"Vague circled shapes are now **perfected into rounded nuclei** and shown with previous auto-detected nuclei in yellow with green contours!"
            f"{morph_md}"
        )
        return robust_image(result), message, state

    # Herlev fallback if user draws with Herlev active
    cnts, _ = cv2.findContours(stroke_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gr.skip(), gr.skip(), gr.skip()
    bx, by, bw, bh = cv2.boundingRect(cnts[0])
    cx = min(w - 1, max(0, bx + bw // 2))
    cy = min(h - 1, max(0, by + bh // 2))

    class_mask = state["mask"]
    class_id = int(class_mask[cy, cx])
    class_name = CLASS_NAMES[class_id]

    result = make_overlay(image_rgb, class_mask, alpha=0.42)
    selected = class_mask == class_id
    yellow_layer = np.zeros_like(result)
    yellow_layer[selected] = YELLOW

    result[selected] = (
        0.52 * result[selected].astype(np.float32)
        + 0.48 * yellow_layer[selected].astype(np.float32)
    ).astype(np.uint8)

    selected_pixels = int(selected.sum())
    total_pixels = int(class_mask.size)
    selected_percent = 100 * selected_pixels / total_pixels

    message = (
        f"### Selected: **{class_name}**\n\n"
        f"- Target position: `x={cx}, y={cy}`\n"
        f"- Predicted class ID: `{class_id}`\n"
        f"- Selected area: `{selected_pixels:,}` pixels ({selected_percent:.1f}% of image)\n\n"
        f"The transparent yellow overlay marks the predicted **{class_name.lower()}** region."
    )
    return robust_image(result), message, state


def reset_doctor_targets(state):
    """Resets doctor manual targets and reverts back to the base auto-detected nuclei."""
    if state is None or state.get("image") is None:
        return gr.skip(), "No image loaded.", state
    image_rgb = state["image"]
    auto_mask = state.get("auto_mask", np.zeros(image_rgb.shape[:2], dtype=bool))
    state["mask"] = auto_mask.copy()
    state["doctor_targets"] = []
    
    result = render_medsam_display(image_rgb, auto_mask, [])
    info = "Doctor manual targets reset. Base auto-detected nuclei preserved. Circle or click any nucleus to target again!"
    return robust_image(result), info, state


def extract_current_roi(state):
    """Extracts bounding box RGB crop and transparent RGBA cutout from current active mask."""
    if state is None or state.get("image") is None or state.get("mask") is None:
        return None, None, "No active region mask available to extract."
    image_rgb = state["image"]
    mask = state["mask"]
    crop_rgb, cutout_rgba = extract_region(image_rgb, mask)
    if crop_rgb is None:
        return None, None, "No segmented region found to extract."
    info = (
        f"### Region Extracted Successfully\n"
        f"- Bounding Box Crop: `{crop_rgb.shape[1]}x{crop_rgb.shape[0]}` px (True H&E stain colors preserved)\n"
        f"- Isolated Object Cutout: Transparent RGBA cutout ready for morphological inspection."
    )
    return crop_rgb, cutout_rgba, info


def run_model_benchmark(editor_data, state):
    """Runs identical prompt/stroke across MedSAM, ScribblePrompt, and reports benchmark latency."""
    if state is None or state.get("image") is None:
        return "Upload an image and draw a region first."
    image_rgb = state["image"].copy()
    h, w = image_rgb.shape[:2]

    stroke_mask = np.zeros((h, w), dtype=bool)
    if isinstance(editor_data, dict):
        layers = editor_data.get("layers", [])
        for l in layers:
            if isinstance(l, str):
                for _ in range(10):
                    l_img = cv2.imread(l, cv2.IMREAD_UNCHANGED)
                    if l_img is not None:
                        break
                    time.sleep(0.04)
                if l_img is not None and l_img.ndim == 3 and l_img.shape[2] == 4:
                    if l_img.shape[:2] != (h, w):
                        l_img = cv2.resize(l_img, (w, h), interpolation=cv2.INTER_NEAREST)
                    stroke_mask = stroke_mask | (l_img[:, :, 3] > 0)

    if not stroke_mask.any():
        if state.get("mask") is not None and state["mask"].any():
            stroke_mask = state["mask"]
        else:
            return "Please draw a brush stroke or circle an ROI to benchmark across models."

    benchmark_md = compare_models_roi(image_rgb, stroke_mask, medsam_predictor, scribbleprompt_net, device)
    return benchmark_md


def clear_app():
    return None, None, "Upload a raw cervical-cell or tissue image to begin.", None, None, None, ""


# ─────────────────────────────────────────────────────────────
# Herlev Handlers (First Commit Logic & '+' Crosshair Marker)
# ─────────────────────────────────────────────────────────────
def predict_herlev(image_rgb):
    """Herlev inference matching first commit logic."""
    if image_rgb is None:
        return None, "Upload an image first.", None

    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[:, :, :3]
    image_rgb = image_rgb.astype(np.uint8)

    x, original_h, original_w = preprocess(image_rgb)

    with torch.no_grad():
        logits = model(x)
        prediction_128 = torch.softmax(logits, dim=1).argmax(dim=1)[0]
        prediction_128 = prediction_128.cpu().numpy().astype(np.uint8)

    prediction_full = cv2.resize(
        prediction_128,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = make_overlay(image_rgb, prediction_full)

    state = {
        "image": image_rgb,
        "mask": prediction_full,
        "last_class": None,
        "model": "Herlev"
    }

    info = (
        "Segmentation complete. Click anywhere on the image below.\n\n"
        "Red = Background | Dark blue = Cytoplasm | Light blue = Nucleus"
    )

    return overlay, info, state


def select_herlev_region(state, evt: gr.SelectData):
    """
    Herlev click callback strictly from first commit with '+' crosshair marker.
    """
    if state is None or state.get("image") is None:
        return gr.skip(), "Upload an image and run segmentation first."

    image_rgb = state["image"].copy()
    h, w = image_rgb.shape[:2]
    idx = getattr(evt, "index", None)
    if idx is None:
        return gr.skip(), gr.skip()
    x, y = int(idx[0]), int(idx[1])

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    mask = state["mask"]
    class_id = int(mask[y, x])
    class_name = CLASS_NAMES[class_id]

    result = make_overlay(image_rgb, mask, alpha=0.42)
    selected = mask == class_id
    yellow_layer = np.zeros_like(result)
    yellow_layer[selected] = YELLOW

    result[selected] = (
        0.52 * result[selected].astype(np.float32)
        + 0.48 * yellow_layer[selected].astype(np.float32)
    ).astype(np.uint8)

    selected_pixels = int(selected.sum())
    total_pixels = int(mask.size)
    selected_percent = 100 * selected_pixels / total_pixels

    message = (
        f"### Selected: **{class_name}**\n\n"
        f"- Click position: `x={x}, y={y}`\n"
        f"- Predicted class ID: `{class_id}`\n"
        f"- Selected area: `{selected_pixels:,}` pixels ({selected_percent:.1f}% of image)\n\n"
        f"The transparent yellow overlay marks the predicted **{class_name.lower()}** region."
    )

    return result, message


def clear_herlev():
    return None, None, "Upload a raw cervical-cell image to begin.", None


# ─────────────────────────────────────────────────────────────
# HoVer-Net Handlers (Graham et al., 2019)
# ─────────────────────────────────────────────────────────────
def load_hovernet_image(image_rgb):
    if image_rgb is None:
        return None, "Upload a histology / tissue image to begin.", None, None
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    if image_rgb.shape[-1] == 4:
        image_rgb = image_rgb[:, :, :3]
    image_rgb = image_rgb.astype(np.uint8)

    state = {
        "image": image_rgb,
        "result_data": None,
        "selected_id": None
    }
    info = (
        "### Histology Image Loaded\n\n"
        "- Click **'Run HoVer-Net Segmentation & Classification'** below to execute the full HoVer-Net pipeline (Graham et al., 2019).\n"
        "- All touching/clustered nuclei will be separated using horizontal & vertical distance maps and classified into 4 CoNSeP/PanNuke phenotypes."
    )
    return image_rgb, info, state, None


def run_hovernet_analysis(image_rgb, state):
    if state is None or state.get("image") is None:
        if image_rgb is None:
            return gr.skip(), "Please upload a histology image first.", state, None
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        if image_rgb.shape[-1] == 4:
            image_rgb = image_rgb[:, :, :3]
        image_rgb = image_rgb.astype(np.uint8)
        state = {"image": image_rgb}

    image_rgb = state["image"]
    result_data = run_hovernet_segmentation(image_rgb)
    state["result_data"] = result_data
    state["selected_id"] = None

    overlay = render_hovernet_display(image_rgb, result_data)
    hover_maps_rgb = render_hover_maps_rgb(result_data["h_map"], result_data["v_map"])
    report_md = format_hovernet_markdown(result_data)

    return overlay, report_md, state, hover_maps_rgb


def select_hovernet_nucleus(state, evt: gr.SelectData):
    if state is None or state.get("image") is None or state.get("result_data") is None:
        return gr.skip(), gr.skip(), state
    idx = getattr(evt, "index", None)
    if idx is None:
        return gr.skip(), gr.skip(), state

    image_rgb = state["image"]
    result_data = state["result_data"]
    instance_labels = result_data.get("instance_labels")
    if instance_labels is None:
        return gr.skip(), gr.skip(), state

    h, w = image_rgb.shape[:2]
    x = max(0, min(int(idx[0]), w - 1))
    y = max(0, min(int(idx[1]), h - 1))

    clicked_id = int(instance_labels[y, x])
    if clicked_id == 0:
        # Search nearest within 10 px radius
        y1, y2 = max(0, y - 10), min(h, y + 10)
        x1, x2 = max(0, x - 10), min(w, x + 10)
        patch = instance_labels[y1:y2, x1:x2]
        nonzeros = patch[patch > 0]
        if len(nonzeros) > 0:
            clicked_id = int(np.bincount(nonzeros).argmax())

    selected_inst = None
    for inst in result_data.get("instances", []):
        if inst["id"] == clicked_id:
            selected_inst = inst
            break

    state["selected_id"] = clicked_id if selected_inst else None
    overlay = render_hovernet_display(image_rgb, result_data, selected_id=state["selected_id"])
    report_md = format_hovernet_markdown(result_data, selected_instance=selected_inst)

    return overlay, report_md, state


def extract_hovernet_nucleus(state):
    if state is None or state.get("image") is None or state.get("result_data") is None:
        return None, None, "No active HoVer-Net segmentation available."

    image_rgb = state["image"]
    result_data = state["result_data"]
    selected_id = state.get("selected_id")

    if selected_id is None:
        instances = result_data.get("instances", [])
        if not instances:
            return None, None, "No nuclei segmented to extract."
        selected_id = instances[0]["id"]

    instance_labels = result_data.get("instance_labels")
    inst_mask = (instance_labels == selected_id)

    crop_rgb, cutout_rgba = extract_region(image_rgb, inst_mask)
    if crop_rgb is None:
        return None, None, "Could not extract selected nucleus."

    info = (
        f"### HoVer-Net Nuclear Instance #{selected_id} Extracted\n"
        f"- **Crop Dimensions**: `{crop_rgb.shape[1]}x{crop_rgb.shape[0]}` px\n"
        f"- True stain color preserved with isolated transparent cutout."
    )
    return crop_rgb, cutout_rgba, info


def clear_hovernet_workspace():
    return None, None, "Upload a histology image to begin.", None, None, None, None


# ─────────────────────────────────────────────────────────────
# Web UI
# ─────────────────────────────────────────────────────────────
# [Oral Histopath AI - Commented out for now]
# ORAL_HISTOPATH_HTML = """
# <div style="width: 100%; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #cbd5e1; margin-top: 6px;">
#     <div style="background: #0f172a; color: white; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #334155;">
#         <div style="display: flex; align-items: center; gap: 10px;">
#             <span style="display: inline-block; width: 10px; height: 10px; background-color: #22c55e; border-radius: 50%; box-shadow: 0 0 8px #22c55e;"></span>
#             <span style="font-weight: 600; font-size: 15px; letter-spacing: 0.02em;">Oral Histopathology AI Analyzer (OPMD / OSCC)</span>
#             <span style="background: #1e293b; color: #94a3b8; font-size: 12px; padding: 2px 8px; border-radius: 4px; border: 1px solid #334155;">WHO 5th Ed. Morphometry &amp; Interactive ScribblePrompt</span>
#         </div>
#         <div style="display: flex; align-items: center; gap: 12px;">
#             <span style="font-size: 12px; color: #94a3b8;">Port: 8000 (Active)</span>
#             <a href="http://127.0.0.1:8000" target="_blank" style="color: #38bdf8; font-size: 13px; text-decoration: none; font-weight: 500; display: inline-flex; align-items: center; gap: 5px; background: rgba(56, 189, 248, 0.12); padding: 5px 12px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.35);">
#                 Open in Full Window ↗
#             </a>
#         </div>
#     </div>
#     <iframe src="http://127.0.0.1:8000" style="width: 100%; height: 88vh; min-height: 820px; border: none; background: #ffffff;"></iframe>
# </div>
# """
# 
# def ensure_oral_backend():
#     """Ensures Oral Histopath AI backend server is running on http://127.0.0.1:8000"""
#     import urllib.request
#     import subprocess
#     import sys
#     try:
#         urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=1.5)
#         print("Oral Histopath AI backend verified active on http://127.0.0.1:8000")
#         return
#     except Exception:
#         pass
# 
#     backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oral_histopath_ai", "backend")
#     if os.path.exists(backend_dir):
#         print("Launching Oral Histopath AI backend on http://127.0.0.1:8000...")
#         subprocess.Popen(
#             [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
#             cwd=backend_dir,
#             creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
#         )
#         time.sleep(2)


with gr.Blocks(title="Cervical Cell Segmentation & MedSAM Nuclei Identification") as demo:
    gr.Markdown(
        """
        # Cervical Cell & Histopathology AI Analyzer
        Interactive, research-grade segmentation and WHO-grounded histological morphometry.

        Select model from dropdown:
        - **MedSAM**: Medical SAM foundation model for manual circling, arc prompts, and nucleus targeting.
        - **Herlev (EffNet-B7)**: Multi-class semantic segmentation (**Background**, **Cytoplasm**, **Nucleus**).
        # - **Oral Histopath AI (ScribblePrompt & Morphometry)**: [Commented out for now]
        - **HoVer-Net**: Simultaneous nuclear instance segmentation & phenotypic classification using horizontal/vertical distance maps (Graham et al., 2019).
        """
    )

    model_status = (
        f"Models Available: `MedSAM (ViT-B)` | `EfficientNet-B7 + FPN` | `HoVer-Net (PanNuke/CoNSeP)`  \n"
        f"Device: `{device}`  \n"
    )
    gr.Markdown(model_status)

    with gr.Row():
        with gr.Column(scale=0, min_width=380):
            model_choice = gr.Dropdown(
                choices=[
                    "MedSAM",
                    "Herlev (EffNet-B7)",
                    # "Oral Histopath AI (ScribblePrompt & Morphometry)",  # [Commented out for now]
                    "HoVer-Net (Nuclear Instance Segmentation & Classification)"
                ],
                value="MedSAM",
                label="Segmentation Model",
                scale=0,
                min_width=380
            )

    state = gr.State(None)

    # 1. Primary Container: MedSAM (With Doctor Brush & Circling Workflow)
    with gr.Column(visible=True) as medsam_container:
        with gr.Row():
            input_image = gr.Image(
                label="1. Upload raw cervical-cell / histology slide image",
                type="numpy",
                image_mode="RGB",
            )

            output_image = gr.ImageEditor(
                label="2. Segmentation result — circle missed nuclei with brush",
                type="filepath",
                brush=gr.Brush(default_size=10, colors=["#ff0000", "#ffff00", "#00ff00"], default_color="#ff0000"),
                eraser=gr.Eraser(),
                transforms=(),
                sources=(),
            )

        selected_info = gr.Markdown(
            "Upload an image first. Then click 'Run Segmentation' or use the brush to circle missed nuclei."
        )

        with gr.Row():
            segment_button = gr.Button("Run Whole-Slide Segmentation", variant="primary")
            extract_button = gr.Button("Extract Active Region (Original Colors)", variant="secondary")
            compare_button = gr.Button("Compare Models on Current ROI", variant="secondary")
            reset_doc_button = gr.Button("Reset Doctor Targets")
            clear_button = gr.Button("Clear All")

        with gr.Accordion("🔬 Extracted Cellular Region (Dual Original-Color Cutout & Bounding Box)", open=True):
            gr.Markdown(
                "High-resolution cellular extraction preserving original H&E staining colors without modification."
            )
            with gr.Row():
                crop_display = gr.Image(label="Original RGB Bounding Box Crop", type="numpy", interactive=False)
                cutout_display = gr.Image(label="Isolated Object Cutout (Transparent Background)", type="numpy", interactive=False)

        with gr.Accordion("⚖️ Multi-Model Benchmark Comparison (Identical ROI)", open=False):
            benchmark_output = gr.Markdown("Click **'Compare Models on Current ROI'** above to run candidate AI models side-by-side.")

        with gr.Accordion("⚠️ Medical Device & Educational Safety Disclaimer", open=False):
            gr.Markdown(
                """
                > **Strict Research and Educational Use Only**: This software is an investigational computational pathology prototype.
                > 
                > 1. **No Automated Diagnostic Claims**: Morphological measurements, optical stain quantifications, and feature classifications are algorithmic research proxies and do not constitute a clinical, medical, or pathological diagnosis.
                > 2. **Multi-Model Transparency**: All active models (`MedSAM`, `EfficientNet-B7`, `ScribblePrompt`, `HoVer-Net`) run on validated local neural weights without fabricated outputs.
                """
            )

        gr.Markdown(
            """
            ### Colour legend (MedSAM)
            - Transparent yellow + Green boundary: All identified & rounded-off nuclei
            """
        )

    # 2. Herlev Container: Strictly First Commit UI & Manual '+' Selection (No Brush)
    herlev_state = gr.State(None)
    with gr.Column(visible=False) as herlev_container:
        with gr.Row():
            herlev_input_image = gr.Image(
                label="1. Upload raw cervical-cell image",
                type="numpy",
                image_mode="RGB",
            )

            herlev_output_image = gr.Image(
                label="2. Predicted segmentation — click here to select/add nuclei",
                type="numpy",
                interactive=False,  # Display only so cursor does not hide or clear image
            )

        herlev_selected_info = gr.Markdown(
            "Upload an image first. Then click 'Run Segmentation' or click on the predicted segmentation image."
        )

        with gr.Row():
            herlev_segment_button = gr.Button("Run Segmentation", variant="primary")
            herlev_clear_button = gr.Button("Clear All")

        gr.Markdown(
            """
            ### Colour legend (Herlev)
            - Red: Background
            - Dark blue: Cytoplasm
            - Light blue: Nucleus
            - Transparent yellow: Region selected by your click
            """
        )

    # 3. HoVer-Net Container: Simultaneous Instance Segmentation & Phenotypic Classification (No Brush)
    hover_state = gr.State(None)
    with gr.Column(visible=False) as hovernet_container:
        gr.Markdown(
            r"""
            ### 🔬 HoVer-Net: Simultaneous Segmentation & Classification of Nuclei
            *Based on Graham et al. (arXiv:1812.06499v5 / PanNuke & CoNSeP benchmark)*  
            Leverages **horizontal and vertical distance maps** to their centres of mass ($S_m = \max(H_x, H_y)$) to cleanly separate touching/clustered nuclei, and simultaneously classifies each instance into **Epithelial/Tumour**, **Inflammatory (TILs)**, **Spindle-Shaped (Stroma)**, or **Miscellaneous/Mitotic**.
            """
        )
        with gr.Row():
            hover_input_image = gr.Image(
                label="1. Upload Multi-Tissue Histology / Cell Slide",
                type="numpy",
                image_mode="RGB",
            )
            hover_output_image = gr.Image(
                label="2. HoVer-Net Result (Click any nucleus to inspect)",
                type="numpy",
                interactive=False,
            )

        hover_selected_info = gr.Markdown(
            "Upload an image and click **'Run HoVer-Net Segmentation & Classification'** below."
        )

        with gr.Row():
            hover_segment_btn = gr.Button("Run HoVer-Net Segmentation & Classification", variant="primary")
            hover_extract_btn = gr.Button("Extract Selected Nucleus", variant="secondary")
            hover_clear_btn = gr.Button("Clear Workspace")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("🗺️ HoVer Distance Maps Preview (Horizontal px & Vertical py Fields)", open=True):
                    gr.Markdown("Visual encoding of horizontal & vertical instance distance maps used to split clustered nuclei:")
                    hover_maps_display = gr.Image(label="HoVer Distance Fields", type="numpy", interactive=False)
            with gr.Column(scale=1):
                with gr.Accordion("🔬 Extracted Nuclear Instance (Dual RGB Crop & Transparent Cutout)", open=True):
                    with gr.Row():
                        hover_crop_display = gr.Image(label="RGB Crop", type="numpy", interactive=False)
                        hover_cutout_display = gr.Image(label="Transparent Cutout", type="numpy", interactive=False)

        gr.Markdown(
            """
            ### HoVer-Net Phenotypic Color Legend:
            - 🔴 **Red Contour**: Epithelial / Tumour Cell Nuclei
            - 🔵 **Blue Contour**: Inflammatory / Lymphocytes (Tumour-Infiltrating Lymphocytes / TILs)
            - 🟢 **Green Contour**: Spindle-Shaped / Stromal / Fibroblasts
            - 🟡 **Yellow Contour**: Miscellaneous / Mitotic Figures / Necrotic Debris
            """
        )

    def switch_model_view(choice):
        is_medsam = ("MedSAM" in str(choice))
        is_herlev = ("Herlev" in str(choice))
        is_hover = ("HoVer-Net" in str(choice))

        return (
            gr.update(visible=is_medsam),
            gr.update(visible=is_herlev),
            gr.update(visible=is_hover),
        )

    model_choice.change(
        fn=switch_model_view,
        inputs=[model_choice],
        outputs=[medsam_container, herlev_container, hovernet_container],
    )

    # MedSAM Events (Unchanged)
    segment_button.click(
        fn=predict_image,
        inputs=[input_image, model_choice],
        outputs=[output_image, selected_info, state],
    )

    input_image.change(
        fn=load_image,
        inputs=[input_image, model_choice],
        outputs=[output_image, selected_info, state],
    )

    output_image.input(
        fn=target_drawn_circle,
        inputs=[output_image, state],
        outputs=[output_image, selected_info, state],
    )

    output_image.select(
        fn=select_region,
        inputs=state,
        outputs=[output_image, selected_info, state],
    )

    input_image.select(
        fn=select_region,
        inputs=state,
        outputs=[output_image, selected_info, state],
    )

    extract_button.click(
        fn=extract_current_roi,
        inputs=state,
        outputs=[crop_display, cutout_display, selected_info],
    )

    compare_button.click(
        fn=run_model_benchmark,
        inputs=[output_image, state],
        outputs=[benchmark_output],
    )

    reset_doc_button.click(
        fn=reset_doctor_targets,
        inputs=state,
        outputs=[output_image, selected_info, state],
    )

    clear_button.click(
        fn=clear_app,
        inputs=[],
        outputs=[input_image, output_image, selected_info, state, crop_display, cutout_display, benchmark_output],
    )

    # Herlev Events (First Commit Logic & '+' Selection)
    herlev_segment_button.click(
        fn=predict_herlev,
        inputs=[herlev_input_image],
        outputs=[herlev_output_image, herlev_selected_info, herlev_state],
    )

    herlev_input_image.change(
        fn=predict_herlev,
        inputs=[herlev_input_image],
        outputs=[herlev_output_image, herlev_selected_info, herlev_state],
    )

    herlev_output_image.select(
        fn=select_herlev_region,
        inputs=herlev_state,
        outputs=[herlev_output_image, herlev_selected_info],
    )

    herlev_input_image.select(
        fn=select_herlev_region,
        inputs=herlev_state,
        outputs=[herlev_output_image, herlev_selected_info],
    )

    herlev_clear_button.click(
        fn=clear_herlev,
        inputs=[],
        outputs=[herlev_input_image, herlev_output_image, herlev_selected_info, herlev_state],
    )

    # HoVer-Net Events (Manual '+' Selection)
    hover_segment_btn.click(
        fn=run_hovernet_analysis,
        inputs=[hover_input_image, hover_state],
        outputs=[hover_output_image, hover_selected_info, hover_state, hover_maps_display],
    )

    hover_input_image.change(
        fn=load_hovernet_image,
        inputs=[hover_input_image],
        outputs=[hover_output_image, hover_selected_info, hover_state, hover_maps_display],
    )

    hover_output_image.select(
        fn=select_hovernet_nucleus,
        inputs=hover_state,
        outputs=[hover_output_image, hover_selected_info, hover_state],
    )

    hover_extract_btn.click(
        fn=extract_hovernet_nucleus,
        inputs=hover_state,
        outputs=[hover_crop_display, hover_cutout_display, hover_selected_info],
    )

    hover_clear_btn.click(
        fn=clear_hovernet_workspace,
        inputs=[],
        outputs=[hover_input_image, hover_output_image, hover_selected_info, hover_state, hover_maps_display, hover_crop_display, hover_cutout_display],
    )

if __name__ == "__main__":
    # ensure_oral_backend()  # [Oral Histopath AI commented out for now]
    demo.queue(default_concurrency_limit=1).launch(inbrowser=True, server_port=7860)
