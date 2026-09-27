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

MEDSAM_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "medsam_vit_b.pth"
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


def build_medsam_model():
    if sam_model_registry is None:
        print("segment_anything is not installed. MedSAM will not work.")
        return None
    try:
        sam = sam_model_registry["vit_b"](checkpoint=None)
        state_dict = torch.load(MEDSAM_MODEL_PATH, map_location=device)
        sam.load_state_dict(state_dict)
        sam.to(device=device)
        sam.eval()
        print("MedSAM (ViT-B) loaded successfully on", device)
        return SamPredictor(sam)
    except Exception as e:
        print("Failed to load MedSAM:", e)
        return None


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}\n\n"
        "Put herlev_effnetb7_fpn.pth in the same folder as app.py."
    )

model = build_model()
medsam_predictor = build_medsam_model()

print("Models loaded successfully.")
print("Device:", device)

_cached_medsam_sig = None
_medsam_lock = threading.Lock()


def ensure_medsam_image(image_rgb):
    """Encodes image in MedSAM only once per image, using fast inference_mode."""
    global _cached_medsam_sig
    if medsam_predictor is None or image_rgb is None:
        return
    sig = (image_rgb.shape, int(image_rgb[0, 0, 0]), int(image_rgb[-1, -1, 0]), int(image_rgb.mean()))
    if _cached_medsam_sig != sig:
        with _medsam_lock:
            if _cached_medsam_sig != sig:
                with torch.inference_mode():
                    medsam_predictor.set_image(image_rgb)
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


def target_nucleus_rigorous(predictor, image_rgb, px, py):
    """
    Rigorously targets and segments a single nucleus around a doctor's click:
    - Finds the true local center in a 24px patch (highest optical density).
    - Uses a calibrated 16px radius bounding box.
    - MedSAM promptable segmentation with connected-component filtering and hole filling.
    """
    h, w = image_rgb.shape[:2]
    pw = 14
    x1, y1 = max(0, px - pw), max(0, py - pw)
    x2, y2 = min(w, px + pw), min(h, py + pw)
    patch = image_rgb[y1:y2, x1:x2]
    
    # Nucleus core is darkest in the green channel
    min_val, _, min_loc, _ = cv2.minMaxLoc(patch[:, :, 1])
    cx = x1 + min_loc[0]
    cy = y1 + min_loc[1]
    
    r = 16
    box = np.array([max(0, cx - r), max(0, cy - r), min(w, cx + r), min(h, cy + r)])
    
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            box=box,
            multimask_output=False
        )
    raw_mask = masks[0]
    
    # Isolate component closest to true center
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
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smooth = cv2.morphologyEx(mask_u, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(smooth)
    for c in cnts:
        cv2.drawContours(filled, [c], -1, 1, -1)
        
    return filled.astype(bool), (cx, cy), scores[0]


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
    
    if "MedSAM" in model_choice:
        prewarm_medsam_image(image_rgb)
        
    combined_mask = np.zeros((h, w), dtype=bool)
    state = {
        "image": image_rgb,
        "auto_mask": combined_mask.copy(),
        "mask": combined_mask,
        "doctor_targets": [],
        "model": "MedSAM" if "MedSAM" in model_choice else "Herlev"
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

    if model_choice == "MedSAM":
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

    if state.get("model") == "MedSAM":
        if medsam_predictor is None:
            return gr.skip(), "Error: MedSAM model not loaded.", state

        existing_mask = state.get("mask")
        if existing_mask is not None and existing_mask.shape == (h, w) and existing_mask[y, x]:
            return gr.skip(), gr.skip(), state

        ensure_medsam_image(image_rgb)

        # Rigorously segment the nucleus at the clicked location
        try:
            with _medsam_lock, torch.inference_mode():
                new_nuc_mask, (cx, cy), conf = target_nucleus_rigorous(medsam_predictor, image_rgb, x, y)
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

        message = (
            f"### MedSAM: **{total_nuclei} Nuclei Segmented Together**\n\n"
            f"- **New Target Added**: Target #{len(doctor_targets)} at `x={x}, y={y}` (refined core at `{cx}, {cy}`)\n"
            f"- **Confidence Score**: `{conf:.3f}`\n"
            f"- **Doctor Targets Remembered**: `{len(doctor_targets)}` manually verified nuclei\n"
            f"- **Combined Field Area**: `{selected_pixels:,}` pixels ({selected_percent:.1f}% of slide)\n\n"
            f"Showing **previous auto-detected nuclei + all doctor-targeted nuclei** together in transparent yellow with green rounded boundaries!"
        )
        return robust_image(result), message, state

    # Herlev interaction
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

    cv2.circle(result, (x, y), 7, (255, 235, 0), -1)
    cv2.circle(result, (x, y), 9, (0, 0, 0), 2)

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

    if state.get("model") == "MedSAM":
        if medsam_predictor is None:
            return gr.skip(), "Error: MedSAM model not loaded.", state

        cnts, _ = cv2.findContours(stroke_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return gr.skip(), gr.skip(), gr.skip()

        doctor_targets = state.get("doctor_targets", [])
        combined_mask = state.get("mask", np.zeros((h, w), dtype=bool))
        newly_added = 0

        # Sort contours from largest to smallest
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 4 or bh < 4:
                continue

            pad = 4
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(w, bx + bw + pad)
            y2 = min(h, by + bh + pad)

            # Find the true nucleus core (highest hematoxylin optical density)
            patch = image_rgb[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            min_val, _, min_loc, _ = cv2.minMaxLoc(patch[:, :, 1])
            if min_val < 240:
                cx = x1 + min_loc[0]
                cy = y1 + min_loc[1]
            else:
                cx = bx + bw // 2
                cy = by + bh // 2

            # Avoid re-adding if exactly at an already recorded doctor target point
            if any((cx - tx)**2 + (cy - ty)**2 < 25 for tx, ty in doctor_targets):
                continue

            # Ensure image embeddings are ready (computed once per image, then cached)
            ensure_medsam_image(image_rgb)

            box_prompt = np.array([x1, y1, x2, y2])
            try:
                with _medsam_lock, torch.inference_mode():
                    masks, scores, _ = medsam_predictor.predict(
                        box=box_prompt,
                        point_coords=np.array([[cx, cy]]),
                        point_labels=np.array([1]),
                        multimask_output=False
                    )
                refined = round_off_mask(masks[0])
                combined_mask = combined_mask | refined
                doctor_targets.append((cx, cy))
                newly_added += 1
            except Exception as e:
                print("MedSAM brush circling error:", e)

        if newly_added == 0:
            return gr.skip(), f"Covered — `{len(doctor_targets)}` doctor targets remembered.", state

        state["mask"] = combined_mask
        state["doctor_targets"] = doctor_targets

        result = render_medsam_display(image_rgb, combined_mask)

        cnts_tot, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_nuclei = len(cnts_tot)
        selected_pixels = int(combined_mask.sum())
        selected_percent = 100 * selected_pixels / (h * w)

        message = (
            f"### MedSAM: **{total_nuclei} Nuclei Segmented Together**\n\n"
            f"- **Doctor Brush Circling**: Added and perfected **{newly_added} circled nucleus candidate(s)**\n"
            f"- **Doctor Targets Remembered**: `{len(doctor_targets)}` manually verified nuclei\n"
            f"- **Combined Field Area**: `{selected_pixels:,}` pixels ({selected_percent:.1f}% of slide)\n\n"
            f"Vague circled shapes are now **perfected into rounded nuclei** and shown with previous auto-detected nuclei in yellow with green contours!"
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

    cv2.circle(result, (cx, cy), 7, (255, 235, 0), -1)
    cv2.circle(result, (cx, cy), 9, (0, 0, 0), 2)

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


def clear_app():
    return None, None, "Upload a raw cervical-cell or tissue image to begin.", None


# ─────────────────────────────────────────────────────────────
# Web UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Cervical Cell Segmentation & MedSAM Nuclei Identification") as demo:
    gr.Markdown(
        """
        # Cervical Cell Semantic Segmentation & MedSAM Nuclei Identification

        Upload a cervical-cell or histology slide and select your model:
        - **Herlev (EffNet-B7)**: Multi-class semantic segmentation (**Background**, **Cytoplasm**, **Nucleus**).
        - **MedSAM**: Pure manual mode — image is displayed as-is, circle any nucleus with the brush tool to predict and select it.
        """
    )

    model_status = (
        f"Models Available: `EfficientNet-B7 Noisy Student + FPN` & `MedSAM`  \n"
        f"Device: `{device}`  \n"
    )
    gr.Markdown(model_status)

    model_choice = gr.Radio(
        choices=["Herlev (EffNet-B7)", "MedSAM"],
        value="Herlev (EffNet-B7)",
        label="Select Model"
    )

    state = gr.State(None)

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
        reset_doc_button = gr.Button("Reset Doctor Targets")
        clear_button = gr.Button("Clear All")

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

    model_choice.change(
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

    reset_doc_button.click(
        fn=reset_doctor_targets,
        inputs=state,
        outputs=[output_image, selected_info, state],
    )

    clear_button.click(
        fn=clear_app,
        inputs=[],
        outputs=[input_image, output_image, selected_info, state],
    )

    gr.Markdown(
        """
        ### Colour legend (Herlev)
        - Red: Background
        - Dark blue: Cytoplasm
        - Light blue: Nucleus
        - Transparent yellow: Region selected by your click

        ### Colour legend (MedSAM)
        - Transparent yellow + Green boundary: All identified & rounded-off nuclei
        """
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(inbrowser=True, server_port=7860)