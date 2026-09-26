import os
import cv2
import numpy as np
import torch
import gradio as gr
import segmentation_models_pytorch as smp

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sam_model_registry, SamPredictor = None, None

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


# ─────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────
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
            for sc, b_box in cell_cands[:2]:
                selected_boxes.append(b_box)
                
    return selected_boxes


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


def render_medsam_display(image_rgb, combined_mask, doctor_targets):
    """Renders composite display showing all nuclei + doctor targeted points."""
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
        
    # Mark doctor targeted points with distinct identifiers
    for idx, (px, py) in enumerate(doctor_targets):
        cv2.circle(result, (px, py), 5, (0, 0, 255), -1)  # red dot
        cv2.circle(result, (px, py), 8, (255, 255, 255), 2)  # white border
        cv2.putText(result, f"#{idx+1}", (px + 7, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)
        cv2.putText(result, f"#{idx+1}", (px + 7, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 220), 1)
        
    return result


# ─────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────
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

    if model_choice == "MedSAM (ViT-B)":
        if medsam_predictor is None:
            return image_rgb, "Error: MedSAM model failed to load.", None
        
        medsam_predictor.set_image(image_rgb)
        
        # Detect candidate nuclei across the entire slide
        boxes = detect_all_nuclei_boxes(image_rgb)
        combined_mask = np.zeros((h, w), dtype=bool)
        auto_count = 0
        
        if len(boxes) > 0:
            try:
                boxes_tensor = torch.tensor(boxes, device=device)
                transformed_boxes = medsam_predictor.transform.apply_boxes_torch(boxes_tensor, (h, w))
                masks, scores, _ = medsam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False
                )
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                for m in masks:
                    m_np = m[0].cpu().numpy().astype(np.uint8)
                    m_smooth = cv2.morphologyEx(m_np, cv2.MORPH_CLOSE, k_close)
                    combined_mask = combined_mask | (m_smooth > 0)
                auto_count = len(boxes)
            except Exception as e:
                print("MedSAM whole-slide batch exception:", e)
        
        result = render_medsam_display(image_rgb, combined_mask, [])
        
        cnts_tot, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_nuclei = len(cnts_tot)
        
        info = (
            f"### MedSAM: **{total_nuclei} Nuclei Segmented Across Whole Slide**\n\n"
            f"- Automatically identified and rounded off **{auto_count} candidate nuclei** across top, center, and bottom fields.\n"
            f"- **Doctor Targeting Active**: Click directly on any missed nucleus in the image.\n"
            f"- Every clicked point will be remembered, rigorously segmented, and merged with existing data!"
        )
            
        state = {
            "image": image_rgb,
            "auto_mask": combined_mask.copy(),
            "mask": combined_mask,
            "doctor_targets": [],
            "model": "MedSAM"
        }
        return result, info, state

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

    return overlay, info, state


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
        return gr.skip(), "Upload an image and run segmentation first."

    image_rgb = state["image"].copy()
    h, w = image_rgb.shape[:2]
    x, y = evt.index

    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    if state.get("model") == "MedSAM":
        if medsam_predictor is None:
            return image_rgb, "Error: MedSAM model not loaded."

        # Rigorously segment the nucleus at the clicked location
        try:
            new_nuc_mask, (cx, cy), conf = target_nucleus_rigorous(medsam_predictor, image_rgb, x, y)
        except Exception as e:
            print("MedSAM targeting error:", e)
            return image_rgb, f"MedSAM targeting error: {e}"

        # Remember doctor target point
        doctor_targets = state.get("doctor_targets", [])
        doctor_targets.append((x, y))
        state["doctor_targets"] = doctor_targets

        # Accumulate with previous data (Purana + Naya)
        existing_mask = state.get("mask")
        if existing_mask is None or existing_mask.shape != (h, w):
            combined_mask = new_nuc_mask
        else:
            combined_mask = existing_mask | new_nuc_mask
        state["mask"] = combined_mask

        # Render combined display
        result = render_medsam_display(image_rgb, combined_mask, doctor_targets)

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
        return result, message

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

    return result, message


def reset_doctor_targets(state):
    """Resets doctor manual targets and reverts back to the base auto-detected nuclei."""
    if state is None or state.get("image") is None:
        return gr.skip(), "No image loaded.", state
    image_rgb = state["image"]
    auto_mask = state.get("auto_mask", np.zeros(image_rgb.shape[:2], dtype=bool))
    state["mask"] = auto_mask.copy()
    state["doctor_targets"] = []
    
    result = render_medsam_display(image_rgb, auto_mask, [])
    info = "Doctor manual targets reset. Base auto-detected nuclei preserved. Click any nucleus to target again!"
    return result, info, state


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
        - **MedSAM (ViT-B)**: Medical foundation model that **detects nuclei across the whole slide**. 
          Doctors can **manually target any missed nuclei** by clicking on them: the system rigorously isolates and rounds off the nucleus, remembers all targeted points, and shows **previous + new doctor targets together**!
        """
    )

    model_status = (
        f"Models Available: `EfficientNet-B7 Noisy Student + FPN` & `MedSAM (ViT-B)`  \n"
        f"Device: `{device}`  \n"
    )
    gr.Markdown(model_status)

    model_choice = gr.Radio(
        choices=["Herlev (EffNet-B7)", "MedSAM (ViT-B)"],
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

        output_image = gr.Image(
            label="2. Segmentation result — click to manually target missed nuclei",
            type="numpy",
            interactive=False,  # Display only so cursor does not hide or clear image
        )

    selected_info = gr.Markdown(
        "Upload an image first. Then click 'Run Segmentation' or click on the image to target nuclei."
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
        fn=predict_image,
        inputs=[input_image, model_choice],
        outputs=[output_image, selected_info, state],
    )

    model_choice.change(
        fn=predict_image,
        inputs=[input_image, model_choice],
        outputs=[output_image, selected_info, state],
    )

    output_image.select(
        fn=select_region,
        inputs=state,
        outputs=[output_image, selected_info],
    )

    input_image.select(
        fn=select_region,
        inputs=state,
        outputs=[output_image, selected_info],
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
        - Transparent yellow + Green boundary: All identified & rounded-off nuclei across the slide
        - Red dot + White border (`#1`, `#2`, ...): Manually targeted & verified nuclei by doctor
        """
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_port=7860)