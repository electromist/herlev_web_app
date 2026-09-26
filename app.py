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
CONTOUR_COLOR = (34, 197, 94)  # vibrant emerald green for crisp rounded borders


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
        # Load weights with map_location to ensure CPU compatibility
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
        if cv2.contourArea(c) > 20:
            cv2.drawContours(filled, [c], -1, 1, -1)
    return filled.astype(bool)


def detect_candidate_nuclei_boxes(image_rgb):
    """
    Detects nuclei bounding boxes for MedSAM segmentation:
    1. First searches for hand-marked annotations (yellow, orange, green, cyan)
       drawn on cytology or tissue slides.
    2. If no annotations are present, falls back to automatic dark nucleus blob detection.
    """
    h, w = image_rgb.shape[:2]
    r, g, b = cv2.split(image_rgb)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    
    # 1. Detect yellow/orange pen or pencil markings
    yellow_mark = (h_ch >= 4) & (h_ch <= 36) & (s_ch >= 22) & (v_ch >= 130)
    yellow_mark = yellow_mark & (((r.astype(int) + g.astype(int)) // 2 - b.astype(int)) > 14)
    
    # Green / Cyan markings
    green_mark = (h_ch >= 37) & (h_ch <= 95) & (s_ch >= 50)
    
    marks = (yellow_mark | green_mark).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(marks, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marked_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if area > 25 and 10 < bw < 130 and 10 < bh < 130 and by > 35:
            pad = 2
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(w, bx + bw + pad)
            y2 = min(h, by + bh + pad)
            marked_boxes.append([x1, y1, x2, y2])
            
    if len(marked_boxes) > 0:
        return marked_boxes, True

    # 2. Fallback: Automatic dark nucleus detector
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    thresh = cv2.adaptiveThreshold(
        inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = max(35, int(h * w * 0.001))
    max_area = int(h * w * 0.15)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            pad = 4
            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(w, bx + bw + pad)
            y2 = min(h, by + bh + pad)
            boxes.append([x1, y1, x2, y2])
            
    if len(boxes) > 20:
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:20]
    return boxes, False


# ─────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────
def predict_image(image_rgb, model_choice):
    """
    Runs segmentation based on model selection:
    - Herlev: full semantic segmentation (Background, Cytoplasm, Nucleus)
    - MedSAM: identifies all marked/candidate nuclei together and rounds them off
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
        
        boxes, is_marked = detect_candidate_nuclei_boxes(image_rgb)
        combined_mask = np.zeros((h, w), dtype=bool)
        nuclei_count = 0
        
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
                for m in masks:
                    m_np = m[0].cpu().numpy()
                    m_rounded = round_off_mask(m_np)
                    combined_mask = combined_mask | m_rounded
                nuclei_count = len(boxes)
            except Exception as e:
                print("MedSAM batch detection exception:", e)
        
        result = image_rgb.copy()
        if combined_mask.any():
            yellow_layer = np.zeros_like(result)
            yellow_layer[combined_mask] = YELLOW
            result[combined_mask] = (
                0.52 * result[combined_mask].astype(np.float32)
                + 0.48 * yellow_layer[combined_mask].astype(np.float32)
            ).astype(np.uint8)
            
            # Draw smooth green/yellow contours rounding off the nuclei
            contours, _ = cv2.findContours(combined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, CONTOUR_COLOR, 2)
            
            if is_marked:
                info = (
                    f"### MedSAM: **{nuclei_count} Hand-Marked Nuclei Identified & Rounded Off Together**\n\n"
                    f"- Detected **{nuclei_count} hand-marked regions** from your annotations.\n"
                    f"- MedSAM successfully isolated and rounded off each marked nucleus.\n"
                    f"- **Interactive Mode**: Click on any additional nucleus or cell to add it to the group!"
                )
            else:
                info = (
                    f"### MedSAM: **{nuclei_count} Nuclei Identified & Rounded Off Together**\n\n"
                    f"- Automatically identified and rounded off **{nuclei_count} candidate nuclei**.\n"
                    f"- **Interactive Mode**: Click on any nucleus or cell to add it or refine the mask.\n"
                    f"- All nuclei are highlighted together in yellow."
                )
        else:
            info = (
                "### MedSAM Ready\n\n"
                "Image loaded. Click anywhere on a nucleus to segment it. Each click will identify and keep multiple nuclei together!"
            )
            
        state = {
            "image": image_rgb,
            "mask": combined_mask,
            "points": [],
            "boxes": boxes,
            "nuclei_count": nuclei_count,
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
# Click interaction
# ─────────────────────────────────────────────────────────────
def select_region(state, evt: gr.SelectData):
    """
    Click callback:
    - Never returns None (so the image never disappears!)
    - For Herlev: highlights all regions belonging to the clicked class
    - For MedSAM: segments the clicked nucleus, rounds it off, and accumulates it with all other nuclei
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
        
        # Check if click is near any existing detected/marked box
        chosen_box = None
        for b in state.get("boxes", []):
            if (b[0] - 15 <= x <= b[2] + 15) and (b[1] - 15 <= y <= b[3] + 15):
                chosen_box = np.array(b)
                break
                
        if chosen_box is None:
            # Optimal compact radius (16px) for sharp, rounded nucleus extraction
            r = max(14, min(22, int(min(h, w) * 0.05)))
            chosen_box = np.array([max(0, x - r), max(0, y - r), min(w, x + r), min(h, y + r)])

        input_point = np.array([[x, y]])
        input_label = np.array([1])
        
        try:
            masks, scores, _ = medsam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=chosen_box,
                multimask_output=False
            )
            raw_mask = masks[0]
            new_mask = round_off_mask(raw_mask)
        except Exception as e:
            print("MedSAM prediction error:", e)
            return image_rgb, f"MedSAM prediction error: {e}"
        
        # Accumulate with previously identified nuclei
        existing_mask = state.get("mask")
        if existing_mask is None or existing_mask.shape != (h, w):
            combined_mask = new_mask
        else:
            combined_mask = existing_mask | new_mask
            
        points = state.get("points", [])
        points.append((x, y))
        
        state["mask"] = combined_mask
        state["points"] = points
        
        mask_uint8 = combined_mask.astype(np.uint8)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_uint8)
        nuclei_count = max(len(points), num_labels - 1)
        state["nuclei_count"] = nuclei_count
        
        # Render accumulated nuclei overlay
        result = image_rgb.copy()
        yellow_layer = np.zeros_like(result)
        yellow_layer[combined_mask] = YELLOW
        
        result[combined_mask] = (
            0.52 * result[combined_mask].astype(np.float32)
            + 0.48 * yellow_layer[combined_mask].astype(np.float32)
        ).astype(np.uint8)
        
        # Draw smooth rounded contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, CONTOUR_COLOR, 2)
        
        # Numbered markers for user clicks
        for idx, (px, py) in enumerate(points):
            cv2.circle(result, (px, py), 5, (255, 235, 0), -1)
            cv2.circle(result, (px, py), 7, (0, 0, 0), 2)
            cv2.putText(result, str(idx + 1), (px + 6, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
            cv2.putText(result, str(idx + 1), (px + 6, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            
        selected_pixels = int(combined_mask.sum())
        total_pixels = int(combined_mask.size)
        selected_percent = 100 * selected_pixels / total_pixels
        
        message = (
            f"### MedSAM: **{nuclei_count} Nuclei Identified & Rounded Off Together**\n\n"
            f"- Last clicked position: `x={x}, y={y}` (Nucleus #{len(points)})\n"
            f"- Nucleus confidence: `{scores[0]:.3f}`\n"
            f"- Total identified nuclei: `{nuclei_count}`\n"
            f"- Combined area: `{selected_pixels:,}` pixels ({selected_percent:.1f}% of image)\n\n"
            f"All nuclei are highlighted together with smooth, rounded contours. Click any additional nucleus to add it!"
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


def reset_medsam_nuclei(state):
    """Resets accumulated MedSAM points and masks while keeping the image loaded."""
    if state is None or state.get("image") is None:
        return gr.skip(), "No image loaded.", state
    image_rgb = state["image"]
    state["mask"] = np.zeros(image_rgb.shape[:2], dtype=bool)
    state["points"] = []
    state["nuclei_count"] = 0
    return image_rgb, "MedSAM nuclei cleared. Click anywhere on the image to identify nuclei.", state


def clear_app():
    return None, None, "Upload a raw cervical-cell image to begin.", None


# ─────────────────────────────────────────────────────────────
# Web UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Cervical Cell Segmentation & Nuclei Identification") as demo:
    gr.Markdown(
        """
        # Cervical Cell Semantic Segmentation & MedSAM Nuclei Identification

        Upload a cervical-cell or histology image and select your segmentation model:
        - **Herlev (EffNet-B7)**: End-to-end multi-class segmentation (**Background**, **Cytoplasm**, **Nucleus**).
        - **MedSAM (ViT-B)**: Medical foundation model that identifies and **rounds off multiple nuclei together**—supports hand-marked annotations and interactive click accumulation!
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
            label="1. Upload raw cervical-cell image (or marked annotations)",
            type="numpy",
            image_mode="RGB",
        )

        output_image = gr.Image(
            label="2. Predicted segmentation — click here to select/add nuclei",
            type="numpy",
            interactive=False,  # Display only so cursor does not hide or clear image
        )

    selected_info = gr.Markdown(
        "Upload an image first. Then click 'Run Segmentation' or click on the predicted segmentation image."
    )

    with gr.Row():
        segment_button = gr.Button("Run Segmentation", variant="primary")
        reset_nuclei_button = gr.Button("Reset MedSAM Nuclei")
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

    reset_nuclei_button.click(
        fn=reset_medsam_nuclei,
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
        - Transparent yellow + Green boundary: Nuclei accurately segmented and rounded off together (both auto-detected / hand-marked & added via clicks)
        """
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_port=7860)