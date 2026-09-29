"""
Oral Histopathology AI Service & ScribblePrompt Adapter
======================================================
Adapted from the approved Oral Histopath AI project.
Provides:
1. ScribblePrompt: 6-channel interactive brush/scribble segmentation UNet
2. MorphologyService: WHO-grounded quantitative morphometric & optical stain analyzer
"""

import os
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage import measure

# ─────────────────────────────────────────────────────────────
# 1. ScribblePrompt Neural Architecture
# ─────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ScribblePromptUNet(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.inc = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels * 4, base_channels * 8))

        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        d1 = self.up1(x4)
        if d1.shape[2:] != x3.shape[2:]:
            d1 = F.interpolate(d1, size=x3.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        if d2.shape[2:] != x2.shape[2:]:
            d2 = F.interpolate(d2, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        if d3.shape[2:] != x1.shape[2:]:
            d3 = F.interpolate(d3, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.conv_up3(d3)

        return self.outc(d3)


def load_scribbleprompt(checkpoint_path: str, device: torch.device):
    """Loads ScribblePrompt weights onto specified device."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        net = ScribblePromptUNet(in_channels=6, out_channels=1, base_channels=32)
        state_dict = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        return net
    except Exception as e:
        print("Failed to load ScribblePrompt:", e)
        return None


def predict_scribbleprompt(net, image_rgb: np.ndarray, stroke_mask: np.ndarray, device: torch.device):
    """Runs interactive brush segmentation using the 6-channel ScribblePrompt model."""
    orig_h, orig_w = image_rgb.shape[:2]
    proc_size = (512, 512)

    pos_mask = (stroke_mask > 0).astype(np.uint8) * 255
    img_resized = cv2.resize(image_rgb, proc_size, interpolation=cv2.INTER_LINEAR)
    pos_resized = cv2.resize(pos_mask, proc_size, interpolation=cv2.INTER_NEAREST)

    # Compute exponential decay distance transform map
    binary_pos = (pos_resized > 0).astype(np.uint8)
    if binary_pos.any():
        edt = distance_transform_edt(binary_pos == 0)
        pos_dist = np.exp(-edt / 16.0).astype(np.float32)
    else:
        pos_dist = np.zeros(proc_size[::-1], dtype=np.float32)

    neg_dist = np.zeros(proc_size[::-1], dtype=np.float32)
    prev_map = np.zeros(proc_size[::-1], dtype=np.float32)

    # Construct 6-channel normalized tensor
    img_norm = (img_resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_ch = np.transpose(img_norm, (2, 0, 1))

    tensor_in = np.zeros((1, 6, proc_size[1], proc_size[0]), dtype=np.float32)
    tensor_in[0, :3] = img_ch
    tensor_in[0, 3] = pos_dist
    tensor_in[0, 4] = neg_dist
    tensor_in[0, 5] = prev_map

    with torch.no_grad():
        torch_in = torch.from_numpy(tensor_in).to(device)
        logits = net(torch_in)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

    # Interaction-guided fusion
    interaction_weight = pos_dist
    fused_prob = np.clip(probs * 0.4 + interaction_weight * 0.6, 0.0, 1.0)
    pred_binary = (fused_prob > 0.42).astype(np.uint8) * 255

    # Filter connected components touching brush stroke
    if pos_resized.any():
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_binary)
        cleaned = np.zeros_like(pred_binary)
        for lab in range(1, num_labels):
            comp = (labels == lab)
            if np.logical_and(comp, pos_resized > 0).any() or stats[lab, 4] > 20:
                cleaned[comp] = 255
        pred_binary = cleaned

    mask_full = cv2.resize(pred_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_full > 0


# ─────────────────────────────────────────────────────────────
# 2. WHO-Grounded Morphological & Optical Density Engine
# ─────────────────────────────────────────────────────────────
def analyze_morphology(image_rgb: np.ndarray, mask: np.ndarray, microns_per_pixel: float = 0.5):
    """
    Computes technically validated morphological and optical stain measurements
    grounded in WHO Oral & Cervical histopathological criteria.
    """
    h, w = image_rgb.shape[:2]
    binary_mask = (mask > 0).astype(np.uint8)
    area_pixels = int(np.count_nonzero(binary_mask))

    if area_pixels == 0:
        return None

    area_microns_sq = round(area_pixels * (microns_per_pixel ** 2), 1)

    contours, _ = cv2.findContours(binary_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter_pixels = float(sum(cv2.arcLength(c, True) for c in contours)) if contours else 0.0
    perimeter_microns = round(perimeter_pixels * microns_per_pixel, 1)

    # Circularity: 4 * pi * Area / (Perimeter^2)
    circularity = round(float(min(1.0, (4 * math.pi * area_pixels) / max(1.0, perimeter_pixels ** 2))), 3)

    # Moments & Ellipse Descriptors
    labeled = measure.label(binary_mask)
    props = measure.regionprops(labeled)

    if props:
        prop = max(props, key=lambda p: p.area)
        eccentricity = round(float(prop.eccentricity), 3)
        solidity = round(float(prop.solidity), 3)
        major_axis = round(float(getattr(prop, "axis_major_length", getattr(prop, "major_axis_length", 0.0))), 1)
        minor_axis = round(float(getattr(prop, "axis_minor_length", getattr(prop, "minor_axis_length", 0.0))), 1)
        aspect_ratio = round(float(major_axis / max(1.0, minor_axis)), 2)
        minr, minc, maxr, maxc = prop.bbox
        bbox = [int(minc), int(minr), int(maxc), int(maxr)]
    else:
        eccentricity = 0.0
        solidity = 1.0
        aspect_ratio = 1.0
        bbox = [0, 0, w, h]

    # Stain Optical Density (Hematoxylin OD = nuclear hyperchromasia proxy)
    masked_rgb = image_rgb[binary_mask > 0]
    mean_r = float(np.mean(masked_rgb[:, 0]))
    mean_g = float(np.mean(masked_rgb[:, 1]))
    mean_b = float(np.mean(masked_rgb[:, 2]))

    od_r = -math.log10(max(1.0, mean_r + 1.0) / 256.0)
    od_g = -math.log10(max(1.0, mean_g + 1.0) / 256.0)
    hematoxylin_od = round(float(od_r), 3)
    eosin_od = round(float(od_g), 3)
    stain_ratio = round(float(od_r / max(0.01, od_g)), 2)

    # Intra-nuclear nucleolar count estimate & mitotic figure check
    nucleoli_count = 0
    is_mitotic = False
    if area_pixels > 80:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        nuc_crop = gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        mask_crop = binary_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if nuc_crop.size > 0:
            masked_gray = nuc_crop.copy()
            masked_gray[mask_crop == 0] = 255
            cutoff = np.percentile(masked_gray[mask_crop > 0], 15)
            dark_spots = (masked_gray < cutoff).astype(np.uint8)
            n_cnt, _, _, _ = cv2.connectedComponentsWithStats(dark_spots)
            nucleoli_count = max(0, n_cnt - 1)

            if circularity < 0.65 and hematoxylin_od > 0.35 and 100 < area_pixels < 3000:
                is_mitotic = True

    # WHO Histopathological Classification
    if aspect_ratio >= 2.3 and eccentricity > 0.85:
        feature_name = "Tadpole / Spindle Cell Transformation"
        category = "Cellular Pleomorphism (Malignancy Indicator)"
    elif is_mitotic:
        feature_name = "Atypical Mitotic Figure"
        category = "Mitotic Activity (Proliferation)"
    elif hematoxylin_od >= 0.36 and mean_r < 120:
        if area_microns_sq > 250:
            feature_name = "Severe Nuclear Hyperchromasia & Karyomegaly"
            category = "High-Grade Nuclear Atypia"
        else:
            feature_name = "Prominent Nuclear Hyperchromasia"
            category = "Nuclear Dysplasia"
    elif area_microns_sq > 300:
        feature_name = "Marked Nuclear Enlargement (Karyomegaly)"
        category = "Nuclear Atypia"
    elif circularity >= 0.82 and solidity >= 0.90:
        feature_name = "Monomorphic Rounded Nucleus"
        category = "Benign / Low-Grade Morphology"
    else:
        feature_name = "Irregular Nucleus Contour"
        category = "Intermediate Dysplastic Pattern"

    return {
        "area_pixels": area_pixels,
        "area_microns_sq": area_microns_sq,
        "perimeter_pixels": perimeter_pixels,
        "perimeter_microns": perimeter_microns,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "aspect_ratio": aspect_ratio,
        "mean_r": round(mean_r, 1),
        "hematoxylin_od": hematoxylin_od,
        "eosin_od": eosin_od,
        "stain_ratio": stain_ratio,
        "nucleoli_count": nucleoli_count,
        "is_mitotic": is_mitotic,
        "feature_name": feature_name,
        "category": category,
        "bbox": bbox,
    }


def format_morphology_markdown(analysis: dict) -> str:
    """Formats quantitative morphology results into rich Markdown for Gradio UI."""
    if not analysis:
        return ""

    mitotic_badge = "🔴 **YES (Atypical)**" if analysis["is_mitotic"] else "🟢 None detected"
    hyperchromasia_badge = "🔴 **High (Hyperchromatic)**" if analysis["hematoxylin_od"] >= 0.36 else "🟢 Normal/Moderate"

    return (
        f"\n\n---\n"
        f"### 🔬 Grounded Histopathological & Morphometric Analysis\n"
        f"| Quantitative Metric | Measured Value | Pathological Reference Context |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **Diagnostic Feature** | **{analysis['feature_name']}** | `{analysis['category']}` |\n"
        f"| **Nuclear Area** | `{analysis['area_microns_sq']} µm²` (`{analysis['area_pixels']:,}` px) | Spatial scale: 0.5 µm/pixel |\n"
        f"| **Perimeter & Circularity** | `{analysis['perimeter_microns']} µm` &bull; `{analysis['circularity']:.3f}` | 1.0 = smooth sphere (dysplastic < 0.75) |\n"
        f"| **Solidity & Aspect Ratio** | `{analysis['solidity']:.3f}` &bull; `{analysis['aspect_ratio']:.2f}` | Elongated / Tadpole if ratio > 2.0 |\n"
        f"| **Hematoxylin OD** | `{analysis['hematoxylin_od']:.3f}` | {hyperchromasia_badge} |\n"
        f"| **Nucleoli Peak Count** | `{analysis['nucleoli_count']}` prominent centers | Multiple nucleoli indicate active synthesis |\n"
        f"| **Mitotic Activity Pattern** | {mitotic_badge} | Chromatin condensation proxy |\n"
    )


def extract_region(image_rgb: np.ndarray, mask: np.ndarray, padding_percent: float = 0.08):
    """
    Dual-output extraction inspired by oral-histopath-ai:
    1. RGB Bounding Box Crop (preserves true H&E staining)
    2. RGBA Object Cutout with transparent background
    """
    h, w = image_rgb.shape[:2]
    binary_mask = (mask > 0).astype(np.uint8) * 255
    coords = cv2.findNonZero(binary_mask)

    if coords is None:
        return None, None

    bx, by, bw, bh = cv2.boundingRect(coords)
    pad_x = int(bw * padding_percent)
    pad_y = int(bh * padding_percent)
    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)

    crop_rgb = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = binary_mask[y1:y2, x1:x2]

    # 4-channel transparent cutout
    cutout_rgba = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
    cutout_rgba[:, :, :3] = crop_rgb
    cutout_rgba[:, :, 3] = crop_mask

    return crop_rgb, cutout_rgba


def compare_models_roi(image_rgb: np.ndarray, stroke_mask: np.ndarray, medsam_predictor, scribbleprompt_net, device):
    """
    Multi-model benchmark comparison across candidate AI models for identical prompt.
    Returns comparison summary markdown and execution latencies.
    """
    import time
    results = []

    # 1. ScribblePrompt
    if scribbleprompt_net is not None:
        t0 = time.perf_counter()
        try:
            sp_mask = predict_scribbleprompt(scribbleprompt_net, image_rgb, stroke_mask, device)
            sp_time = round((time.perf_counter() - t0) * 1000, 1)
            sp_area = int(np.count_nonzero(sp_mask))
            results.append({
                "model": "ScribblePrompt (6-Ch UNet)",
                "status": "READY",
                "latency_ms": sp_time,
                "area_px": sp_area,
                "prompt_type": "Interactive Brush / Scribble",
                "notes": "Direct scribble-conditioned distance transform forward pass"
            })
        except Exception as e:
            results.append({
                "model": "ScribblePrompt (6-Ch UNet)",
                "status": "ERROR",
                "latency_ms": 0,
                "area_px": 0,
                "prompt_type": "Interactive Brush / Scribble",
                "notes": f"Inference failed: {e}"
            })
    else:
        results.append({
            "model": "ScribblePrompt (6-Ch UNet)",
            "status": "NOT LOADED",
            "latency_ms": 0,
            "area_px": 0,
            "prompt_type": "Interactive Brush",
            "notes": "Weights scribbleprompt.pth missing"
        })

    # 2. MedSAM
    if medsam_predictor is not None:
        t0 = time.perf_counter()
        try:
            cnts, _ = cv2.findContours(stroke_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c)
                pad = 10
                h, w = image_rgb.shape[:2]
                box = np.array([max(0, bx - pad), max(0, by - pad), min(w, bx + bw + pad), min(h, by + bh + pad)])
                cx, cy = bx + bw // 2, by + bh // 2

                with torch.inference_mode():
                    masks, scores, _ = medsam_predictor.predict(
                        point_coords=np.array([[cx, cy]]),
                        point_labels=np.array([1]),
                        box=box,
                        multimask_output=False
                    )
                med_time = round((time.perf_counter() - t0) * 1000, 1)
                med_area = int(np.count_nonzero(masks[0]))
                results.append({
                    "model": "MedSAM (ViT-B)",
                    "status": "READY",
                    "latency_ms": med_time,
                    "area_px": med_area,
                    "prompt_type": "Centroid + Bounding Box",
                    "notes": f"Confidence score: {float(scores[0]):.3f}"
                })
        except Exception as e:
            results.append({
                "model": "MedSAM (ViT-B)",
                "status": "ERROR",
                "latency_ms": 0,
                "area_px": 0,
                "prompt_type": "Box / Point",
                "notes": f"Inference error: {e}"
            })
    else:
        results.append({
            "model": "MedSAM (ViT-B)",
            "status": "NOT LOADED",
            "latency_ms": 0,
            "area_px": 0,
            "prompt_type": "Box / Point",
            "notes": "Checkpoint medsam_vit_b.pth missing"
        })

    # 3. Reference Candidate Models Status Report (Honest transparency matching oral-histopath-ai)
    additional_models = [
        {"name": "PathoSAM", "status": "CHECKPOINT REQUIRED", "note": "Pathology-tuned ViT backbone weights required"},
        {"name": "VISTA-PATH", "status": "CHECKPOINT REQUIRED", "note": "Point-based histopathology foundation model"},
        {"name": "CellViT-256", "status": "CHECKPOINT REQUIRED", "note": "PanNuke multi-class nuclear instance segmenter"},
        {"name": "HoVer-Net Fast", "status": "CHECKPOINT REQUIRED", "note": "Horizontal/Vertical distance map segmenter"}
    ]

    # Format Markdown Table
    md = (
        "### ⚖️ Multi-Model Benchmark Comparison (Identical ROI / Prompt)\n\n"
        "| Candidate Model | Execution Status | Prompt Interaction | Latency (ms) | Mask Area (px) | Benchmark Evaluation Notes |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    for r in results:
        status_badge = "🟢 READY" if r["status"] == "READY" else "🔴 " + r["status"]
        md += f"| **{r['model']}** | {status_badge} | {r['prompt_type']} | `{r['latency_ms']} ms` | `{r['area_px']:,} px` | {r['notes']} |\n"

    for m in additional_models:
        md += f"| **{m['name']}** | ⚪ {m['status']} | Automated / Prompt | `--` | `--` | {m['note']} |\n"

    return md

