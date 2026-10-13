"""
Cell Division & Whole-Cell Boundary Segmentation Module (Mitosis / CellSAM / WHO Morphometry)

This module implements Whole-Cell Boundary (Plasma Membrane & Cytoplasm) segmentation
and Mitotic / Cell Division Figure detection and morphometric analysis in H&E histology.

Features:
1. Whole-Cell Boundary Segmentation: Segments the complete cellular boundary (cytoplasm + membrane),
   NOT merely the internal nucleus.
2. Dual Compartment Extraction: Dissects the cell into:
   - Outer Whole-Cell Boundary (Cell Membrane)
   - Inner Nuclear / Dividing Chromatin Core
   - Cytoplasm Body
3. Mitotic Figure & Cell Division Assessment:
   - Identifies active mitotic figures (prophase, metaphase, anaphase, telophase, cytokinesis).
   - Computes Nuclear-to-Cytoplasmic (N:C) Ratio.
   - Measures Cleavage Furrow & Cellular Elongation (Aspect Ratio).
   - Analyzes Chromatin Hyperchromasia & Condensation.
"""

import time
import math
import numpy as np
import cv2
import torch
from PIL import Image

MICRONS_PER_PIXEL = 0.5  # Standard 40x digital pathology magnification (0.5 um/pixel)


def compute_hematoxylin_od(image_rgb: np.ndarray) -> np.ndarray:
    """Ruifrok-Johnston optical density projection for hematoxylin absorption."""
    img_float = np.maximum(image_rgb.astype(np.float32), 1.0)
    od = -np.log10(img_float / 255.0)
    h_vec = np.array([0.650, 0.704, 0.286], dtype=np.float32)
    h_vec = h_vec / np.linalg.norm(h_vec)
    return np.maximum(np.dot(od, h_vec), 0.0)


def segment_whole_cell_at_point(
    predictor,
    image_rgb: np.ndarray,
    cx: int,
    cy: int,
    user_box: np.ndarray = None,
    herlev_model=None
) -> dict:
    """
    Given a user point prompt (+) at (cx, cy), extracts:
    - Whole-Cell Boundary (Plasma Membrane & Cytoplasm)
    - Nuclear / Dividing Chromatin Core
    - Cell Division & Mitosis Morphometry (N:C ratio, mitotic figure likelihood)
    """
    h, w = image_rgb.shape[:2]
    cx = max(0, min(int(cx), w - 1))
    cy = max(0, min(int(cy), h - 1))

    # 1. Prompt SAM for multi-mask levels (Level 0 = nucleus, Level 1/2 = whole cell)
    point_coords = np.array([[cx, cy]])
    point_labels = np.array([1])

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=user_box,
            multimask_output=True
        )

    # Sort masks by pixel area
    areas = [float(m.sum()) for m in masks]
    sorted_indices = np.argsort(areas)

    # Level 0 candidate: smallest realistic mask covering the chromatin core
    nuc_candidate = masks[sorted_indices[0]]

    # Whole-cell candidates: intermediate/larger masks
    # In SAM, Level 1 typically corresponds to the full cellular body / boundary
    whole_candidate = None
    for idx in sorted_indices[1:]:
        m = masks[idx]
        m_area = float(m.sum())
        # Whole cell should be at least 1.1x larger than nucleus and not consume whole image
        if m_area >= (areas[sorted_indices[0]] * 1.15) and m_area <= (h * w * 0.45):
            whole_candidate = m
            break

    if whole_candidate is None:
        whole_candidate = masks[sorted_indices[-1]]

    # Ensure nucleus is subset of whole cell
    whole_cell_mask = whole_candidate.copy().astype(bool)
    nuc_mask = (nuc_candidate & whole_cell_mask).astype(bool)

    # If nucleus is too small or absent, use optical density thresholding within whole cell
    if nuc_mask.sum() < 12:
        h_od = compute_hematoxylin_od(image_rgb)
        if whole_cell_mask.any():
            cell_od = h_od[whole_cell_mask]
            thresh = np.percentile(cell_od, 70) if len(cell_od) > 0 else 0.2
            nuc_mask = whole_cell_mask & (h_od > max(0.15, thresh))

    # Keep only connected component closest to click
    def keep_nearest_component(binary_mask, px, py):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8))
        if num_labels <= 1:
            return binary_mask
        best_lbl = 1
        min_d = float("inf")
        for lbl in range(1, num_labels):
            cnt_x, cnt_y = centroids[lbl]
            d = (cnt_x - px) ** 2 + (cnt_y - py) ** 2
            if d < min_d:
                min_d = d
                best_lbl = lbl
        return labels == best_lbl

    whole_cell_mask = keep_nearest_component(whole_cell_mask, cx, cy)
    nuc_mask = keep_nearest_component(nuc_mask, cx, cy) & whole_cell_mask

    # Morphological smoothing for natural cellular membrane boundaries
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    whole_smooth = cv2.morphologyEx(whole_cell_mask.astype(np.uint8), cv2.MORPH_CLOSE, k_close)
    whole_cnts, _ = cv2.findContours(whole_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not whole_cnts:
        return None
    whole_contour = max(whole_cnts, key=cv2.contourArea)

    nuc_smooth = cv2.morphologyEx(nuc_mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    nuc_cnts, _ = cv2.findContours(nuc_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nuc_contour = max(nuc_cnts, key=cv2.contourArea) if nuc_cnts else None

    # Cytoplasm compartment = Whole cell - Nucleus
    cyto_mask = whole_cell_mask & (~nuc_mask)

    whole_area_px = int(cv2.contourArea(whole_contour))
    nuc_area_px = int(cv2.contourArea(nuc_contour)) if nuc_contour is not None else int(nuc_mask.sum())
    cyto_area_px = max(1, whole_area_px - nuc_area_px)

    whole_area_um2 = round(whole_area_px * (MICRONS_PER_PIXEL ** 2), 1)
    nuc_area_um2 = round(nuc_area_px * (MICRONS_PER_PIXEL ** 2), 1)
    cyto_area_um2 = round(cyto_area_px * (MICRONS_PER_PIXEL ** 2), 1)

    # Nuclear-to-Cytoplasmic (N:C) Ratio
    nc_ratio = round(float(nuc_area_px) / float(max(1, cyto_area_px)), 2)

    # Shape metrics (Circularity & Aspect Ratio / Cleavage Furrow)
    perimeter = float(cv2.arcLength(whole_contour, True))
    circularity = round(float(4.0 * np.pi * whole_area_px / (perimeter ** 2 + 1e-6)), 3)

    rect = cv2.minAreaRect(whole_contour)
    rw, rh = rect[1]
    aspect_ratio = round(float(max(rw, rh) / max(1.0, min(rw, rh))), 2)

    # 2. Cell Division & Mitosis Analysis
    h_od = compute_hematoxylin_od(image_rgb)
    nuc_od_values = h_od[nuc_mask] if nuc_mask.any() else np.array([0.0])
    mean_chromatin_od = float(np.mean(nuc_od_values))
    std_chromatin_od = float(np.std(nuc_od_values))

    # Mitotic figures exhibit:
    # 1. Chromatin hyperchromasia & dense clumping (high mean_chromatin_od > 0.40)
    # 2. Elevated N:C ratio (> 0.45)
    # 3. Spindle elongation or cleavage indentation (aspect_ratio >= 1.45 or circularity < 0.65)
    # 4. Irregular chromatin boundary (lack of smooth nuclear envelope)
    mitosis_score = 0.0
    if mean_chromatin_od > 0.35:
        mitosis_score += min(0.40, (mean_chromatin_od - 0.35) * 1.5)
    if nc_ratio > 0.45:
        mitosis_score += min(0.35, (nc_ratio - 0.45) * 0.8)
    if aspect_ratio >= 1.45:
        mitosis_score += min(0.25, (aspect_ratio - 1.45) * 0.25)
    mitosis_score = round(min(0.98, max(0.12, mitosis_score)), 2)

    is_mitotic = (mitosis_score >= 0.50)

    if is_mitotic:
        if aspect_ratio >= 1.60 and circularity < 0.62:
            division_stage = "Dividing Cell (Cytokinesis / Cleavage Furrow)"
            stage_badge = "🌱"
        elif mean_chromatin_od > 0.45:
            division_stage = "Active Mitotic Figure (Metaphase / Anaphase Chromatin Plate)"
            stage_badge = "⚡"
        else:
            division_stage = "Early Prophase / Mitotic Condensation"
            stage_badge = "🔬"
    else:
        division_stage = "Interphase / Non-Dividing Quiescent Cell"
        stage_badge = "🔵"

    bx, by, bw, bh = cv2.boundingRect(whole_contour)
    whole_centroid = (int(rect[0][0]), int(rect[0][1]))

    return {
        "id": 1,
        "whole_cell_mask": whole_cell_mask,
        "nuc_mask": nuc_mask,
        "cyto_mask": cyto_mask,
        "whole_contour": whole_contour,
        "nuc_contour": nuc_contour,
        "whole_area_px": whole_area_px,
        "whole_area_um2": whole_area_um2,
        "nuc_area_px": nuc_area_px,
        "nuc_area_um2": nuc_area_um2,
        "cyto_area_px": cyto_area_px,
        "cyto_area_um2": cyto_area_um2,
        "nc_ratio": nc_ratio,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "mean_chromatin_od": round(mean_chromatin_od, 3),
        "is_mitotic": is_mitotic,
        "mitosis_score": mitosis_score,
        "division_stage": division_stage,
        "stage_badge": stage_badge,
        "centroid": whole_centroid,
        "bbox": [bx, by, bx + bw, by + bh]
    }


def scan_candidate_dividing_cells(predictor, image_rgb: np.ndarray, max_candidates: int = 15) -> list:
    """
    Scans the field for hyperchromatic condensed chromatin cores (candidate mitotic figures)
    and segments their complete cellular boundaries.
    """
    h, w = image_rgb.shape[:2]
    h_od = compute_hematoxylin_od(image_rgb)
    h_smooth = cv2.GaussianBlur(h_od, (5, 5), 0)

    # Discard non-tissue background
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    tissue_mask = gray < 235

    if not tissue_mask.any():
        return []

    # High OD threshold for condensed chromatin
    thresh_val = np.percentile(h_smooth[tissue_mask], 85)
    dense_chromatin = (h_smooth > max(0.25, thresh_val)) & tissue_mask

    num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(dense_chromatin.astype(np.uint8))
    candidates = []

    for lbl in range(1, num_l):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 15 or area > (h * w * 0.05):
            continue
        cx, cy = int(centroids[lbl][0]), int(centroids[lbl][1])
        mean_od = float(np.mean(h_od[labels == lbl]))
        candidates.append((cx, cy, mean_od))

    # Sort by chromatin density
    candidates.sort(key=lambda item: item[2], reverse=True)
    candidates = candidates[:max_candidates]

    instances = []
    inst_id = 1
    for cx, cy, _ in candidates:
        # Avoid duplicate overlapping centers
        if any((cx - inst["centroid"][0]) ** 2 + (cy - inst["centroid"][1]) ** 2 < 256 for inst in instances):
            continue
        try:
            cell_data = segment_whole_cell_at_point(predictor, image_rgb, cx, cy)
            if cell_data is not None:
                cell_data["id"] = inst_id
                instances.append(cell_data)
                inst_id += 1
        except Exception as e:
            print("Scan cell error:", e)

    return instances


def render_celldivision_display(image_rgb: np.ndarray, instances: list, selected_id: int = None) -> np.ndarray:
    """
    Renders visual overlay showing:
    - Outer Whole-Cell Boundary (Plasma Membrane) in bold Emerald / Amber outline
    - Translucent Cytoplasm Shading
    - Inner Nuclear / Dividing Chromatin Core in Violet / Cyan outline
    - High-contrast glowing highlight on selected cell
    """
    overlay = image_rgb.copy()
    if not instances:
        return overlay

    # Create translucent cytoplasm fills
    cyto_fill = np.zeros_like(overlay)
    nuc_fill = np.zeros_like(overlay)

    for inst in instances:
        wc_cnt = inst.get("whole_contour")
        nuc_cnt = inst.get("nuc_contour")
        is_mit = inst.get("is_mitotic", False)

        # Cytoplasm color: Amber for dividing/mitotic, Emerald for quiescent
        cyto_col = (245, 158, 11) if is_mit else (16, 185, 129)
        # Nucleus color: Violet/Magenta for mitotic, Dark Blue for resting
        nuc_col = (168, 85, 247) if is_mit else (59, 130, 246)

        if wc_cnt is not None:
            cv2.drawContours(cyto_fill, [wc_cnt], -1, cyto_col, -1)
        if nuc_cnt is not None:
            cv2.drawContours(nuc_fill, [nuc_cnt], -1, nuc_col, -1)

    has_cyto = cyto_fill.any(axis=-1)
    overlay[has_cyto] = (
        0.70 * overlay[has_cyto].astype(float)
        + 0.30 * cyto_fill[has_cyto].astype(float)
    ).astype(np.uint8)

    has_nuc = nuc_fill.any(axis=-1)
    overlay[has_nuc] = (
        0.65 * overlay[has_nuc].astype(float)
        + 0.35 * nuc_fill[has_nuc].astype(float)
    ).astype(np.uint8)

    # Draw crisp outer whole-cell boundaries
    for inst in instances:
        wc_cnt = inst.get("whole_contour")
        nuc_cnt = inst.get("nuc_contour")
        is_mit = inst.get("is_mitotic", False)
        is_sel = (selected_id is not None and inst.get("id") == selected_id)

        cyto_col = (245, 158, 11) if is_mit else (16, 185, 129)
        nuc_col = (168, 85, 247) if is_mit else (59, 130, 246)

        if wc_cnt is not None:
            if is_sel:
                # Glowing dual outline for selected cell boundary
                cv2.drawContours(overlay, [wc_cnt], -1, (255, 235, 0), 4)
                cv2.drawContours(overlay, [wc_cnt], -1, (255, 255, 255), 2)
                cx, cy = inst["centroid"]
                cv2.circle(overlay, (cx, cy), 5, (255, 235, 0), -1)
                cv2.circle(overlay, (cx, cy), 7, (0, 0, 0), 1)
            else:
                cv2.drawContours(overlay, [wc_cnt], -1, cyto_col, 2)

        if nuc_cnt is not None:
            cv2.drawContours(overlay, [nuc_cnt], -1, nuc_col, 2)

    return overlay


def format_celldivision_markdown(inst: dict, total_cells: int = 1) -> str:
    """Formats clinical morphometry report for whole-cell boundary and mitosis."""
    if not inst:
        return "### Cell Division & Boundary Analyzer: No cell selected yet. Click any cell with '+' to inspect!"

    cid = inst.get("id", 1)
    wc_um2 = inst.get("whole_area_um2", 0)
    wc_px = inst.get("whole_area_px", 0)
    nuc_um2 = inst.get("nuc_area_um2", 0)
    cyto_um2 = inst.get("cyto_area_um2", 0)
    nc_ratio = inst.get("nc_ratio", 0)
    circ = inst.get("circularity", 0)
    asp = inst.get("aspect_ratio", 0)
    stage = inst.get("division_stage", "Quiescent")
    badge = inst.get("stage_badge", "🔬")
    score = inst.get("mitosis_score", 0)
    cx, cy = inst.get("centroid", (0, 0))

    report = [
        f"### {badge} Cell Boundary & Division Report — **Instance #{cid}**\n\n",
        f"- **Cell Division Status**: {badge} **{stage}**\n",
        f"- **Mitotic Activity Index / Likelihood**: `{score * 100:.1f}%`\n",
        f"- **Selected Position**: `x={cx}, y={cy}`\n\n",
        "#### Whole-Cell Compartmental Morphometry (Outer Membrane vs Nucleus):\n",
        f"| Cellular Compartment | Area (µm²) | Area (Pixels) | Visual Contour |\n",
        f"| :--- | :---: | :---: | :---: |\n",
        f"| 🟡 **Whole Cell Boundary (Plasma Membrane)** | **`{wc_um2} µm²`** | `{wc_px:,} px` | Outer Amber Outline |\n",
        f"| 🟣 **Nuclear / Dividing Chromatin Core** | **`{nuc_um2} µm²`** | `{inst.get('nuc_area_px', 0):,} px` | Inner Violet Outline |\n",
        f"| 🟢 **Cytoplasm Body** | **`{cyto_um2} µm²`** | `{inst.get('cyto_area_px', 0):,} px` | Translucent Interior |\n\n",
        "#### Morphological Indices:\n",
        f"- **Nuclear-to-Cytoplasmic (N:C) Ratio**: `1 : {nc_ratio:.2f}` (Mitotic figures typically exceed 0.50)\n",
        f"- **Cellular Elongation / Aspect Ratio**: `{asp}` (Elongation indicates mitotic cleavage furrow)\n",
        f"- **Boundary Circularity**: `{circ}` (1.0 = perfect sphere)\n\n",
        "> **Whole-Cell Boundary Segmentation Active**: The outer contour marks the complete cell boundary (cytoplasm + plasma membrane), not just the nucleus. Click **'Extract Cell Cutout'** below to isolate this cell!"
    ]

    return "".join(report)
