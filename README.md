# Cervical Cell Segmentation & MedSAM Multi-Nuclei Identification

A web application built with [Gradio](https://gradio.app/) for automated and interactive cervical cell semantic segmentation, featuring:
1. **Herlev Model (EfficientNet-B7 + FPN)**: End-to-end multi-class semantic segmentation predicting **Background**, **Cytoplasm**, and **Nucleus**.
2. **MedSAM (Medical SAM ViT-B)**: Medical foundation model capable of identifying **multiple nuclei together**—both via automatic batch detection and interactive point-prompt accumulation.

---

## Features

- **Dual-Model Selection**: Toggle smoothly between Herlev FPN and MedSAM with real-time UI updates.
- **Herlev Multi-Class Segmentation**: Accurately colors regions:
  - 🔴 Red = Background
  - 🔵 Dark Blue = Cytoplasm
  - 🔷 Light Blue = Nucleus
  - 🟡 Clicking on any nucleus or cytoplasm highlights all regions of that class across the entire image in transparent yellow.
- **MedSAM Multi-Nuclei Identification**:
  - Automatically identifies candidate dark nuclei across the cytology image and segments them together in a single batch forward pass.
  - Interactive click mode: click any additional or missed nucleus to segment and add it to the multi-nucleus mask.
  - All identified nuclei remain highlighted together with numbered labels, count badge, and total pixel area.
- **Stable Interactive Canvas**: Cursor hover and clicks do not clear or reset the image viewport.
- **Reset Button**: Quickly clear accumulated MedSAM points and nuclei without re-uploading the image.

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/electromist/herlev_web_app.git
cd herlev_web_app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Weights
The model weights are hosted on [GitHub Releases v1.0.0](https://github.com/electromist/herlev_web_app/releases/tag/v1.0.0):

| Model | File | Size | Download Link |
|---|---|---|---|
| **Herlev EffNet-B7 + FPN** | `herlev_effnetb7_fpn.pth` | ~257 MB | [Download](https://github.com/electromist/herlev_web_app/releases/download/v1.0.0/herlev_effnetb7_fpn.pth) |
| **MedSAM ViT-B** | `medsam_vit_b.pth` | ~375 MB | [Download](https://github.com/electromist/herlev_web_app/releases/download/v1.0.0/medsam_vit_b.pth) |

You can download them directly into the repository directory using PowerShell or terminal:

**Windows PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://github.com/electromist/herlev_web_app/releases/download/v1.0.0/herlev_effnetb7_fpn.pth" -OutFile "herlev_effnetb7_fpn.pth"
Invoke-WebRequest -Uri "https://github.com/electromist/herlev_web_app/releases/download/v1.0.0/medsam_vit_b.pth" -OutFile "medsam_vit_b.pth"
```

**Linux / macOS (curl):**
```bash
curl -L -o herlev_effnetb7_fpn.pth "https://github.com/electromist/herlev_web_app/releases/download/v1.0.0/herlev_effnetb7_fpn.pth"
curl -L -o medsam_vit_b.pth "https://github.com/electromist/herlev_web_app/releases/download/v1.0.0/medsam_vit_b.pth"
```

---

## Running the Web App

Launch the application:
```bash
python app.py
```
Open your browser at:
```
http://127.0.0.1:7860
```
A sample cytology image is included in `samples/sample_cell.png` for quick testing.

---

<p align="center"><sub>Developed by Harsh Malakar, Anurag Dey Sarkar, and Morris Bhagat as a team</sub></p>
