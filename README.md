# Cervical Cell Segmentation & MedSAM Multi-Nuclei Identification

A Gradio-based web application for automated and interactive cervical cell semantic segmentation, featuring:
1. **Herlev Model (EfficientNet-B7 + FPN)**: End-to-end multi-class semantic segmentation predicting Background, Cytoplasm, and Nucleus.
2. **MedSAM (Medical SAM ViT-B)**: Medical foundation model capable of identifying multiple nuclei together—both via automatic batch detection and interactive point-prompt accumulation.

---

## Features

- **Dual-Model Support**: Easily switch between the Herlev FPN model and MedSAM via radio buttons.
- **Herlev Semantic Segmentation**: Automatically partitions images into Background (red), Cytoplasm (dark blue), and Nucleus (light blue). Clicking any region highlights all pixels of that class in yellow.
- **MedSAM Multi-Nuclei Identification**:
  - Automatically detects candidate dark nuclei across the cytology image and segments them together in a single batch pass.
  - Interactive click prompting allows you to click on any additional nuclei to segment and accumulate them into the total mask.
  - Numbered markers and real-time statistics (total nuclei count, confidence score, combined pixel area).
- **Interactive Display**: Safe viewport preventing image disappearance on cursor hover or click.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/electromist/herlev_web_app.git
   cd herlev_web_app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Model Weights Setup

Due to GitHub file size limits, the pre-trained weights are hosted separately:
- `herlev_effnetb7_fpn.pth` (~257 MB)
- `medsam_vit_b.pth` (~375 MB)

Place both `.pth` files into the root directory alongside `app.py`:
```
herlev_web_app/
├── app.py
├── herlev_effnetb7_fpn.pth  <-- Place here
├── medsam_vit_b.pth         <-- Place here
└── requirements.txt
```

---

## Running the Web App

Run the application:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.
