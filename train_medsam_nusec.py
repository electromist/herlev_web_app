"""
Fine-tuning MedSAM on NuSeC & MiDeSeC Histopathology Datasets
=============================================================
This script fine-tunes the MedSAM Prompt Encoder and Mask Decoder on the
Ankara University NuSeC (Nuclei Segmentation) and MiDeSeC (Mitosis Detection)
datasets, making MedSAM significantly more accurate on histological nuclei,
curved arcs, and doctor brush strokes.

Usage:
------
Run on your local NVIDIA RTX 5050 GPU:
  python train_medsam_nusec.py --epochs 10 --batch_size 4 --lr 1e-4

Or run on Kaggle / Google Colab with GPU:
  python train_medsam_nusec.py --data_dir /kaggle/input/nusec-and-midesec/Ankara\\ University\\ Datasets
"""

import os
import glob
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry


# ─────────────────────────────────────────────────────────────
# 1. Dataset Loader for NuSeC & MiDeSeC
# ─────────────────────────────────────────────────────────────
class NucleiDataset(Dataset):
    """
    Unified dataset for:
    1) NuSeC: 1024x1024 images + multi-instance label masks (TIF)
    2) MiDeSeC: 1024x1024 images + CSV polygon boundary coordinates
    """
    def __init__(self, data_root, split="train", img_size=1024, max_samples=None):
        self.samples = []
        self.img_size = img_size
        
        # Paths for NuSeC
        nusec_root = os.path.join(data_root, "NuSeC")
        if split == "train":
            nusec_img_dir = os.path.join(nusec_root, "train nuclei")
            nusec_mask_dir = os.path.join(nusec_root, "mask of train nuclei")
        else:
            nusec_img_dir = os.path.join(nusec_root, "test nuclei")
            nusec_mask_dir = os.path.join(nusec_root, "mask of test nuclei")
            
        if os.path.exists(nusec_img_dir) and os.path.exists(nusec_mask_dir):
            for img_name in sorted(os.listdir(nusec_img_dir)):
                img_path = os.path.join(nusec_img_dir, img_name)
                mask_path = os.path.join(nusec_mask_dir, img_name)
                if os.path.exists(mask_path):
                    self.samples.append({
                        "type": "nusec",
                        "img_path": img_path,
                        "mask_path": mask_path,
                    })

        # Paths for MiDeSeC
        midesec_root = os.path.join(data_root, "MiDeSeC")
        midesec_dir = os.path.join(midesec_root, f"{split} images")
        if os.path.exists(midesec_dir):
            csv_files = sorted(glob.glob(os.path.join(midesec_dir, "*.csv")))
            for c in csv_files:
                base = os.path.splitext(c)[0]
                jpg = base + ".jpg"
                bmp = base + ".bmp"
                img_path = jpg if os.path.exists(jpg) else (bmp if os.path.exists(bmp) else None)
                if img_path:
                    self.samples.append({
                        "type": "midesec",
                        "img_path": img_path,
                        "csv_path": c,
                    })
                    
        if max_samples:
            self.samples = self.samples[:max_samples]
            
        print(f"[{split.upper()}] Loaded {len(self.samples)} images ({sum(1 for s in self.samples if s['type']=='nusec')} NuSeC, {sum(1 for s in self.samples if s['type']=='midesec')} MiDeSeC)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.imread(item["img_path"])
        if img is None:
            raise RuntimeError(f"Cannot read image: {item['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        nuclei_masks = []
        
        if item["type"] == "nusec":
            mask = cv2.imread(item["mask_path"], cv2.IMREAD_UNCHANGED)
            if mask is not None:
                lbls = np.unique(mask)
                for lbl in lbls:
                    if lbl == 0:
                        continue
                    nuc = (mask == lbl)
                    if nuc.sum() >= 15:  # filter tiny artifacts
                        nuclei_masks.append(nuc)
        else:
            with open(item["csv_path"], "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                parts = [float(x) for x in line.split(",") if x.strip()]
                if len(parts) >= 6:
                    pts = np.array(parts).reshape(-1, 2)
                    poly_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [pts.astype(np.int32)], 1)
                    if poly_mask.sum() >= 15:
                        nuclei_masks.append(poly_mask.astype(bool))

        # Sample up to 5 nuclei per image for prompt training
        if len(nuclei_masks) == 0:
            # Fallback dummy mask if image has no nuclei
            target_mask = np.zeros((256, 256), dtype=np.float32)
            box = np.array([0, 0, 10, 10], dtype=np.float32)
            pt = np.array([5, 5], dtype=np.float32)
        else:
            # Pick a random nucleus from the annotations
            nuc = random.choice(nuclei_masks)
            y_idxs, x_idxs = np.where(nuc)
            x1, y1, x2, y2 = x_idxs.min(), y_idxs.min(), x_idxs.max(), y_idxs.max()
            cx, cy = float(np.mean(x_idxs)), float(np.mean(y_idxs))
            
            # Apply training jitter (±0..8 pixels) to simulate doctor circling
            jitter_x = random.randint(-4, 4)
            jitter_y = random.randint(-4, 4)
            pad = random.randint(2, 8)
            bx1 = max(0, x1 - pad + jitter_x)
            by1 = max(0, y1 - pad + jitter_y)
            bx2 = min(w, x2 + pad + jitter_x)
            by2 = min(h, y2 + pad + jitter_y)
            box = np.array([bx1, by1, bx2, by2], dtype=np.float32)
            pt = np.array([cx, cy], dtype=np.float32)
            
            # Resize target mask to SAM mask decoder resolution (256x256)
            nuc_256 = cv2.resize(nuc.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            target_mask = nuc_256.astype(np.float32)

        # Standard SAM normalization
        img_1024 = cv2.resize(img, (1024, 1024))
        img_tensor = torch.from_numpy(img_1024).permute(2, 0, 1).float()
        mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        img_norm = (img_tensor - mean) / std

        return {
            "image": img_norm,
            "target_mask": torch.from_numpy(target_mask).unsqueeze(0),
            "box": torch.from_numpy(box),
            "point": torch.from_numpy(pt),
        }


# ─────────────────────────────────────────────────────────────
# 2. Combined Dice & Focal / BCE Loss
# ─────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target):
        bce_loss = self.bce(pred_logits, target)
        
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * target).sum(dim=(-2, -1))
        union = pred_probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 0.5 * bce_loss + 0.5 * dice_loss.mean()


def decode_masks(sam, image_embeddings, boxes, points, point_labels):
    """
    Decodes masks for a batch. Meta AI's mask_decoder expects a single image embedding
    per prompt set internally, so we decode per-sample in the batch to avoid shape mismatch.
    """
    b_size = image_embeddings.shape[0]
    masks_list = []
    for i in range(b_size):
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(points[i:i+1], point_labels[i:i+1]),
            boxes=boxes[i:i+1],
            masks=None
        )
        low_res, _ = sam.mask_decoder(
            image_embeddings=image_embeddings[i:i+1],
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks_list.append(low_res)
    return torch.cat(masks_list, dim=0)


# ─────────────────────────────────────────────────────────────
# 3. Main Training Routine
# ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load base MedSAM
    print(f"Loading base MedSAM weights: {args.checkpoint} ...")
    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    sam.to(device)

    # Freeze Image Encoder (ViT Backbone) to train quickly on 8GB GPU
    print("Freezing ViT Image Encoder to save GPU memory & focus on Nuclei Mask Decoder...")
    for param in sam.image_encoder.parameters():
        param.requires_grad = False

    # Train Prompt Encoder & Mask Decoder
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = True
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    train_params = [p for p in sam.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in train_params):,}")

    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    criterion = DiceBCELoss()

    train_dataset = NucleiDataset(args.data_dir, split="train")
    val_dataset = NucleiDataset(args.data_dir, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_val_dice = 0.0
    backup_path = args.checkpoint + ".orig_backup"
    if not os.path.exists(backup_path) and os.path.exists(args.checkpoint):
        import shutil
        shutil.copy2(args.checkpoint, backup_path)
        print(f"Created automatic backup of original weights: {backup_path}")

    print("\nStarting MedSAM Nuclei Fine-Tuning...\n" + "=" * 55)

    for epoch in range(1, args.epochs + 1):
        sam.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            targets = batch["target_mask"].to(device)
            boxes = batch["box"].to(device)
            points = batch["point"].to(device).unsqueeze(1)
            point_labels = torch.ones((imgs.shape[0], 1), dtype=torch.int, device=device)

            optimizer.zero_grad()

            with torch.no_grad():
                image_embeddings = sam.image_encoder(imgs)

            low_res_masks = decode_masks(sam, image_embeddings, boxes, points, point_labels)

            loss = criterion(low_res_masks, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step + 1:02d}/{len(train_loader):02d}] - Batch Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation step
        sam.eval()
        val_dice_list = []
        print(f"  Evaluating on test split...", flush=True)
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                targets = batch["target_mask"].to(device)
                boxes = batch["box"].to(device)
                points = batch["point"].to(device).unsqueeze(1)
                point_labels = torch.ones((imgs.shape[0], 1), dtype=torch.int, device=device)

                image_embeddings = sam.image_encoder(imgs)
                low_res_masks = decode_masks(sam, image_embeddings, boxes, points, point_labels)

                preds = (torch.sigmoid(low_res_masks) > 0.5).float()
                inter = (preds * targets).sum(dim=(-2, -1))
                union = preds.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
                dice = (2.0 * inter) / (union + 1e-5)
                val_dice_list.extend(dice.cpu().numpy().flatten())

        avg_val_dice = float(np.mean(val_dice_list)) if val_dice_list else 0.0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice * 100:.2f}%")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_model_path = os.path.join(os.path.dirname(args.checkpoint), "medsam_vit_b_best.pth")
            torch.save(sam.state_dict(), best_model_path)
            print(f"  ★ New best model saved! (Dice: {best_val_dice * 100:.2f}%) -> {best_model_path}")

    print("\nTraining Complete!")
    print(f"Best Validation Dice: {best_val_dice * 100:.2f}%")
    print(f"Saved Checkpoint: medsam_vit_b_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MedSAM on NuSeC & MiDeSeC")
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\electromist\Desktop\Ankara University Datasets")
    parser.add_argument("--checkpoint", type=str, default=r"c:\Users\electromist\Desktop\herlev_web_app\medsam_vit_b.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(args)
