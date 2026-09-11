"""
train.py
─────────
Local GPU fine-tuning script for Real-ESRGAN on Sentinel-2 synthetic pairs.

Since you have a local NVIDIA GPU, this script runs locally instead of
requiring Colab. The Colab notebook (notebooks/colab_finetune.ipynb)
remains as a backup / documentation reference.

Training strategy:
  - Load pretrained RealESRGAN_x4plus generator (RRDB backbone)
  - Fine-tune with L1 (pixel) loss + VGG16 perceptual loss
  - No discriminator — generator-only training for stability
  - Learning rate: 1e-4 with cosine annealing decay
  - Save checkpoint every N epochs + best checkpoint (by val loss)

Usage:
  python src/train.py --pairs-dir data/synthetic_pairs --epochs 50
  python src/train.py --pairs-dir data/synthetic_pairs --epochs 100 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F_t

sys.modules["torchvision.transforms.functional_tensor"] = F_t

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Perceptual Loss (VGG16 features)
# ──────────────────────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss on RGB features."""

    def __init__(self, device: torch.device):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Use features up to relu3_3 (layers 0..15)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).to(device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.device = device

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : (B, C, H, W) float in [0,1]
        If C != 3, use first 3 channels or replicate grayscale to 3ch.
        """
        def _prep(x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] > 3:
                x = x[:, :3]
            # ImageNet normalisation
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x - mean) / std

        feat_pred = self.feature_extractor(_prep(pred))
        feat_target = self.feature_extractor(_prep(target.detach()))
        return F.l1_loss(feat_pred, feat_target)


# ──────────────────────────────────────────────────────────────────────────────
# Training collate (handles variable-band tensors)
# ──────────────────────────────────────────────────────────────────────────────

def _collate_fn(batch):
    """Stack (lr, hr) pairs; skip None items."""
    batch = [b for b in batch if b is not None]
    lrs = torch.stack([b[0] for b in batch])
    hrs = torch.stack([b[1] for b in batch])
    return lrs, hrs


# ──────────────────────────────────────────────────────────────────────────────
# Patch crop for training (HR is 4× larger than LR)
# ──────────────────────────────────────────────────────────────────────────────

class CropDataset(torch.utils.data.Dataset):
    """
    Wraps SyntheticPairDataset and returns random crops.
    LR crop: crop_size × crop_size
    HR crop: (crop_size * scale) × (crop_size * scale)
    Uses only RGB bands (first 3) if multi-band.
    """

    def __init__(self, base_dataset, crop_size: int = 128, scale: int = 4, rgb_only: bool = True):
        self.ds = base_dataset
        self.crop_size = crop_size
        self.scale = scale
        self.rgb_only = rgb_only

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        lr, hr = self.ds[idx]   # (C, H_lr, W_lr), (C, H_hr, W_hr)

        # Use RGB only for faster training (Real-ESRGAN is 3-ch)
        if self.rgb_only and lr.shape[0] >= 3:
            lr = lr[:3]
            hr = hr[:3]

        c, h, w = lr.shape
        if h < self.crop_size or w < self.crop_size:
            # Pad if too small
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            lr = F.pad(lr, (0, pad_w, 0, pad_h))
            hr = F.pad(hr, (0, pad_w * self.scale, 0, pad_h * self.scale))
            h, w = lr.shape[1], lr.shape[2]

        # Random crop
        top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        lr_crop = lr[:, top:top + self.crop_size, left:left + self.crop_size]
        hr_crop = hr[:,
                     top * self.scale:(top + self.crop_size) * self.scale,
                     left * self.scale:(left + self.crop_size) * self.scale]

        # Random horizontal flip
        if torch.rand(1) > 0.5:
            lr_crop = torch.flip(lr_crop, [2])
            hr_crop = torch.flip(hr_crop, [2])

        return lr_crop, hr_crop


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    pairs_dir: Path | str,
    output_dir: Path | str,
    model_key: str = "x4plus",
    pretrained_checkpoint: Optional[Path | str] = None,
    epochs: int = 50,
    batch_size: int = 4,
    crop_size: int = 128,
    lr: float = 1e-4,
    lambda_perceptual: float = 0.1,
    save_every: int = 10,
    num_workers: int = 2,
) -> dict:
    """
    Fine-tune the Real-ESRGAN generator on synthetic Sentinel-2 pairs.

    Returns dict with training history and checkpoint path.
    """
    from src.pair_generation import SyntheticPairDataset
    from src.model import load_generator_for_training, save_generator_checkpoint

    pairs_dir = Path(pairs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)
    if device.type == "cpu":
        logger.warning(
            "No GPU detected — training on CPU will be VERY slow. "
            "Consider using Colab (notebooks/colab_finetune.ipynb) or Kaggle."
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_base = SyntheticPairDataset(pairs_dir, split="train")
    val_base = SyntheticPairDataset(pairs_dir, split="val")
    train_ds = CropDataset(train_base, crop_size=crop_size, scale=4, rgb_only=True)
    val_ds = CropDataset(val_base, crop_size=crop_size, scale=4, rgb_only=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=_collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    generator = load_generator_for_training(
        model_key=model_key,
        checkpoint_path=pretrained_checkpoint,
        device=device,
    )
    generator.train()

    # ── Loss functions ────────────────────────────────────────────────────────
    l1_loss = nn.L1Loss()
    perceptual = PerceptualLoss(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── Training ──────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "model_finetuned_best.pth"

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        generator.train()
        train_losses = []

        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            sr_batch = generator(lr_batch)

            loss_l1 = l1_loss(sr_batch, hr_batch)
            loss_perceptual = perceptual(sr_batch, hr_batch) if lambda_perceptual > 0 else 0.0
            loss = loss_l1 + lambda_perceptual * loss_perceptual
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    "  Epoch %d/%d | Batch %d/%d | L1=%.4f | Perc=%.4f",
                    epoch, epochs, batch_idx + 1, len(train_loader),
                    loss_l1.item(),
                    loss_perceptual.item() if isinstance(loss_perceptual, torch.Tensor) else 0,
                )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        generator.eval()
        val_losses = []
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                sr_batch = generator(lr_batch)
                val_loss = l1_loss(sr_batch, hr_batch)
                val_losses.append(val_loss.item())

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        epoch_time = time.time() - t_start

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        history["epoch_time"].append(round(epoch_time, 2))

        logger.info(
            "Epoch %d/%d | Train: %.4f | Val: %.4f | LR: %.6f | %.1fs",
            epoch, epochs, mean_train, mean_val,
            optimizer.param_groups[0]["lr"], epoch_time,
        )

        # Save best
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            save_generator_checkpoint(generator, best_ckpt_path, epoch=epoch,
                                      extra_info={"val_loss": mean_val})
            logger.info("  ✓ Best checkpoint saved (val_loss=%.4f)", mean_val)

        # Periodic checkpoint
        if epoch % save_every == 0:
            periodic_path = output_dir / f"model_finetuned_epoch{epoch:03d}.pth"
            save_generator_checkpoint(generator, periodic_path, epoch=epoch)

    # Final checkpoint
    final_path = output_dir / "model_finetuned_final.pth"
    save_generator_checkpoint(generator, final_path, epoch=epochs,
                              extra_info={"history": history})

    # Save training history
    hist_path = output_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training complete. History → %s", hist_path)

    return {"best_checkpoint": str(best_ckpt_path), "history": history}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune Real-ESRGAN on Sentinel-2 pairs (local GPU)")
    parser.add_argument("--pairs-dir", default="data/synthetic_pairs", help="Directory with lr/ and hr/ subfolders")
    parser.add_argument("--output-dir", default="src/checkpoints", help="Where to save checkpoints")
    parser.add_argument("--pretrained", default=None, help="Override pretrained checkpoint path")
    parser.add_argument("--model-key", default="x4plus", choices=["x4plus", "x4plus_anime"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-perceptual", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    result = train(
        pairs_dir=args.pairs_dir,
        output_dir=args.output_dir,
        model_key=args.model_key,
        pretrained_checkpoint=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        lr=args.lr,
        lambda_perceptual=args.lambda_perceptual,
        save_every=args.save_every,
        num_workers=args.num_workers,
    )
    print(f"\nBest checkpoint: {result['best_checkpoint']}")
