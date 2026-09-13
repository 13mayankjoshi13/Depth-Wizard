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
# ──────────────────────────────────────────────────────────────────────────────
# Advanced Satellite Loss Functions
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


class SAMLoss(nn.Module):
    """
    Differentiable Spectral Angle Mapper (SAM) loss.
    Measures the angular spectral distortion across bands at each pixel:
        SAM(u, v) = arccos( (u · v) / (||u|| ||v|| + eps) )
    Guarantees physical reflectance curves are preserved across spectral bands.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W) in [0, 1]
        dot = torch.sum(pred * target, dim=1)
        norm_pred = torch.norm(pred, p=2, dim=1)
        norm_target = torch.norm(target, p=2, dim=1)
        cos_sim = dot / (norm_pred * norm_target + self.eps)
        cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)
        sam_rad = torch.acos(cos_sim)
        return torch.mean(sam_rad)


class EdgeLoss(nn.Module):
    """
    Sobel spatial gradient loss to penalize blurry borders on linear infrastructure:
    roads, runways, coastlines, agricultural boundaries, and building footprints.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x = sobel_x.to(device)
        self.sobel_y = sobel_y.to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        kernel_x = self.sobel_x.repeat(c, 1, 1, 1)
        kernel_y = self.sobel_y.repeat(c, 1, 1, 1)

        grad_pred_x = F.conv2d(pred, kernel_x, padding=1, groups=c)
        grad_pred_y = F.conv2d(pred, kernel_y, padding=1, groups=c)
        grad_target_x = F.conv2d(target, kernel_x, padding=1, groups=c)
        grad_target_y = F.conv2d(target, kernel_y, padding=1, groups=c)

        loss_x = F.l1_loss(grad_pred_x, grad_target_x)
        loss_y = F.l1_loss(grad_pred_y, grad_target_y)
        return loss_x + loss_y


class FrequencyLoss(nn.Module):
    """
    2D Fast Fourier Transform (FFT) loss:
    Recovers high-frequency micro-textures and prevents over-smoothed outputs.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


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
# Patch crop with Dihedral D4 Symmetry Augmentation
# ──────────────────────────────────────────────────────────────────────────────

class CropDataset(torch.utils.data.Dataset):
    """
    Wraps SyntheticPairDataset and returns random crops with full D4 dihedral
    rotations and reflection symmetry (overhead satellite imagery is orientation-invariant).
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

        # Use RGB only for standard 3-channel RRDBNet
        if self.rgb_only and lr.shape[0] >= 3:
            lr = lr[:3]
            hr = hr[:3]

        c, h, w = lr.shape
        if h < self.crop_size or w < self.crop_size:
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

        # Full D4 Dihedral symmetry augmentations:
        # 1. Random 90-degree rotations (0°, 90°, 180°, 270°)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            lr_crop = torch.rot90(lr_crop, k, [1, 2])
            hr_crop = torch.rot90(hr_crop, k, [1, 2])

        # 2. Random horizontal flip
        if torch.rand(1) > 0.5:
            lr_crop = torch.flip(lr_crop, [2])
            hr_crop = torch.flip(hr_crop, [2])

        # 3. Random vertical flip
        if torch.rand(1) > 0.5:
            lr_crop = torch.flip(lr_crop, [1])
            hr_crop = torch.flip(hr_crop, [1])

        return lr_crop, hr_crop


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    pairs_dir: Path | str,
    output_dir: Path | str,
    model_key: str = "x4plus",
    pretrained_checkpoint: Optional[Path | str] = None,
    resume_checkpoint: Optional[Path | str] = None,
    epochs: int = 50,
    batch_size: int = 4,
    crop_size: int = 128,
    lr: float = 1e-4,
    lambda_perceptual: float = 0.1,
    lambda_sam: float = 0.05,
    lambda_edge: float = 0.05,
    lambda_freq: float = 0.02,
    save_every: int = 10,
    num_workers: int = 2,
    use_amp: bool = True,
    time_budget_hours: Optional[float] = None,
) -> dict:
    """
    Fine-tune the Real-ESRGAN generator with deep multi-loss optimization:
    L1 + Perceptual (VGG) + Spectral (SAM) + Spatial Gradient (Edge) + FFT Frequency.
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
            "No GPU detected — training on CPU will be slow. "
            "For full 50-100 epoch deep training, use Google Colab with free T4 GPU (notebooks/colab_finetune.ipynb)."
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
    ckpt_to_load = resume_checkpoint or pretrained_checkpoint
    generator = load_generator_for_training(
        model_key=model_key,
        checkpoint_path=ckpt_to_load,
        device=device,
    )
    generator.train()

    # ── Multi-loss suite ──────────────────────────────────────────────────────
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss(device) if lambda_perceptual > 0 else None
    sam_loss = SAMLoss().to(device) if lambda_sam > 0 else None
    edge_loss = EdgeLoss(device) if lambda_edge > 0 else None
    freq_loss = FrequencyLoss().to(device) if lambda_freq > 0 else None

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == "cuda") else None

    # ── Training state ────────────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "epoch_time": [],
    }
    best_val_psnr = -float("inf")
    best_ckpt_path = output_dir / "model_finetuned_best.pth"
    t_training_start = time.time()   # wall clock for --time-budget-hours

    logger.info("=" * 70)
    logger.info("🚀 Deep Super-Resolution Training Started (%d epochs, batch_size=%d)", epochs, batch_size)
    logger.info("Losses: L1 + Perceptual(%.2f) + SAM(%.2f) + Edge(%.2f) + Freq(%.2f)",
                lambda_perceptual, lambda_sam, lambda_edge, lambda_freq)
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        generator.train()
        train_losses = []

        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    sr_batch = generator(lr_batch)
                    loss = l1_loss(sr_batch, hr_batch)
                    if perceptual_loss:
                        loss = loss + lambda_perceptual * perceptual_loss(sr_batch, hr_batch)
                    if sam_loss:
                        loss = loss + lambda_sam * sam_loss(sr_batch, hr_batch)
                    if edge_loss:
                        loss = loss + lambda_edge * edge_loss(sr_batch, hr_batch)
                    if freq_loss:
                        loss = loss + lambda_freq * freq_loss(sr_batch, hr_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                sr_batch = generator(lr_batch)
                loss = l1_loss(sr_batch, hr_batch)
                if perceptual_loss:
                    loss = loss + lambda_perceptual * perceptual_loss(sr_batch, hr_batch)
                if sam_loss:
                    loss = loss + lambda_sam * sam_loss(sr_batch, hr_batch)
                if edge_loss:
                    loss = loss + lambda_edge * edge_loss(sr_batch, hr_batch)
                if freq_loss:
                    loss = loss + lambda_freq * freq_loss(sr_batch, hr_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(loss.item())

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(
                    "  Epoch %02d/%02d | Batch %03d/%03d | Loss=%.4f",
                    epoch, epochs, batch_idx + 1, len(train_loader), loss.item(),
                )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        generator.eval()
        val_losses = []
        val_psnrs = []
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                sr_batch = generator(lr_batch)

                v_loss = l1_loss(sr_batch, hr_batch)
                val_losses.append(v_loss.item())

                # Quick batch PSNR calculation
                mse = F.mse_loss(sr_batch, hr_batch)
                batch_psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8)).item()
                val_psnrs.append(batch_psnr)

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        mean_psnr = float(np.mean(val_psnrs)) if val_psnrs else 0.0
        epoch_time = time.time() - t_start

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        history["val_psnr"].append(round(mean_psnr, 2))
        history["epoch_time"].append(round(epoch_time, 2))

        logger.info(
            "Epoch %02d/%02d | Train: %.4f | Val: %.4f | Val PSNR: %.2f dB | LR: %.6f | %.1fs",
            epoch, epochs, mean_train, mean_val, mean_psnr,
            optimizer.param_groups[0]["lr"], epoch_time,
        )

        # Save best model by validation PSNR
        if mean_psnr > best_val_psnr:
            best_val_psnr = mean_psnr
            save_generator_checkpoint(
                generator, best_ckpt_path, epoch=epoch,
                extra_info={"val_loss": mean_val, "val_psnr": mean_psnr},
            )
            logger.info("  ✓ Best checkpoint updated! (Val PSNR=%.2f dB)", mean_psnr)

        # Periodic checkpoint
        if epoch % save_every == 0:
            periodic_path = output_dir / f"model_finetuned_epoch{epoch:03d}.pth"
            save_generator_checkpoint(generator, periodic_path, epoch=epoch)

        # ── Time-budget guard (demo-day safety net) ──────────────────────────
        if time_budget_hours is not None:
            elapsed_hours = (time.time() - t_training_start) / 3600.0
            if elapsed_hours >= time_budget_hours:
                logger.warning(
                    "Time budget of %.2fh reached after epoch %d/%d — stopping early.",
                    time_budget_hours, epoch, epochs,
                )
                break

    # Final checkpoint
    final_path = output_dir / "model_finetuned_final.pth"
    save_generator_checkpoint(
        generator, final_path, epoch=epochs,
        extra_info={"history": history, "best_val_psnr": best_val_psnr},
    )

    # Save training history JSON
    hist_path = output_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Deep training completed. History saved → %s", hist_path)

    return {"best_checkpoint": str(best_ckpt_path), "history": history}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Deep Super-Resolution Fine-Tuning (SIH26142)")
    parser.add_argument("--pairs-dir", default="data/synthetic_pairs", help="Directory with lr/ and hr/ pairs")
    parser.add_argument("--output-dir", default="src/checkpoints", help="Where to save model checkpoints")
    parser.add_argument("--pretrained", default=None, help="Override pretrained checkpoint path")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--model-key", default="x4plus", choices=["x4plus", "x4plus_anime"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-perceptual", type=float, default=0.1)
    parser.add_argument("--lambda-sam", type=float, default=0.05, help="Spectral Angle Mapper loss weight")
    parser.add_argument("--lambda-edge", type=float, default=0.05, help="Sobel edge gradient loss weight")
    parser.add_argument("--lambda-freq", type=float, default=0.02, help="FFT frequency loss weight")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--time-budget-hours", type=float, default=None,
        help="Stop training gracefully after this many wall-clock hours (saves checkpoint). "
             "Use before demo day to avoid a mid-run cutoff.",
    )
    args = parser.parse_args()

    result = train(
        pairs_dir=args.pairs_dir,
        output_dir=args.output_dir,
        model_key=args.model_key,
        pretrained_checkpoint=args.pretrained,
        resume_checkpoint=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        lr=args.lr,
        lambda_perceptual=args.lambda_perceptual,
        lambda_sam=args.lambda_sam,
        lambda_edge=args.lambda_edge,
        lambda_freq=args.lambda_freq,
        save_every=args.save_every,
        num_workers=args.num_workers,
        time_budget_hours=args.time_budget_hours,
    )
    print(f"\nDeep training finished. Best checkpoint saved at: {result['best_checkpoint']}")
