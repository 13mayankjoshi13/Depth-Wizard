"""
train_enhanced.py
─────────────────
Enhanced fine-tuning script with improved hyperparameters for SIH showcase.

Key improvements:
  - Larger crop sizes (192px instead of 128px)
  - Lower learning rate with warmup (5e-5 from 1e-4)
  - Better loss weights for satellite imagery
  - Learning rate warmup for stable training
  - Enhanced logging with real-time metrics
  - Checkpoint selection by PSNR + SSIM hybrid metric

Usage:
  python src/train_enhanced.py --pairs-dir data/synthetic_pairs --epochs 50
  python src/train_enhanced.py --pairs-dir data/synthetic_pairs --epochs 100 --batch-size 4
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
# SSIM calculation implemented manually to avoid scipy DLL issues
def compute_ssim_manual(img1, img2):
    """Compute SSIM manually without skimage to avoid scipy DLL issues."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1 * 255
    img2 = img2 * 255

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    return ssim

sys.modules["torchvision.transforms.functional_tensor"] = F_t

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Advanced Satellite Loss Functions (same as train.py but with minor tweaks)
# ──────────────────────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss on RGB features."""

    def __init__(self, device: torch.device):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).to(device).eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.device = device

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def _prep(x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] > 3:
                x = x[:, :3]
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x - mean) / std

        with torch.amp.autocast('cuda', enabled=False):
            feat_pred = self.feature_extractor(_prep(pred.float()))
            feat_target = self.feature_extractor(_prep(target.detach().float()))
            return F.l1_loss(feat_pred, feat_target)


class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss for spectral consistency."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            pred_f = pred.float()
            target_f = target.float()
            dot = torch.sum(pred_f * target_f, dim=1)
            norm_pred = torch.norm(pred_f, p=2, dim=1)
            norm_target = torch.norm(target_f, p=2, dim=1)
            cos_sim = dot / (norm_pred * norm_target + self.eps)
            cos_sim = torch.clamp(cos_sim, -0.9999, 0.9999)
            sam_rad = torch.acos(cos_sim)
            return torch.mean(sam_rad)


class EdgeLoss(nn.Module):
    """Sobel edge loss for sharp boundaries."""

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
    """FFT-based frequency loss for texture preservation."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
            target_fft = torch.fft.rfft2(target.float(), norm="ortho")
            return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def _collate_fn(batch):
    """Stack (lr, hr) pairs; skip None items."""
    batch = [b for b in batch if b is not None]
    lrs = torch.stack([b[0] for b in batch])
    hrs = torch.stack([b[1] for b in batch])
    return lrs, hrs


class CropDataset(torch.utils.data.Dataset):
    """Wraps SyntheticPairDataset with D4 augmentation and larger crops."""

    def __init__(self, base_dataset, crop_size: int = 192, scale: int = 4, rgb_only: bool = True):
        self.ds = base_dataset
        self.crop_size = crop_size
        self.scale = scale
        self.rgb_only = rgb_only

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        lr, hr = self.ds[idx]

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

        top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        lr_crop = lr[:, top:top + self.crop_size, left:left + self.crop_size]
        hr_crop = hr[:,
                     top * self.scale:(top + self.crop_size) * self.scale,
                     left * self.scale:(left + self.crop_size) * self.scale]

        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            lr_crop = torch.rot90(lr_crop, k, [1, 2])
            hr_crop = torch.rot90(hr_crop, k, [1, 2])

        if torch.rand(1) > 0.5:
            lr_crop = torch.flip(lr_crop, [2])
            hr_crop = torch.flip(hr_crop, [2])

        if torch.rand(1) > 0.5:
            lr_crop = torch.flip(lr_crop, [1])
            hr_crop = torch.flip(hr_crop, [1])

        return lr_crop, hr_crop


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images without requiring scipy/skimage."""
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    return float(compute_ssim_manual(img1, img2))


def train_enhanced(
    pairs_dir: Path | str,
    output_dir: Path | str,
    model_key: str = "x4plus",
    pretrained_checkpoint: Optional[Path | str] = None,
    resume_checkpoint: Optional[Path | str] = None,
    epochs: int = 50,
    batch_size: int = 4,
    crop_size: int = 192,
    lr: float = 5e-5,
    warmup_epochs: int = 5,
    lambda_perceptual: float = 0.15,
    lambda_sam: float = 0.03,
    lambda_edge: float = 0.08,
    lambda_freq: float = 0.03,
    save_every: int = 5,
    num_workers: int = 2,
    use_amp: bool = True,
    gradient_checkpointing: bool = True,
    time_budget_hours: Optional[float] = None,
) -> dict:
    """
    Enhanced fine-tuning with:
      - Warmup learning rate schedule
      - Larger crop sizes
      - Better loss balance for satellite imagery
      - SSIM metric tracking
      - Hybrid checkpoint selection (PSNR + SSIM)
    """
    from src.pair_generation import SyntheticPairDataset
    from src.model import load_generator_for_training, save_generator_checkpoint

    pairs_dir = Path(pairs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

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

    # Enable gradient checkpointing to reduce VRAM (critical for Colab T4)
    if gradient_checkpointing and hasattr(generator, 'enable_gradient_checkpointing'):
        generator.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing ENABLED — saves ~40-60%% VRAM")

    # Clear CUDA cache before allocating training tensors
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Loss functions ────────────────────────────────────────────────────────
    l1_loss = nn.L1Loss()
    perceptual_loss = PerceptualLoss(device) if lambda_perceptual > 0 else None
    sam_loss = SAMLoss().to(device) if lambda_sam > 0 else None
    edge_loss = EdgeLoss(device) if lambda_edge > 0 else None
    freq_loss = FrequencyLoss().to(device) if lambda_freq > 0 else None

    # ── Optimizer with warmup schedule ────────────────────────────────────────
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.99))

    # Warmup + Cosine annealing
    def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
        def warmup_cosine(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)

    scheduler = get_lr_scheduler(optimizer, warmup_epochs, epochs)
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == "cuda") else None

    # ── Training state ────────────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "epoch_time": [],
    }
    best_val_metric = -float("inf")
    best_ckpt_path = output_dir / "model_finetuned_best.pth"
    t_training_start = time.time()

    logger.info("=" * 80)
    logger.info("🚀 ENHANCED Fine-Tuning Started (%d epochs, crop_size=%d)", epochs, crop_size)
    logger.info("Losses: L1 + Perceptual(%.2f) + SAM(%.2f) + Edge(%.2f) + Freq(%.2f)",
                lambda_perceptual, lambda_sam, lambda_edge, lambda_freq)
    logger.info("LR Warmup: %d epochs, then cosine decay", warmup_epochs)
    logger.info("=" * 80)

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
                    "  Epoch %02d/%02d | Batch %03d/%03d | Loss=%.4f | LR=%.6f",
                    epoch, epochs, batch_idx + 1, len(train_loader), loss.item(),
                    optimizer.param_groups[0]["lr"],
                )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        generator.eval()
        val_losses = []
        val_psnrs = []
        val_ssims = []
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                sr_batch = generator(lr_batch)

                v_loss = l1_loss(sr_batch, hr_batch)
                val_losses.append(v_loss.item())

                mse = F.mse_loss(sr_batch, hr_batch)
                batch_psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8)).item()
                val_psnrs.append(batch_psnr)

                # SSIM calculation
                sr_np = sr_batch.cpu().numpy().clip(0, 1)
                hr_np = hr_batch.cpu().numpy()
                batch_ssim = np.mean([compute_ssim(sr_np[i], hr_np[i]) for i in range(len(sr_np))])
                val_ssims.append(batch_ssim)

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")
        mean_psnr = float(np.mean(val_psnrs)) if val_psnrs else 0.0
        mean_ssim = float(np.mean(val_ssims)) if val_ssims else 0.0
        epoch_time = time.time() - t_start

        # Hybrid metric: 0.6*PSNR + 0.4*SSIM (normalized to 0-1)
        hybrid_metric = 0.6 * (mean_psnr / 40.0) + 0.4 * mean_ssim

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        history["val_psnr"].append(round(mean_psnr, 2))
        history["val_ssim"].append(round(mean_ssim, 4))
        history["epoch_time"].append(round(epoch_time, 2))

        logger.info(
            "Epoch %02d/%02d | Train: %.4f | Val: %.4f | PSNR: %.2f dB | SSIM: %.4f | Hybrid: %.4f | %.1fs",
            epoch, epochs, mean_train, mean_val, mean_psnr, mean_ssim, hybrid_metric, epoch_time,
        )

        # Save best model by hybrid metric
        if hybrid_metric > best_val_metric:
            best_val_metric = hybrid_metric
            save_generator_checkpoint(
                generator, best_ckpt_path, epoch=epoch,
                extra_info={"val_loss": mean_val, "val_psnr": mean_psnr, "val_ssim": mean_ssim},
            )
            logger.info("  ✓ Best checkpoint updated! (PSNR=%.2f dB, SSIM=%.4f)", mean_psnr, mean_ssim)

        # Periodic checkpoint
        if epoch % save_every == 0:
            periodic_path = output_dir / f"model_finetuned_epoch{epoch:03d}.pth"
            save_generator_checkpoint(generator, periodic_path, epoch=epoch)

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
        extra_info={"history": history, "best_val_metric": best_val_metric},
    )

    # Save training history JSON
    hist_path = output_dir / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Enhanced training completed. History saved → %s", hist_path)

    return {"best_checkpoint": str(best_ckpt_path), "history": history}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Enhanced Fine-Tuning for SIH")
    parser.add_argument("--pairs-dir", default="data/synthetic_pairs", help="Directory with lr/ and hr/ pairs")
    parser.add_argument("--output-dir", default="src/checkpoints", help="Where to save checkpoints")
    parser.add_argument("--pretrained", default=None, help="Override pretrained checkpoint path")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--model-key", default="x4plus", choices=["x4plus", "x4plus_anime"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lambda-perceptual", type=float, default=0.15)
    parser.add_argument("--lambda-sam", type=float, default=0.03)
    parser.add_argument("--lambda-edge", type=float, default=0.08)
    parser.add_argument("--lambda-freq", type=float, default=0.03)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    args = parser.parse_args()

    result = train_enhanced(
        pairs_dir=args.pairs_dir,
        output_dir=args.output_dir,
        model_key=args.model_key,
        pretrained_checkpoint=args.pretrained,
        resume_checkpoint=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        lambda_perceptual=args.lambda_perceptual,
        lambda_sam=args.lambda_sam,
        lambda_edge=args.lambda_edge,
        lambda_freq=args.lambda_freq,
        save_every=args.save_every,
        num_workers=args.num_workers,
        time_budget_hours=args.time_budget_hours,
    )
    print(f"\nEnhanced training finished. Best checkpoint: {result['best_checkpoint']}")
