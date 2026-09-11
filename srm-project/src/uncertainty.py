"""
uncertainty.py
───────────────
Uncertainty quantification for the SR model output.

Method: Test-Time Augmentation (TTA) Ensemble
─────────────────────────────────────────────
We apply 8 geometric augmentations (identity + 3 rotations + 4 flips),
run SR on each, undo the augmentation, then measure per-pixel variance across
the N outputs.

  uncertainty_map[h,w] = var_across_augmentations(sr_outputs[h,w])

This gives a principled estimate of model confidence:
  - Low variance  → model is consistent → high confidence
  - High variance → model disagrees across augmentations → low confidence

The variance heatmap is meaningful because satellite texture patterns are
approximately rotationally consistent, so inconsistency across augmentations
indicates the model is "guessing" fine detail.

Alternative (MC Dropout):
  If the model has Dropout layers that can be activated at inference time,
  N stochastic forward passes can be used instead. A fallback implementation
  is provided but TTA is the default because RRDBNet has no dropout by default.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _augment_chw(img: np.ndarray, aug_id: int) -> np.ndarray:
    """Apply augmentation aug_id ∈ [0,7] to (C, H, W) float32 array."""
    _, h, w = img.shape
    t = img.transpose(1, 2, 0)   # (H, W, C) for cv2

    if aug_id == 0:
        out = t
    elif aug_id == 1:                            # rot90
        out = np.rot90(t, 1)
    elif aug_id == 2:                            # rot180
        out = np.rot90(t, 2)
    elif aug_id == 3:                            # rot270
        out = np.rot90(t, 3)
    elif aug_id == 4:                            # flip-LR
        out = np.fliplr(t)
    elif aug_id == 5:                            # flip-UD
        out = np.flipud(t)
    elif aug_id == 6:                            # rot90 + flip-LR
        out = np.fliplr(np.rot90(t, 1))
    else:                                        # rot90 + flip-UD
        out = np.flipud(np.rot90(t, 1))

    return out.transpose(2, 0, 1)   # back to (C, H, W)


def _deaugment_chw(img: np.ndarray, aug_id: int) -> np.ndarray:
    """Reverse augmentation aug_id from (C, H, W) float32 array."""
    _, h, w = img.shape
    t = img.transpose(1, 2, 0)

    if aug_id == 0:
        out = t
    elif aug_id == 1:
        out = np.rot90(t, -1)
    elif aug_id == 2:
        out = np.rot90(t, -2)
    elif aug_id == 3:
        out = np.rot90(t, -3)
    elif aug_id == 4:
        out = np.fliplr(t)
    elif aug_id == 5:
        out = np.flipud(t)
    elif aug_id == 6:
        out = np.fliplr(np.rot90(t, -1))
    else:
        out = np.flipud(np.rot90(t, -1))

    return out.transpose(2, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# TTA Ensemble
# ──────────────────────────────────────────────────────────────────────────────

def tta_ensemble(
    lr_patch: np.ndarray,
    enhance_fn: Callable[[np.ndarray], np.ndarray],
    n_augmentations: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run TTA ensemble and return (mean_sr, uncertainty_map).

    Parameters
    ----------
    lr_patch     : float32 (C, H, W) in [0, 1] — the low-resolution input
    enhance_fn   : function that takes a (C, H, W) float32 array and returns
                   a (C, H*scale, W*scale) float32 array.
                   You can wrap model.enhance_multiband() as this function.
    n_augmentations: number of augmentations to use (max 8)

    Returns
    -------
    mean_sr       : float32 (C, H_sr, W_sr) — mean SR output across augmentations
    uncertainty   : float32 (H_sr, W_sr)    — per-pixel std dev across augmentations
    """
    n_aug = min(n_augmentations, 8)
    outputs: List[np.ndarray] = []

    for aug_id in range(n_aug):
        aug_lr = _augment_chw(lr_patch, aug_id)
        try:
            sr_aug = enhance_fn(aug_lr)           # (C, H_sr, W_sr)
        except Exception as exc:
            logger.warning("TTA augmentation %d failed: %s — skipping", aug_id, exc)
            continue
        sr_canonical = _deaugment_chw(sr_aug, aug_id)
        outputs.append(sr_canonical)

    if not outputs:
        raise RuntimeError("All TTA augmentations failed")

    stack = np.stack(outputs, axis=0)             # (N, C, H_sr, W_sr)
    mean_sr = np.mean(stack, axis=0)              # (C, H_sr, W_sr)

    # Per-pixel std across bands and augmentations
    std_map = np.std(stack, axis=0)               # (C, H_sr, W_sr)
    uncertainty = np.mean(std_map, axis=0)        # (H_sr, W_sr) — mean across bands

    return mean_sr.astype(np.float32), uncertainty.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty visualisation
# ──────────────────────────────────────────────────────────────────────────────

def uncertainty_to_heatmap(
    uncertainty: np.ndarray,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """
    Convert a float32 (H, W) uncertainty map to a uint8 (H, W, 3) RGB heatmap.

    Normalises to [0, 255] using percentile stretch for visual clarity.
    Higher intensity = higher uncertainty (less confident SR detail).
    """
    unc = uncertainty.astype(np.float32)
    lo, hi = np.percentile(unc, 2), np.percentile(unc, 98)
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    unc_norm = np.clip((unc - lo) / (hi - lo), 0, 1)
    unc_u8 = (unc_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(unc_u8, colormap)   # (H, W, 3) BGR
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # RGB


def save_uncertainty_heatmap(
    uncertainty: np.ndarray,
    out_path: str | Path,
    sr_image_rgb: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> None:
    """
    Save an uncertainty heatmap (optionally blended with the SR image).

    Parameters
    ----------
    uncertainty    : float32 (H, W) uncertainty map
    out_path       : output PNG path
    sr_image_rgb   : optional uint8 (H, W, 3) SR image for overlay blend
    alpha          : blend ratio (0 = pure heatmap, 1 = pure SR image)
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = uncertainty_to_heatmap(uncertainty)

    if sr_image_rgb is not None:
        # Resize heatmap to match SR image if needed
        if heatmap.shape[:2] != sr_image_rgb.shape[:2]:
            heatmap = cv2.resize(heatmap, (sr_image_rgb.shape[1], sr_image_rgb.shape[0]))
        blended = cv2.addWeighted(sr_image_rgb, alpha, heatmap, 1 - alpha, 0)
        canvas = np.hstack([sr_image_rgb, blended, heatmap])
    else:
        canvas = heatmap

    # Save with matplotlib (adds colorbar legend)
    fig, axes = plt.subplots(1, 1 if sr_image_rgb is None else 3, figsize=(12 if sr_image_rgb is not None else 5, 4))
    if sr_image_rgb is None:
        axes = [axes]

    if sr_image_rgb is not None:
        axes[0].imshow(sr_image_rgb)
        axes[0].set_title("SR Output", fontsize=10)
        axes[0].axis("off")
        axes[1].imshow(blended)
        axes[1].set_title("SR + Uncertainty Overlay", fontsize=10)
        axes[1].axis("off")
        im = axes[2].imshow(uncertainty, cmap="viridis")
        axes[2].set_title("Uncertainty (per-pixel std)", fontsize=10)
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Std Dev")
    else:
        im = axes[0].imshow(uncertainty, cmap="viridis")
        axes[0].set_title("Uncertainty Map", fontsize=10)
        axes[0].axis("off")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Std Dev")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Uncertainty heatmap saved → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# MC Dropout fallback (optional)
# ──────────────────────────────────────────────────────────────────────────────

def mc_dropout_uncertainty(
    lr_patch: np.ndarray,
    generator_fn: Callable[[np.ndarray], np.ndarray],
    n_passes: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout uncertainty estimate (fallback / alternative to TTA).

    Requires the model to have Dropout layers that activate in train() mode.
    RRDBNet by default has no Dropout — use TTA instead unless you've added
    Dropout layers via monkeypatching.

    Parameters
    ----------
    generator_fn : function called with lr_patch (C,H,W) → sr (C,H_sr,W_sr)
                   Must be called with model.train() active.

    Returns
    -------
    mean_sr, uncertainty  (same semantics as tta_ensemble)
    """
    outputs = []
    for _ in range(n_passes):
        sr = generator_fn(lr_patch)
        outputs.append(sr)

    stack = np.stack(outputs, axis=0)     # (N, C, H_sr, W_sr)
    mean_sr = np.mean(stack, axis=0)
    uncertainty = np.mean(np.std(stack, axis=0), axis=0)
    return mean_sr.astype(np.float32), uncertainty.astype(np.float32)
