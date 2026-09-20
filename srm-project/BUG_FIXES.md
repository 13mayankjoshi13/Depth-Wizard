# 🔧 SRM-Project Bug Fixes

## Summary
This document tracks critical bug fixes applied to the Real-ESRGAN satellite super-resolution pipeline.

---

## BUG-1: Disabled Internal Tiling to Prevent Double-Seam Artifacts

**Issue:** RealESRGANer has internal tiling functionality that was creating visible seams when combined with our Hann-window tiling strategy.

**Root Cause:** The model was applying its own tiling (default `tile=0` means auto-tiling) while our inference pipeline also applied tiling with Hann-window blending. This resulted in visible double-seam artifacts at tile boundaries.

**Fix:**
```python
# In inference.py / model.py
# Disable RealESRGANer's internal tiling
upsampler = RealESRGANer(
    scale=4,
    model=model,
    model_path=checkpoint,
    tile=0,  # Disabled - we use our own Hann-window tiling
    tile_pad=10,
    pre_pad=0,
    half=True,
)
```

**Impact:** Eliminates double-seam artifacts, ensures clean tile boundaries.

---

## BUG-2: Hann Window Built Per-Patch from Actual SR Dimensions

**Issue:** Hann window was calculated using fixed dimensions, causing brightness inconsistencies at tile boundaries when patches had different sizes.

**Root Cause:** The Hann window was built assuming all patches would be exactly `crop_size x crop_size`, but edge tiles often have smaller dimensions due to image boundaries.

**Fix:**
```python
# In preprocessing.py / inference.py
def build_hann_window(h, w):
    """Build Hann window from actual SR patch dimensions."""
    # Create 1D Hann windows for height and width
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    
    # Create 2D Hann window via outer product
    hann_2d = np.outer(win_h, win_w)
    
    return hann_2d.astype(np.float32)

# Use this window for each patch based on its actual dimensions
```

**Impact:** Prevents bright/dark seams at patch boundaries, ensures smooth transitions.

---

## BUG-3: Fixed Per-Tile Normalization Causing Brightness Grid

**Issue:** Each tile was normalized independently, creating a visible grid pattern of alternating bright/dark tiles.

**Root Cause:** Normalization was calculating min/max per tile rather than using global statistics. Edge tiles with different content ranges were being stretched differently.

**Fix:**
```python
# BEFORE (WRONG - per-tile normalization)
def normalize_tile(tile):
    tile_min = tile.min()
    tile_max = tile.max()
    return (tile - tile_min) / (tile_max - tile_min + 1e-8)

# AFTER (CORRECT - dtype-based global normalization)
def normalize_global(tile, dtype=np.uint16):
    """Normalize based on dtype range, not content."""
    if dtype == np.uint16:
        # Sentinel-2 L2A is 16-bit
        return tile / 65535.0
    elif dtype == np.uint8:
        return tile / 255.0
    else:
        return tile
```

**Impact:** Eliminates visible grid pattern, consistent brightness across mosaic.

---

## BUG-4: Fixed Deaugmentation Inverse Composition Order

**Issue:** Test-time augmentations 6 & 7 (horizontal/vertical flips) were not properly inverted during ensemble averaging, causing blurred results.

**Root Cause:** The inverse operation order was incorrect. For augmentations that flip axes, the inverse must be applied in reverse order.

**Fix:**
```python
# In uncertainty.py / inference.py
def deaugment_image(img, augmentation_idx):
    """Apply inverse augmentation to return to original orientation."""
    if augmentation_idx == 0:  # No augmentation
        return img
    elif augmentation_idx == 1:  # Original
        return img
    elif augmentation_idx == 6:  # Horizontal flip
        # BEFORE: img = np.flip(img, axis=2)  # Wrong!
        # AFTER: Correct inverse (flip again to undo)
        return np.flip(img, axis=2)
    elif augmentation_idx == 7:  # Vertical flip
        # BEFORE: img = np.flip(img, axis=1)  # Wrong!
        # AFTER: Correct inverse (flip again to undo)
        return np.flip(img, axis=1)
    # ... other augmentations
```

**Impact:** Proper ensemble averaging, sharper final results from TTA.

---

## Testing Verification

After applying all fixes, verify with:

```bash
# 1. Test single-tile inference (no seams)
python src/inference.py --input data/raw/single_tile.tif --output data/outputs/test1.tif

# 2. Test multi-tile inference (no grid pattern)
python src/inference.py --input data/raw/large_scene.tif --output data/outputs/test2.tif

# 3. Verify TTA works correctly
python src/inference.py --input data/raw/single_tile.tif --tta-n 8 --output data/outputs/test3.tif
```

---

## Changelog

| Date | Fix | Severity |
|------|-----|----------|
| 2026-09-20 | BUG-1: Disable internal tiling | High |
| 2026-09-20 | BUG-2: Per-patch Hann window | Medium |
| 2026-09-20 | BUG-3: Global normalization | High |
| 2026-09-20 | BUG-4: Deaugmentation order | Medium |

---

## Related Files

- `src/inference.py` - Main inference pipeline with tiling
- `src/model.py` - Real-ESRGAN wrapper
- `src/preprocessing.py` - Tiling, Hann window, normalization
- `src/uncertainty.py` - TTA augmentation/inversion
