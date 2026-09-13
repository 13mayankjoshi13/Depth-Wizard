"""
test_blending_fixes.py
───────────────────────
Regression tests for the four tile-grid artifact bug fixes.

  Bug 1 — inference.py: RealESRGANer tile=0 (internal tiling disabled)
  Bug 2 — inference.py: Hann window built per-patch from actual SR dimensions
  Bug 3 — preprocessing.py: tile_image uses fixed max_val for float dtype
  Bug 4 — uncertainty.py: _deaugment_chw is the exact inverse of _augment_chw

Key property of np.hanning:
  hanning(N)[0] == hanning(N)[-1] == 0.0  (exact boundary zeros by definition)
  Therefore the *outermost* pixel row/column of any Hann-blended canvas always
  has zero accumulated weight and is treated as a hard boundary.  All tests
  that validate the weight map check the INTERIOR region (excluding a 1-px
  border) to avoid triggering this mathematical boundary condition.

Run:
  pytest tests/test_blending_fixes.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _random_chw(c, h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((c, h, w)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Bug 4 — TTA de-augmentation round-trip identity
# ─────────────────────────────────────────────────────────────────────────────

class TestTTARoundTrip:
    """
    For every aug_id in [0, 7], _augment_chw followed by _deaugment_chw must
    return a pixel-for-pixel identical array (round-trip identity).
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.uncertainty import _augment_chw, _deaugment_chw
        self._augment = _augment_chw
        self._deaugment = _deaugment_chw

    @pytest.mark.parametrize("aug_id", range(8))
    def test_round_trip_square(self, aug_id):
        """Square patches: augment → deaugment must recover original."""
        img = _random_chw(3, 64, 64)
        recovered = self._deaugment(self._augment(img, aug_id), aug_id)
        np.testing.assert_allclose(
            recovered, img, atol=1e-6,
            err_msg=f"aug_id={aug_id}: deaugmented result differs from original",
        )

    @pytest.mark.parametrize("aug_id", [0, 2, 4, 5])
    def test_round_trip_nonsquare(self, aug_id):
        """Shape-preserving augmentations on a non-square patch (H≠W)."""
        img = _random_chw(3, 48, 64)
        recovered = self._deaugment(self._augment(img, aug_id), aug_id)
        np.testing.assert_allclose(
            recovered, img, atol=1e-6,
            err_msg=f"aug_id={aug_id} non-square: deaugmented result differs",
        )

    def test_all_aug_ids_change_image(self):
        """Every non-identity augmentation must actually change the array."""
        img = _random_chw(3, 64, 64, seed=99)
        for aug_id in range(1, 8):
            aug = self._augment(img, aug_id)
            assert not np.allclose(aug, img), f"aug_id={aug_id} produced identity (no change)"


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2 — Hann weight map correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestHannWeightMap:
    """
    Simulate the Hann-blend loop from inference.py on a small scene and verify:
      1. The INTERIOR weight map has no zeros (boundary zeros are expected and OK)
      2. For a constant-1 input the normalised INTERIOR is exactly 1.0
      3. A boundary patch gets a full 0→peak→0 taper (not just a near-zero edge)
    """

    @staticmethod
    def _make_hann_window(h, w):
        """Mirror of _make_hann_window in inference.py."""
        return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

    def _simulate_blend(self, img_h, img_w, patch_size, overlap, scale=1):
        """
        Run the Hann blend loop on a constant-1 synthetic SR output.
        Returns (weight_map, normalised_canvas) — same logic as inference.py.
        """
        canvas = np.zeros((img_h * scale, img_w * scale), dtype=np.float64)
        weight = np.zeros_like(canvas)
        stride = patch_size - overlap

        rows = list(range(0, img_h - patch_size + 1, stride))
        if not rows or rows[-1] + patch_size < img_h:
            rows.append(max(0, img_h - patch_size))
        cols = list(range(0, img_w - patch_size + 1, stride))
        if not cols or cols[-1] + patch_size < img_w:
            cols.append(max(0, img_w - patch_size))

        for r in rows:
            for c in cols:
                sr_r, sr_c = r * scale, c * scale
                sr_ph = min(patch_size * scale, img_h * scale - sr_r)
                sr_pw = min(patch_size * scale, img_w * scale - sr_c)
                hann = self._make_hann_window(sr_ph, sr_pw)
                # constant-1 SR patch: canvas += hann, weight += hann
                canvas[sr_r:sr_r + sr_ph, sr_c:sr_c + sr_pw] += hann
                weight[sr_r:sr_r + sr_ph, sr_c:sr_c + sr_pw] += hann

        wm = np.where(weight > 0, weight, 1.0)
        return weight, canvas / wm

    def test_interior_weight_map_no_zeros(self):
        """
        The INTERIOR of the weight map (excluding the 1-px Hann-zero border)
        must have strictly positive weight everywhere.
        The 1-px border is always zero by the mathematical definition of np.hanning
        (hanning[0] = hanning[-1] = 0) and is expected / handled by inference.py.
        """
        weight, _ = self._simulate_blend(img_h=128, img_w=128, patch_size=64, overlap=16)
        interior = weight[1:-1, 1:-1]
        assert interior.min() > 0, (
            f"Interior weight_map has zero entries (min={interior.min():.8f}). "
            "Non-boundary pixels must receive positive Hann weight."
        )

    def test_interior_normalised_output_is_one(self):
        """
        For a constant-1 SR input, the weighted-average interior output must
        equal 1.0 everywhere — the Hann weights cancel in numerator/denominator.
        """
        _, normed = self._simulate_blend(img_h=128, img_w=128, patch_size=64, overlap=16)
        interior = normed[1:-1, 1:-1]
        np.testing.assert_allclose(
            interior, np.ones_like(interior), atol=1e-5,
            err_msg="Normalised interior deviates from 1.0 for constant-1 input",
        )

    def test_boundary_patch_hann_is_full_taper(self):
        """
        A boundary-sized Hann window (e.g. 48×48) must span [0→peak→0] across
        its entire extent — not just start from the leading near-zero edge.
        """
        h, w = 48, 48
        hann = self._make_hann_window(h, w)
        peak_r, peak_c = np.unravel_index(np.argmax(hann), hann.shape)
        assert abs(peak_r - h // 2) <= 2, f"Hann peak row={peak_r} not near centre h//2={h//2}"
        assert abs(peak_c - w // 2) <= 2, f"Hann peak col={peak_c} not near centre w//2={w//2}"
        # Exact corners are 0 by definition; second row/col should be non-trivially small
        assert hann[1, 1] < 0.01, f"Hann [1,1]={hann[1,1]:.4f} is too large (should be near zero)"
        # Centre should be the clear maximum
        assert hann[h // 2, w // 2] > 0.9, f"Hann centre={hann[h//2, w//2]:.4f} is not near 1"

    def test_interior_uniformity_with_50pct_overlap(self):
        """
        With 50% overlap (stride=32), pixels that are at least one full stride
        (32 px) from any scene edge are always covered by >=2 overlapping patches.
        Within that "safe interior" the weight map peak/min ratio must be < 20,
        confirming smooth blending without extreme hot-spots.

        Pixels closer than one stride to the scene edge receive contribution from
        only the single boundary patch's Hann taper and legitimately have much
        lower weight — we exclude them from this check.
        """
        patch_size, overlap = 64, 32
        stride = patch_size - overlap   # 32
        img_h = img_w = 256
        weight, _ = self._simulate_blend(img_h, img_w, patch_size, overlap)
        # Only check pixels at least one stride away from every scene edge.
        # These pixels are guaranteed to be covered by >=2 patches.
        safe = weight[stride:-stride, stride:-stride]
        assert safe.size > 0, "Safe interior region is empty; increase test image size"
        ratio = safe.max() / safe.min()
        assert ratio < 20.0, (
            f"Safe-interior weight map peak/min={ratio:.2f} is too high — "
            "blending is uneven within multi-patch covered region"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Bug 3 — preprocessing tile normalisation consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestTileNormalisation:
    """
    tile_image() must normalise float-dtype patches with max_val=1.0 always,
    NOT with data.max() per tile, so adjacent tiles with different local maxima
    remain on a globally consistent scale.
    """

    def test_float_dtype_uses_constant_maxval(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        rng = np.random.default_rng(0)
        h, w = 128, 256
        data = np.zeros((1, h, w), dtype=np.float32)
        # Left half: values in (0, 0.2]; right half: values in (0, 0.8]
        data[0, :, :w // 2] = rng.random((h, w // 2)).astype(np.float32) * 0.2
        data[0, :, w // 2:] = rng.random((h, w // 2)).astype(np.float32) * 0.8

        tif = tmp_path / "test_float.tif"
        transform = from_bounds(0, 0, 1, 1, width=w, height=h)
        with rasterio.open(
            tif, "w", driver="GTiff", height=h, width=w,
            count=1, dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)

        from src.preprocessing import tile_image
        patches = list(tile_image(str(tif), patch_size=128, overlap=0, min_valid_fraction=0.0))
        assert len(patches) == 2, f"Expected 2 patches, got {len(patches)}"

        left_patch = patches[0][0]
        right_patch = patches[1][0]

        # With per-tile normalisation the left patch max would approach 1.0 (WRONG).
        # With fixed max_val=1.0 the left patch max must stay ≤ 0.25.
        assert left_patch.max() <= 0.25, (
            f"Left patch max={left_patch.max():.4f}; expected ≤0.25. "
            f"Per-tile independent normalisation may still be active."
        )
        # Right patch remains brighter — global scale is preserved
        assert right_patch.mean() > left_patch.mean() * 2, (
            "Right patch is not consistently brighter than left — global scale not preserved"
        )

