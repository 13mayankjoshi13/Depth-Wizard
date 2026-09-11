"""
test_geo.py
────────────
Tests that geo-referencing (CRS + affine transform + bounding box) is
preserved after the preprocessing and SR pipeline.

All tests use a synthetic in-memory GeoTIFF — no external data required.
"""

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import validate_tile, extract_rgb_preview


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_geotiff(tmp_path) -> Path:
    """Create a synthetic 256×256 4-band GeoTIFF with known CRS and transform."""
    tif_path = tmp_path / "test_tile.tif"
    data = (np.random.randint(1000, 10000, (4, 256, 256), dtype=np.uint16))

    transform = Affine(10.0, 0.0, 500000.0,   # 10m pixel size, origin at (500000, 3000000)
                       0.0, -10.0, 3000000.0)
    crs = CRS.from_epsg(32643)  # WGS84 / UTM zone 43N (covers India)

    with rasterio.open(
        tif_path, "w",
        driver="GTiff",
        height=256, width=256,
        count=4,
        dtype=np.uint16,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    return tif_path


@pytest.fixture
def sr_geotiff(tmp_path) -> Path:
    """Create a synthetic 1024×1024 (4× SR) GeoTIFF with proportionally adjusted transform."""
    tif_path = tmp_path / "test_tile_sr.tif"
    data = np.random.randint(1000, 10000, (4, 1024, 1024), dtype=np.uint16)

    # SR transform: pixel size = 10/4 = 2.5m
    transform = Affine(2.5, 0.0, 500000.0,
                       0.0, -2.5, 3000000.0)
    crs = CRS.from_epsg(32643)

    with rasterio.open(
        tif_path, "w",
        driver="GTiff",
        height=1024, width=1024,
        count=4,
        dtype=np.uint16,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    return tif_path


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateTile:
    def test_valid_tile_returns_metadata(self, synthetic_geotiff):
        meta = validate_tile(synthetic_geotiff)
        assert meta["width"] == 256
        assert meta["height"] == 256
        assert meta["bands"] == 4
        assert "32643" in meta["crs"]   # EPSG:32643

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_tile(tmp_path / "nonexistent.tif")

    def test_crs_is_not_none(self, synthetic_geotiff):
        meta = validate_tile(synthetic_geotiff)
        assert meta["crs"] is not None
        assert meta["crs"] != "None"

    def test_transform_is_affine(self, synthetic_geotiff):
        meta = validate_tile(synthetic_geotiff)
        t = meta["transform"]
        assert hasattr(t, "a"), "transform should be an Affine object"
        assert abs(t.a - 10.0) < 1e-6, "Expected 10m pixel width"
        assert abs(t.e - (-10.0)) < 1e-6, "Expected -10m pixel height"


class TestGeoPreservation:
    """
    The core requirement: after 4× SR upscaling, the top-left corner
    coordinate (origin) must match exactly, and pixel size must be 1/scale
    of the original.
    """

    def test_origin_preserved(self, synthetic_geotiff, sr_geotiff):
        with rasterio.open(synthetic_geotiff) as src:
            orig_transform = src.transform
        with rasterio.open(sr_geotiff) as src:
            sr_transform = src.transform

        # Origin must match
        assert abs(orig_transform.c - sr_transform.c) < 1e-3, \
            f"X origin mismatch: {orig_transform.c} vs {sr_transform.c}"
        assert abs(orig_transform.f - sr_transform.f) < 1e-3, \
            f"Y origin mismatch: {orig_transform.f} vs {sr_transform.f}"

    def test_pixel_size_scaled(self, synthetic_geotiff, sr_geotiff):
        scale = 4
        with rasterio.open(synthetic_geotiff) as src:
            orig_px = abs(src.transform.a)
        with rasterio.open(sr_geotiff) as src:
            sr_px = abs(src.transform.a)

        expected_sr_px = orig_px / scale
        assert abs(sr_px - expected_sr_px) < 1e-6, \
            f"SR pixel size should be {expected_sr_px}m, got {sr_px}m"

    def test_bounding_box_preserved(self, synthetic_geotiff, sr_geotiff):
        """Bounding box of SR output must match original (within floating-point)."""
        with rasterio.open(synthetic_geotiff) as src:
            orig_bounds = src.bounds
        with rasterio.open(sr_geotiff) as src:
            sr_bounds = src.bounds

        tol = 5.0  # 5m tolerance (< 1 original pixel)
        assert abs(orig_bounds.left - sr_bounds.left) < tol,   "Left bound mismatch"
        assert abs(orig_bounds.bottom - sr_bounds.bottom) < tol, "Bottom bound mismatch"
        assert abs(orig_bounds.right - sr_bounds.right) < tol,  "Right bound mismatch"
        assert abs(orig_bounds.top - sr_bounds.top) < tol,      "Top bound mismatch"

    def test_crs_preserved(self, synthetic_geotiff, sr_geotiff):
        with rasterio.open(synthetic_geotiff) as src:
            orig_crs = src.crs
        with rasterio.open(sr_geotiff) as src:
            sr_crs = src.crs
        assert orig_crs == sr_crs, f"CRS changed: {orig_crs} → {sr_crs}"


class TestExtractRGBPreview:
    def test_returns_uint8_rgb(self, synthetic_geotiff):
        rgb = extract_rgb_preview(synthetic_geotiff)
        assert rgb.dtype == np.uint8
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3

    def test_values_in_range(self, synthetic_geotiff):
        rgb = extract_rgb_preview(synthetic_geotiff)
        assert rgb.min() >= 0
        assert rgb.max() <= 255
