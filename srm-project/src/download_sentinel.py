"""
download_sentinel.py
─────────────────────
Sentinel-2 data acquisition helper.

Since you don't yet have a Copernicus Data Space account, this script:
  1. Prints step-by-step instructions for creating a free account and
     manually downloading Sentinel-2 L2A tiles.
  2. After tiles are placed in data/raw/, validates them automatically.

Run this script at any time to check tile status:
  python src/download_sentinel.py --validate-only

Once you have an account and want automated downloads, uncomment the
sentinelsat section at the bottom.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║        SENTINEL-2 DATA ACQUISITION — MANUAL DOWNLOAD INSTRUCTIONS          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  STEP 1 — Create a FREE Copernicus Data Space account                       ║
║  ─────────────────────────────────────────────────────                       ║
║  1. Go to: https://dataspace.copernicus.eu/                                  ║
║  2. Click "Register" (top right)                                             ║
║  3. Fill in the form (name, email, institution) — completely free            ║
║  4. Verify your email and log in                                             ║
║                                                                              ║
║  STEP 2 — Open the Browser / EO Browser                                     ║
║  ─────────────────────────────────────────────────                           ║
║  1. Go to: https://browser.dataspace.copernicus.eu/                         ║
║  2. Log in with your new credentials                                         ║
║                                                                              ║
║  STEP 3 — Search for Sentinel-2 L2A tiles                                   ║
║  ──────────────────────────────────────────                                  ║
║  Recommended areas (varied terrain):                                         ║
║    A) Urban:       Delhi NCR region (India, ~28.6°N 77.2°E)                 ║
║    B) Agricultural: Punjab plains (~30.5°N 75.0°E)                          ║
║    C) Coastal:     Mumbai/Konkan coast (~18.9°N 72.8°E)                     ║
║                                                                              ║
║  Search settings:                                                            ║
║    • Data source:  Sentinel-2 L2A (atmospherically corrected)               ║
║    • Cloud cover:  < 20%                                                     ║
║    • Date range:   2023-10-01 to 2024-03-31 (dry season, less cloud)        ║
║    • Resolution:   10m bands                                                 ║
║                                                                              ║
║  STEP 4 — Download as GeoTIFF                                               ║
║  ──────────────────────────────                                              ║
║  Option A (Recommended — Analytical GeoTIFF):                               ║
║    1. Select a scene → click "Visualise"                                     ║
║    2. In the panel, click "Analytical" tab                                   ║
║    3. Select bands: B02, B03, B04, B08 (Blue, Green, Red, NIR)              ║
║       (or download a 4-band stack)                                           ║
║    4. Click "Download image" → choose GeoTIFF format                        ║
║    5. Resolution: 10m, Format: GeoTIFF                                       ║
║                                                                              ║
║  Option B (Full .SAFE product via API):                                      ║
║    1. Go to: https://scihub.copernicus.eu/ (legacy) or                      ║
║       https://dataspace.copernicus.eu/odata/v1/                             ║
║    2. Search for "MSIL2A" products for your AOI                              ║
║    3. Download and unzip the .SAFE folder                                    ║
║    4. Stack bands using QGIS or our provided script (see below)              ║
║                                                                              ║
║  STEP 5 — Place files in data/raw/                                          ║
║  ──────────────────────────────────                                          ║
║  • Save each GeoTIFF as: data/raw/<area>_<date>.tif                         ║
║    Examples:                                                                 ║
║      data/raw/delhi_20240115.tif                                             ║
║      data/raw/punjab_20240210.tif                                            ║
║      data/raw/mumbai_20231120.tif                                            ║
║  • Target: 5-10 tiles covering varied terrain                               ║
║  • Minimum size: 512×512 pixels (ideally 2000×2000 or larger)               ║
║                                                                              ║
║  STEP 6 — Validate tiles                                                    ║
║  ───────────────────────                                                     ║
║  Run: python src/download_sentinel.py --validate-only                       ║
║  Expected output: ✓ with CRS, dimensions, band count for each tile          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ALTERNATIVE: EarthExplorer (USGS) — also free                              ║
║  https://earthexplorer.usgs.gov/                                             ║
║  Search for "Sentinel-2" under "Sentinel" data category                     ║
║  Login required (free USGS account)                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

SAFE_STACK_SCRIPT = """
# ─── Stack Sentinel-2 bands from a .SAFE folder using GDAL ────────────────
# Run this in the folder containing the .SAFE directory
# Adjust the tile name (e.g. T43RGQ) to match your download

SAFE_DIR="S2A_MSIL2A_*.SAFE"
GRANULE_DIR="${SAFE_DIR}/GRANULE/*/IMG_DATA/R10m"
OUTPUT="data/raw/stacked_10m.tif"

# Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) at 10m
gdal_merge.py \\
  -separate \\
  -o ${OUTPUT} \\
  ${GRANULE_DIR}/*B02_10m.jp2 \\
  ${GRANULE_DIR}/*B03_10m.jp2 \\
  ${GRANULE_DIR}/*B04_10m.jp2 \\
  ${GRANULE_DIR}/*B08_10m.jp2

echo "Stacked GeoTIFF saved to ${OUTPUT}"
"""


def print_instructions() -> None:
    print(INSTRUCTIONS)
    print("\n─── If you downloaded a .SAFE folder, use this GDAL command to stack bands: ───")
    print(SAFE_STACK_SCRIPT)


def validate_tiles(raw_dir: Path | str = Path("data/raw")) -> list:
    """Validate all GeoTIFFs in raw_dir and return results."""
    from src.preprocessing import validate_all_tiles

    raw_dir = Path(raw_dir)
    results = validate_all_tiles(raw_dir)

    if not results:
        print(f"\n⚠  No valid GeoTIFF files found in {raw_dir}")
        print("   Please follow the download instructions above and try again.")
        return []

    print(f"\n{'─'*70}")
    print(f"✓ Found {len(results)} valid tile(s) in {raw_dir}:")
    for r in results:
        p = Path(r["path"])
        print(f"  {p.name:40s} | {r['width']:5d}×{r['height']:5d} | {r['bands']:2d} bands | {r['crs']}")
    print(f"{'─'*70}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Optional: Automated download via sentinelsat (requires Copernicus account)
# Uncomment AFTER creating an account and installing sentinelsat:
#   pip install sentinelsat
# ──────────────────────────────────────────────────────────────────────────────

# def automated_download(
#     username: str,
#     password: str,
#     raw_dir: Path = Path("data/raw"),
#     aois: list | None = None,
# ) -> None:
#     """
#     Download Sentinel-2 L2A tiles automatically via sentinelsat.
#     aois: list of (name, WKT footprint) tuples
#     """
#     from sentinelsat import SentinelAPI
#     from datetime import date
#
#     api = SentinelAPI(username, password, "https://scihub.copernicus.eu/dhus")
#     raw_dir.mkdir(parents=True, exist_ok=True)
#
#     if aois is None:
#         # Default: Delhi NCR bounding box
#         aois = [
#             ("delhi", "POLYGON((76.8 28.4, 77.5 28.4, 77.5 28.9, 76.8 28.9, 76.8 28.4))"),
#         ]
#
#     for name, wkt in aois:
#         products = api.query(
#             area=wkt,
#             date=("20231001", "20240301"),
#             platformname="Sentinel-2",
#             producttype="S2MSI2A",
#             cloudcoverpercentage=(0, 20),
#         )
#         if not products:
#             print(f"No products found for {name}")
#             continue
#         # Download the most recent one
#         product_id = list(products.keys())[-1]
#         api.download(product_id, directory_path=raw_dir)
#         print(f"Downloaded {name} → {raw_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Sentinel-2 download instructions + tile validation")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate tiles already in data/raw/ (skip instructions)")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw tile directory")
    args = parser.parse_args()

    if not args.validate_only:
        print_instructions()

    results = validate_tiles(args.raw_dir)
    if results:
        print("\n✅ Tiles are ready. Next step: python src/pair_generation.py")
    else:
        print("\n📋 Follow the instructions above, place tiles in data/raw/, then re-run this script.")
