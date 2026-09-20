"""
download_sentinel2_tiles.py
---------------------------
Download Sentinel-2 tiles from Copernicus Data Space using their API.

Requirements:
  pip install requests tqdm

Usage:
  python download_sentinel2_tiles.py --locations "Delhi,Mumbai,Bangalore" --max-cloud 10
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install requests tqdm")
    sys.exit(1)


# Approximate coordinates for Indian cities (you can customize)
LOCATIONS = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707),
    "hyderabad": (17.3850, 78.4867),
    "kolkata": (22.5726, 88.3639),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
}


def search_sentinel2_tiles(
    lat: float,
    lon: float,
    max_cloud_cover: int = 10,
    days_back: int = 90,
) -> list:
    """
    Search for Sentinel-2 L2A tiles near given coordinates.

    Returns list of tile metadata (including download URLs).
    """
    # Copernicus Data Space Ecosystem API
    base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    # Bounding box (approx 20km × 20km around point)
    bbox_delta = 0.1
    bbox = f"{lon - bbox_delta},{lat - bbox_delta},{lon + bbox_delta},{lat + bbox_delta}"

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Query parameters
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' and "
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover}) and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({bbox.replace(',', ' ')}))')"
        ),
        "$orderby": "ContentDate/Start desc",
        "$top": 10,
    }

    print(f"Searching tiles for ({lat}, {lon}) with <{max_cloud_cover}% clouds...")

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        tiles = data.get("value", [])
        print(f"  Found {len(tiles)} tiles")

        return tiles

    except requests.RequestException as e:
        print(f"  Error searching: {e}")
        return []


def download_tile_preview(tile: dict, output_dir: Path) -> bool:
    """
    Download True Color Image (TCI) preview of a Sentinel-2 tile.

    This is much smaller than full product (~50 MB vs ~1 GB).
    """
    tile_id = tile.get("Name", "unknown")
    tile_date = tile.get("ContentDate", {}).get("Start", "unknown")
    cloud_cover = None

    # Extract cloud cover from attributes
    for attr in tile.get("Attributes", []):
        if attr.get("Name") == "cloudCover":
            cloud_cover = attr.get("Value")
            break

    print(f"\nTile: {tile_id}")
    print(f"  Date: {tile_date}")
    print(f"  Cloud cover: {cloud_cover}%")

    # Download link (you'll need authentication for full access)
    # For now, we'll guide manual download
    download_url = f"https://dataspace.copernicus.eu/browser/?zoom=10&lat={tile.get('GeoFootprint', {}).get('coordinates', [[[[0,0]]]])[0][0][1]}&lng={tile.get('GeoFootprint', {}).get('coordinates', [[[[0,0]]]])[0][0][0]}&datasetId=S2_L2A&productId={tile_id}"

    print(f"  Manual download: {download_url}")
    print(f"  → Open in browser, click 'Visualize', then download True Color preview")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Search and guide download of Sentinel-2 tiles"
    )
    parser.add_argument(
        "--locations",
        default="delhi,mumbai,bangalore",
        help="Comma-separated city names (delhi,mumbai,bangalore)",
    )
    parser.add_argument(
        "--max-cloud",
        type=int,
        default=10,
        help="Maximum cloud cover percentage (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/raw",
        help="Output directory for tiles (default: ../data/raw)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Search tiles from last N days (default: 90)",
    )

    args = parser.parse_args()

    # Parse locations
    location_names = [loc.strip().lower() for loc in args.locations.split(",")]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Sentinel-2 Tile Downloader")
    print("=" * 60)
    print(f"Searching locations: {', '.join(location_names)}")
    print(f"Max cloud cover: {args.max_cloud}%")
    print(f"Output directory: {output_dir.absolute()}")
    print()

    all_tiles = []

    for loc_name in location_names:
        if loc_name not in LOCATIONS:
            print(f"Warning: Unknown location '{loc_name}', skipping")
            continue

        lat, lon = LOCATIONS[loc_name]
        print(f"\n{'=' * 60}")
        print(f"Location: {loc_name.title()} ({lat}, {lon})")
        print('=' * 60)

        tiles = search_sentinel2_tiles(
            lat=lat,
            lon=lon,
            max_cloud_cover=args.max_cloud,
            days_back=args.days_back,
        )

        if tiles:
            # Show first 3 tiles for this location
            for tile in tiles[:3]:
                download_tile_preview(tile, output_dir)

        all_tiles.extend(tiles)

    print("\n" + "=" * 60)
    print(f"Total tiles found: {len(all_tiles)}")
    print("=" * 60)

    print("\n📝 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Visit: https://dataspace.copernicus.eu/browser/")
    print("2. Sign in (free account)")
    print("3. Search for the locations above")
    print("4. Filter: Sentinel-2 L2A, Cloud cover <10%")
    print("5. Download 'True Color' preview for each tile")
    print(f"6. Save to: {output_dir.absolute()}")

    print("\n✅ After downloading 5-10 tiles, run:")
    print(f"   python src/pair_generation.py --raw-dir {args.output_dir}")


if __name__ == "__main__":
    main()
