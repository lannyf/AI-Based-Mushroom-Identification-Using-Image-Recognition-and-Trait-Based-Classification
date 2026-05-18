#!/usr/bin/env python3
"""Download mushroom images from iNaturalist for all 16 training species.

Search priority:
  1. Sweden (place_id=7599)
  2. Denmark (place_id=8051)
  3. Norway (place_id=7016)
  4. Finland (place_id=7020)

Only research-grade observations with photos are considered.
Images are saved under data/raw/images/<Swedish name (Latin)>/.

Usage:
    python scripts/download_nordic_images.py [--target 40]
"""

from __future__ import annotations

import argparse
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"

# All 16 species with Swedish common names, scientific names, and species_ids
SPECIES = [
    {"id": "AG.AU", "swedish": "Kungschampinjon", "scientific": "Agaricus augustus", "target": 40},
    {"id": "AM.MU", "swedish": "Flugsvamp", "scientific": "Amanita muscaria", "target": 40},
    {"id": "AM.VI", "swedish": "Änglsvamp", "scientific": "Amanita virosa", "target": 40},
    {"id": "BO.BA", "swedish": "Brunsopp", "scientific": "Boletus badius", "target": 40},
    {"id": "BO.ED", "swedish": "Karljohan", "scientific": "Boletus edulis", "target": 40},
    {"id": "CA.CI", "swedish": "Kantarell", "scientific": "Cantharellus cibarius", "target": 40},
    {"id": "CO.CO", "swedish": "Spindelskivling", "scientific": "Coprinellus comatus", "target": 40},
    {"id": "CR.CO", "swedish": "Svart trumpetsvamp", "scientific": "Craterellus cornucopioides", "target": 40},
    {"id": "FO.BE", "swedish": "Björkticka", "scientific": "Fomitopsis betulina", "target": 40},
    {"id": "HY.PS", "swedish": "Falsk kantarell", "scientific": "Hygrophoropsis aurantiaca", "target": 40},
    {"id": "LA.HE", "swedish": "Lakritsriska", "scientific": "Lactarius helvus", "target": 40},
    {"id": "LA.VO", "swedish": "Mandelriska", "scientific": "Lactarius volemus", "target": 40},
    {"id": "LY.PE", "swedish": "Rökslöjpa", "scientific": "Lycoperdon perlatum", "target": 40},
    {"id": "RA.BO", "swedish": "Druvfingersvamp", "scientific": "Ramaria botrytis", "target": 40},
    {"id": "RA.PA", "swedish": "Blek fingersvamp", "scientific": "Ramaria pallida", "target": 40},
    {"id": "SP.CR", "swedish": "Blomkålssvamp", "scientific": "Sparassis crispa", "target": 40},
]

NORDIC_PLACES = [
    ("Sweden", 7599),
    ("Denmark", 8051),
    ("Norway", 7016),
    ("Finland", 7020),
]

MIN_FILE_SIZE_KB = 15
MIN_DIMENSION = 200
SIZES = ["original", "large", "medium"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def try_download(url_template: str, seen_urls: set):
    for size in SIZES:
        url = url_template.replace("/square.jpg", f"/{size}.jpg")
        url = url.replace("/thumb.jpg", f"/{size}.jpg")
        url = url.replace("/small.jpg", f"/{size}.jpg")
        url = url.replace("/medium.jpg", f"/{size}.jpg")
        url = url.replace("/large.jpg", f"/{size}.jpg")
        url = url.replace("/original.jpg", f"/{size}.jpg")

        if url in seen_urls:
            continue
        seen_urls.add(url)

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                continue
            if len(resp.content) < MIN_FILE_SIZE_KB * 1024:
                continue
            return resp.content, url
        except Exception:
            continue
    return None, None


def verify_image(img_bytes: bytes) -> bool:
    try:
        img = Image.open(BytesIO(img_bytes))
        w, h = img.size
        return w >= MIN_DIMENSION and h >= MIN_DIMENSION
    except Exception:
        return False


def download_species(species: dict, target: int):
    folder_name = f"{species['swedish']} ({species['scientific']})"
    folder = IMAGES_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing = len([p for p in folder.iterdir() if p.suffix in IMAGE_EXTS])
    if existing >= target:
        print(f"\n{'='*60}")
        print(f"SKIP: {folder_name} already has {existing}/{target} images")
        print(f"{'='*60}")
        return existing

    print(f"\n{'='*60}")
    print(f"Downloading: {species['scientific']} -> {folder_name}")
    print(f"Existing: {existing}, Target: {target}")
    print(f"{'='*60}")

    downloaded = existing
    seen_urls: set[str] = set()
    total_skipped = 0

    for country_name, place_id in NORDIC_PLACES:
        if downloaded >= target:
            break

        url = "https://api.inaturalist.org/v1/observations"
        params = {
            "taxon_name": species["scientific"],
            "quality_grade": "research",
            "photos": "true",
            "place_id": place_id,
            "per_page": 200,
            "order": "desc",
            "order_by": "created_at",
        }

        try:
            resp = requests.get(url, params=params, timeout=60)
            data = resp.json()
        except Exception as e:
            print(f"  ERROR: API request failed for {country_name}: {e}")
            continue

        results = data.get("results", [])
        print(f"  {country_name}: {len(results)} observations found")

        if len(results) == 0:
            continue

        skipped = 0
        for obs in results:
            if downloaded >= target:
                break

            photos = obs.get("photos", [])
            if not photos:
                continue

            for photo in photos:
                if downloaded >= target:
                    break

                img_url = photo.get("url", "")
                if not img_url:
                    continue

                img_bytes, used_url = try_download(img_url, seen_urls)
                if img_bytes is None:
                    skipped += 1
                    continue

                if not verify_image(img_bytes):
                    skipped += 1
                    continue

                ext = ".jpg"
                if img_bytes[:4] == b"\x89PNG":
                    ext = ".png"
                elif img_bytes[:2] == b"\xff\xd8":
                    ext = ".jpg"

                filename = f"{species['scientific'].replace(' ', '_')}_{downloaded + 1}{ext}"
                filepath = folder / filename

                # Avoid overwriting by checking existence
                if filepath.exists():
                    downloaded += 1
                    continue

                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                downloaded += 1
                print(f"  [{downloaded:2d}/{target:2d}] {country_name}: {filename} ({len(img_bytes)//1024}KB)")
                time.sleep(0.3)

        total_skipped += skipped
        print(f"  {country_name}: {downloaded} total so far")

    print(f"  Done: {downloaded} total, {total_skipped} skipped")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download Nordic mushroom images from iNaturalist")
    parser.add_argument("--target", type=int, default=40, help="Target images per species")
    parser.add_argument("--species", type=str, default=None, help="Comma-separated species_ids to download (e.g. CA.CI,HY.PS)")
    args = parser.parse_args()

    to_download = SPECIES
    if args.species:
        ids = {s.strip() for s in args.species.split(",")}
        to_download = [s for s in SPECIES if s["id"] in ids]

    total = 0
    for sp in to_download:
        total += download_species(sp, args.target)

    print(f"\n{'='*60}")
    print(f"Grand total: {total} images across all species")
    print(f"{'='*60}")

    # Final summary
    print("\n--- Final counts ---")
    for sp in SPECIES:
        folder_name = f"{sp['swedish']} ({sp['scientific']})"
        folder = IMAGES_DIR / folder_name
        n = len([p for p in folder.iterdir() if p.suffix in IMAGE_EXTS]) if folder.exists() else 0
        status = "✓" if n >= args.target else f"({n}/{args.target})"
        print(f"  {folder_name}: {n} {status}")


if __name__ == "__main__":
    main()
