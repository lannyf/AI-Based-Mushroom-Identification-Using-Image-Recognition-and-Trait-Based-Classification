#!/usr/bin/env python3
"""
Download mushroom images from iNaturalist (v2).
Searches across Nordic countries: Sweden, Denmark, Norway, Finland.
Tries multiple photo sizes (original > large > medium) and verifies dimensions.

Usage:
    python scripts/download_inaturalist_v2.py
"""

import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"

SPECIES = [
    {"id": "AG.AU", "scientific": "Agaricus augustus", "target": 30},
    {"id": "LA.VO", "scientific": "Lactarius volemus", "target": 30},
    {"id": "LA.HE", "scientific": "Lactarius helvus", "target": 30},
    {"id": "RA.PA", "scientific": "Ramaria pallida", "target": 30},
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


def try_download(url_template, seen_urls):
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


def verify_image(img_bytes):
    try:
        img = Image.open(BytesIO(img_bytes))
        w, h = img.size
        return w >= MIN_DIMENSION and h >= MIN_DIMENSION
    except Exception:
        return False


def download_species(species):
    folder = IMAGES_DIR / species["id"]
    folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {species['scientific']} -> {folder}")
    print(f"{'='*60}")

    downloaded = len(list(folder.glob("*")))
    seen_urls = set()
    total_skipped = 0

    for country_name, place_id in NORDIC_PLACES:
        if downloaded >= species["target"]:
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
            if downloaded >= species["target"]:
                break

            photos = obs.get("photos", [])
            if not photos:
                continue

            for photo in photos:
                if downloaded >= species["target"]:
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

                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                downloaded += 1
                print(f"  [{downloaded:2d}/{species['target']:2d}] {country_name}: {filename} ({len(img_bytes)//1024}KB)")
                time.sleep(0.3)

        total_skipped += skipped
        print(f"  {country_name}: {downloaded} total so far")

    print(f"  Done: {downloaded} downloaded, {total_skipped} skipped")
    return downloaded


def main():
    total = 0
    for sp in SPECIES:
        total += download_species(sp)

    print(f"\n{'='*60}")
    print(f"Total downloaded: {total} images")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
