#!/usr/bin/env python3
"""
Download mushroom images from iNaturalist for species not yet in the dataset.
Filters: research-grade, located in Sweden, with photos.

Usage:
    python scripts/download_inaturalist.py
"""

import requests
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"

SPECIES = [
    {"id": "AG.AU", "scientific": "Agaricus augustus", "target": 30},
    {"id": "LA.VO", "scientific": "Lactarius volemus", "target": 30},
    {"id": "LA.HE", "scientific": "Lactarius helvus", "target": 30},
    {"id": "RA.PA", "scientific": "Ramaria pallida", "target": 30},
]

SWEDEN_PLACE_ID = 7599


def download_species(species):
    folder = IMAGES_DIR / species["id"]
    folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {species['scientific']} -> {folder}")
    print(f"{'='*60}")

    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_name": species["scientific"],
        "quality_grade": "research",
        "photos": "true",
        "place_id": SWEDEN_PLACE_ID,
        "per_page": 200,
        "order": "desc",
        "order_by": "created_at",
    }

    try:
        resp = requests.get(url, params=params, timeout=60)
        data = resp.json()
    except Exception as e:
        print(f"  ERROR: API request failed: {e}")
        return 0

    results = data.get("results", [])
    print(f"  Found {len(results)} research-grade observations in Sweden")

    if len(results) == 0:
        print(f"  WARNING: No observations found for {species['scientific']} in Sweden")
        return 0

    downloaded = 0
    skipped = 0
    seen_urls = set()

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
            if not img_url or img_url in seen_urls:
                continue
            seen_urls.add(img_url)

            # Request large/original size
            img_url = img_url.replace("/square.jpg", "/large.jpg")
            img_url = img_url.replace("/thumb.jpg", "/large.jpg")
            img_url = img_url.replace("/small.jpg", "/large.jpg")
            img_url = img_url.replace("/medium.jpg", "/large.jpg")

            ext = ".jpg"
            if ".jpeg" in img_url.lower():
                ext = ".jpeg"
            elif ".png" in img_url.lower():
                ext = ".png"

            filename = f"{species['scientific'].replace(' ', '_')}_{downloaded + 1}{ext}"
            filepath = folder / filename

            if filepath.exists():
                downloaded += 1
                continue

            try:
                img_resp = requests.get(img_url, timeout=30)
                if img_resp.status_code != 200:
                    skipped += 1
                    continue

                content_type = img_resp.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    skipped += 1
                    continue

                content_length = len(img_resp.content)
                if content_length < 5000:
                    skipped += 1
                    continue

                with open(filepath, "wb") as f:
                    f.write(img_resp.content)

                downloaded += 1
                print(f"  [{downloaded:2d}/{species['target']:2d}] {filename} ({content_length // 1024}KB)")
                time.sleep(0.3)

            except Exception as e:
                skipped += 1
                print(f"  ERROR: {e}")
                continue

    print(f"  Done: {downloaded} downloaded, {skipped} skipped")
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
