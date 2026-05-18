#!/usr/bin/env python3
"""
Download Ramaria pallida images from multiple sources:
- iNaturalist (global, all 31 research-grade observations)
- GBIF (Global Biodiversity Information Facility)
- Mushroom Observer
- Svampe.databasen.org (Danish mushroom database)

Usage:
    python scripts/download_ramaria_pallida.py
"""

import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images" / "RA.PA"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 30
MIN_FILE_SIZE_KB = 15
MIN_DIMENSION = 200
SIZES = ["original", "large", "medium"]


def try_inaturalist_sizes(url_template):
    for size in SIZES:
        url = url_template.replace("/square.jpg", f"/{size}.jpg")
        url = url.replace("/thumb.jpg", f"/{size}.jpg")
        url = url.replace("/small.jpg", f"/{size}.jpg")
        url = url.replace("/medium.jpg", f"/{size}.jpg")
        url = url.replace("/large.jpg", f"/{size}.jpg")
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
                if len(resp.content) >= MIN_FILE_SIZE_KB * 1024:
                    return resp.content
        except Exception:
            continue
    return None


def verify_image(img_bytes):
    try:
        img = Image.open(BytesIO(img_bytes))
        w, h = img.size
        return w >= MIN_DIMENSION and h >= MIN_DIMENSION
    except Exception:
        return False


def save_image(img_bytes, folder, prefix, idx):
    ext = ".jpg"
    if img_bytes[:4] == b"\x89PNG":
        ext = ".png"
    elif img_bytes[:2] == b"\xff\xd8":
        ext = ".jpg"
    filename = f"{prefix}_{idx}{ext}"
    filepath = folder / filename
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    return filename


def download_inaturalist_global():
    print("Source 1: iNaturalist (global)")
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_name": "Ramaria pallida",
        "quality_grade": "research",
        "photos": "true",
        "per_page": 200,
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    results = data.get("results", [])
    print(f"  Found {len(results)} observations")
    return results


def download_gbif_images():
    print("Source 2: GBIF")
    all_urls = []
    for offset in [0, 100, 200, 300, 400]:
        url = "https://api.gbif.org/v1/occurrence/search"
        params = {
            "scientificName": "Ramaria pallida",
            "mediaType": "StillImage",
            "limit": 100,
            "offset": offset,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            for occ in data.get("results", []):
                for m in occ.get("media", []):
                    img_url = m.get("identifier") or m.get("references")
                    if img_url and img_url.startswith("http"):
                        all_urls.append(img_url)
        except Exception as e:
            print(f"  GBIF offset {offset} error: {e}")
        time.sleep(0.5)

    unique = list(set(all_urls))
    print(f"  Found {len(unique)} unique image URLs")
    return unique


def main():
    print(f"Downloading Ramaria pallida images to {IMAGES_DIR}")
    print(f"Target: {TARGET} images\n")

    downloaded = len(list(IMAGES_DIR.glob("*")))
    if downloaded >= TARGET:
        print(f"Already have {downloaded} images. Done.")
        return

    seen_urls = set()
    skipped = 0

    # Source 1: iNaturalist global
    observations = download_inaturalist_global()
    for obs in observations:
        if downloaded >= TARGET:
            break
        for photo in obs.get("photos", []):
            if downloaded >= TARGET:
                break
            img_url = photo.get("url", "")
            if not img_url or img_url in seen_urls:
                continue
            img_bytes = try_inaturalist_sizes(img_url)
            if img_bytes and verify_image(img_bytes):
                seen_urls.add(img_url)
                filename = save_image(img_bytes, IMAGES_DIR, "Ramaria_pallida", downloaded + 1)
                downloaded += 1
                print(f"  [iNat {downloaded}/{TARGET}] {filename} ({len(img_bytes)//1024}KB)")
                time.sleep(0.3)
            else:
                skipped += 1

    # Source 2: GBIF
    if downloaded < TARGET:
        gbif_urls = download_gbif_images()
        for img_url in gbif_urls:
            if downloaded >= TARGET:
                break
            if img_url in seen_urls:
                continue
            try:
                resp = requests.get(img_url, timeout=20)
                if resp.status_code != 200:
                    skipped += 1
                    continue
                if not resp.headers.get("Content-Type", "").startswith("image/"):
                    skipped += 1
                    continue
                if len(resp.content) < MIN_FILE_SIZE_KB * 1024:
                    skipped += 1
                    continue
                if not verify_image(resp.content):
                    skipped += 1
                    continue

                seen_urls.add(img_url)
                filename = save_image(resp.content, IMAGES_DIR, "Ramaria_pallida", downloaded + 1)
                downloaded += 1
                print(f"  [GBIF {downloaded}/{TARGET}] {filename} ({len(resp.content)//1024}KB)")
                time.sleep(0.3)
            except Exception:
                skipped += 1
                continue

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped")
    print(f"Total images in {IMAGES_DIR}: {len(list(IMAGES_DIR.glob('*')))}")


if __name__ == "__main__":
    main()
