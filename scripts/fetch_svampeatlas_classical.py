"""
Fetch classical mushroom shape images from the Atlas of Danish Fungi (Svampeatlas)
via GBIF. Paced to respect rate limits.
"""
import requests
import os
import time
import random

OUTDIR = "data/Yolov8/new_images/svampeatlas_classical"
TARGET = 100
PER_GENUS_MAX = 8
os.makedirs(OUTDIR, exist_ok=True)

DATASET_KEY = "84d26682-f762-11e1-a439-00145eb45e9a"

GENERA = [
    "Amanita", "Boletus", "Cantharellus", "Russula", "Agaricus",
    "Coprinus", "Lactarius", "Macrolepiota", "Craterellus",
    "Hydnum", "Gyromitra", "Pleurotus", "Calocybe", "Lepista",
    "Imleria", "Hygrophoropsis", "Lycoperdon", "Ramaria",
    "Cortinarius", "Inocybe", "Tricholoma", "Marasmius",
    "Panellus", "Ganoderma", "Scleroderma"
]

HEADERS = {"User-Agent": "MushroomDatasetBot/1.0"}


def get_taxon_id(genus_name):
    url = "https://api.gbif.org/v1/species/search"
    params = {"q": genus_name, "rank": "GENUS", "limit": 1}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            return results[0]["key"]
    except Exception as e:
        print(f"  Taxon lookup failed for {genus_name}: {e}")
    return None


def fetch_observations(taxon_id, offset=0, limit=200):
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "datasetKey": DATASET_KEY,
        "taxonKey": taxon_id,
        "mediaType": "StillImage",
        "limit": limit,
        "offset": offset,
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("results", []), data.get("count", 0)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("  Rate limited. Waiting 10s...")
            time.sleep(10)
            return fetch_observations(taxon_id, offset, limit)
        print(f"  Observation fetch failed: {e}")
        return [], 0
    except Exception as e:
        print(f"  Observation fetch failed: {e}")
        return [], 0


def download_photo(url, outpath):
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        with open(outpath, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def main():
    downloaded = 0
    seen_ids = set()
    random.shuffle(GENERA)

    for genus in GENERA:
        if downloaded >= TARGET:
            break

        taxon_id = get_taxon_id(genus)
        time.sleep(1.0)  # Respect rate limits between taxon lookups
        if not taxon_id:
            continue

        genus_count = 0
        offset = 0
        max_attempts = 5

        for attempt in range(max_attempts):
            if downloaded >= TARGET or genus_count >= PER_GENUS_MAX:
                break

            obs_list, total = fetch_observations(taxon_id, offset=offset, limit=200)
            if not obs_list:
                break

            random.shuffle(obs_list)

            for obs in obs_list:
                if downloaded >= TARGET or genus_count >= PER_GENUS_MAX:
                    break

                obs_id = obs.get("gbifID")
                if obs_id in seen_ids:
                    continue
                seen_ids.add(obs_id)

                media = obs.get("media", [])
                if not media:
                    continue

                img_url = media[0].get("identifier")
                if not img_url or "thumb" in img_url.lower():
                    continue

                fname = f"{genus}_gbif{obs_id}.jpg"
                outpath = os.path.join(OUTDIR, fname)
                if os.path.exists(outpath):
                    continue

                if download_photo(img_url, outpath):
                    downloaded += 1
                    genus_count += 1
                    print(f"  [{downloaded}/{TARGET}] {genus} #{genus_count}: {fname}")
                    time.sleep(0.5)

            offset += len(obs_list)
            time.sleep(1.0)

        print(f"  -> {genus}: {genus_count} images")
        time.sleep(1.5)

    print(f"\n{'='*50}")
    print(f"Done. Total downloaded: {downloaded}")
    print(f"Saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
