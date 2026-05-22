"""
Fetch ~100 research-grade single-mushroom images from iNaturalist.
Caps downloads per genus to force diversity across taxa.
"""
import requests
import os
import time
import random

OUTDIR = "data/Yolov8/new_images/classical"
TARGET = 100
PER_GENUS_MAX = 8  # Cap per genus so we get ~12+ genera represented
os.makedirs(OUTDIR, exist_ok=True)

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
    url = "https://api.inaturalist.org/v1/taxa/autocomplete"
    params = {"q": genus_name, "rank": "genus", "per_page": 1}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            return results[0]["id"]
    except Exception as e:
        print(f"  Taxon lookup failed for {genus_name}: {e}")
    return None


def fetch_observations(taxon_id, per_page=200, page=1):
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photos": "true",
        "per_page": per_page,
        "page": page,
        "order": "desc",
        "order_by": "created_at",
        "iconic_taxa": "Fungi",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"  Observation fetch failed: {e}")
        return []


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


def description_has_cluster(text):
    if not text:
        return False
    text = text.lower()
    cluster_words = ["cluster", "clusters", "group", "groups", "many",
                     "several", "dozens", "numerous", "abundant",
                     " fruiting bodies", " specimens"]
    return any(w in text for w in cluster_words)


def main():
    downloaded = 0
    seen_obs_ids = set()
    random.shuffle(GENERA)

    for genus in GENERA:
        if downloaded >= TARGET:
            break

        taxon_id = get_taxon_id(genus)
        if not taxon_id:
            continue

        genus_count = 0
        page = 1
        max_pages = 5

        while page <= max_pages and downloaded < TARGET and genus_count < PER_GENUS_MAX:
            obs_list = fetch_observations(taxon_id, per_page=200, page=page)
            if not obs_list:
                break

            for obs in obs_list:
                if downloaded >= TARGET or genus_count >= PER_GENUS_MAX:
                    break

                obs_id = obs.get("id")
                if obs_id in seen_obs_ids:
                    continue
                seen_obs_ids.add(obs_id)

                desc = obs.get("description") or ""
                if description_has_cluster(desc):
                    continue

                photos = obs.get("photos", [])
                if not photos:
                    continue

                photo = photos[0]
                photo_url = photo.get("url")
                if not photo_url:
                    continue

                photo_url = photo_url.replace("/square.", "/medium.")

                fname = f"{genus}_obs{obs_id}.jpg"
                outpath = os.path.join(OUTDIR, fname)
                if os.path.exists(outpath):
                    continue

                if download_photo(photo_url, outpath):
                    downloaded += 1
                    genus_count += 1
                    print(f"  [{downloaded}/{TARGET}] {genus} #{genus_count}: {fname}")
                    time.sleep(0.3)

            page += 1
            time.sleep(0.4)

        print(f"  -> {genus}: {genus_count} images")

    print(f"\n{'='*50}")
    print(f"Done. Total downloaded: {downloaded}")
    print(f"Saved to: {OUTDIR}")
    print("\nIMPORTANT: Review images manually. Discard cluster/group photos.")


if __name__ == "__main__":
    main()
