"""
Download 250 research-grade Russula images from iNaturalist.
6 edible species, split evenly between cap-view and gill-view (underside).
"""
import requests
import os
import time
import random

OUTDIR = "data/Yolov8/new_images/russula_batch"
os.makedirs(OUTDIR, exist_ok=True)

HEADERS = {"User-Agent": "MushroomDatasetBot/1.0"}

# 6 edible Russula species
SPECIES = [
    "Russula cyanoxantha",
    "Russula vesca",
    "Russula rosea",
    "Russula xerampelina",
    "Russula claroflava",
    "Russula mustelina",
]

# For gill-view searches, rotate through these keywords
GILL_KEYWORDS = ["gills", "gill", "lamellae", "underside"]

TARGET_PER_SPECIES = 42  # 42 cap + 42 gill ≈ 84 per species, but we cap at ~42 total per species split
# Actually: 21 cap + 21 gill per species × 6 = 252 ≈ 250
CAP_PER_SPECIES = 21
GILL_PER_SPECIES = 21


def get_taxon_id(sci_name):
    url = "https://api.inaturalist.org/v1/taxa"
    params = {"q": sci_name, "rank": "species", "per_page": 1}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            return results[0]["id"]
    except Exception as e:
        print(f"  Taxon lookup failed for {sci_name}: {e}")
    return None


def fetch_observations(taxon_id, keyword=None, per_page=200, page=1):
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photos": "true",
        "per_page": per_page,
        "page": page,
        "order": "desc",
        "order_by": "created_at",
    }
    if keyword:
        params["q"] = keyword

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"  Fetch failed: {e}")
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


def fetch_for_species(taxon_id, species_name, keyword, target, label):
    """Fetch `target` images for a species, optionally with a keyword."""
    downloaded = 0
    seen_ids = set()
    page = 1
    max_pages = 10
    species_short = species_name.replace(" ", "_")

    while page <= max_pages and downloaded < target:
        obs_list = fetch_observations(taxon_id, keyword=keyword, per_page=200, page=page)
        if not obs_list:
            break

        random.shuffle(obs_list)

        for obs in obs_list:
            if downloaded >= target:
                break

            obs_id = obs.get("id")
            if obs_id in seen_ids:
                continue
            seen_ids.add(obs_id)

            photos = obs.get("photos", [])
            if not photos:
                continue

            photo = photos[0]
            photo_url = photo.get("url")
            if not photo_url:
                continue
            photo_url = photo_url.replace("/square.", "/medium.")

            fname = f"{species_short}_obs{obs_id}_{label}.jpg"
            outpath = os.path.join(OUTDIR, fname)
            if os.path.exists(outpath):
                continue

            if download_photo(photo_url, outpath):
                downloaded += 1
                print(f"    [{label}] [{downloaded}/{target}] {fname}")
                time.sleep(0.3)

        page += 1
        time.sleep(0.5)

    return downloaded


def main():
    total_downloaded = 0
    total_cap = 0
    total_gill = 0

    for species in SPECIES:
        print(f"\n=== {species} ===")
        taxon_id = get_taxon_id(species)
        time.sleep(1.0)
        if not taxon_id:
            continue
        print(f"  taxon_id: {taxon_id}")

        # CAP view: no keyword
        cap_count = fetch_for_species(taxon_id, species, None, CAP_PER_SPECIES, "cap")
        total_cap += cap_count
        total_downloaded += cap_count
        print(f"  -> cap: {cap_count}")
        time.sleep(1.0)

        # GILL view: rotate keywords
        gill_keyword = GILL_KEYWORDS[SPECIES.index(species) % len(GILL_KEYWORDS)]
        gill_count = fetch_for_species(taxon_id, species, gill_keyword, GILL_PER_SPECIES, "gill")
        total_gill += gill_count
        total_downloaded += gill_count
        print(f"  -> gill ({gill_keyword}): {gill_count}")
        time.sleep(1.0)

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Total: {total_downloaded}")
    print(f"  Cap view: {total_cap}")
    print(f"  Gill view: {total_gill}")
    print(f"  Saved to: {OUTDIR}")
    print(f"\nIMPORTANT: Review manually before annotating.")
    print(f"  - Keep only single-fruiting-body photos")
    print(f"  - 'cap' images → annotate cap + stem")
    print(f"  - 'gill' images → annotate underside (solid mask)")


if __name__ == "__main__":
    main()
