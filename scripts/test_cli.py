#!/usr/bin/env python3
"""
CLI tool for testing the Mushroom Identification API.

Usage examples:
  # Check API is running
  python3 scripts/test_cli.py health

  # Identify a single image (Step 1 + CNN only, fast)
  python3 scripts/test_cli.py identify path/to/mushroom.jpg

  # Full 4-step pipeline
  python3 scripts/test_cli.py identify path/to/mushroom.jpg --full

  # Also print raw JSON
  python3 scripts/test_cli.py identify path/to/mushroom.jpg --json

  # Batch test all images in a folder
  python3 scripts/test_cli.py batch /home/iannyf/projekt/Mushroom_examples/Kantarell/
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:8000"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _colour(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _colour(t, "32")
def yellow(t): return _colour(t, "33")
def red(t):    return _colour(t, "31")
def bold(t):   return _colour(t, "1")
def cyan(t):   return _colour(t, "36")


def print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def check_alive() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_health(args) -> int:
    print(f"Checking {BASE_URL}/health …")
    if check_alive():
        print(green("✓ API is up and running"))
        return 0
    else:
        print(red("✗ API is not reachable. Is it running? (make start)"))
        return 1


def _step1(image_path: Path) -> dict:
    with image_path.open("rb") as f:
        r = requests.post(f"{BASE_URL}/identify", files={"image": (image_path.name, f, "image/jpeg")}, timeout=60)
    r.raise_for_status()
    return r.json()


def _print_step1(result: dict, image_name: str) -> None:
    step1 = result.get("trait_extraction", {})
    ml    = step1.get("ml_prediction", {})
    top   = result.get("top_prediction") or ml.get("top_species", "?")
    conf  = result.get("overall_confidence", ml.get("confidence", 0))
    method = ml.get("method", "?")

    print(bold(f"\n{'─'*55}"))
    print(bold(f"  Image : {image_name}"))
    print(bold(f"{'─'*55}"))
    print(f"  {'Prediction':<18} {green(top)}")
    cnn_conf_str = f"{ml.get('confidence', 0)*100:.1f}%"
    print(f"  {'CNN confidence':<18} {yellow(cnn_conf_str)}  [{method}]")
    print(f"  {'Overall (hybrid)':<18} {yellow(f'{conf*100:.1f}%')}")

    topk = ml.get("top_k", [])
    if topk:
        print(f"\n  Top-{len(topk)} CNN candidates:")
        for i, entry in enumerate(topk, 1):
            bar_len = int(entry["confidence"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {i}. {entry['species']:<25} {bar}  {entry['confidence']*100:.1f}%")

    warnings = result.get("safety_warnings", [])
    if warnings:
        print()
        for w in warnings:
            print(f"  {red(w)}")


def cmd_identify(args) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        print(red(f"File not found: {image_path}"))
        return 1

    if not check_alive():
        print(red("API is not reachable. Run: make start"))
        return 1

    print(f"\nSending {cyan(image_path.name)} to API …")
    t0 = time.time()

    # ── Step 1 ──────────────────────────────────────────────────────────────
    trait_extraction_result = _step1(image_path)
    elapsed = time.time() - t0
    _print_step1(trait_extraction_result, image_path.name)
    print(f"\n  {cyan(f'Step 1 completed in {elapsed:.2f}s')}")

    if not args.full:
        if args.json:
            print_json(trait_extraction_result)
        return 0

    # ── Step 2 (key traversal) ───────────────────────────────────────────────
    print(bold(f"\n{'─'*55}"))
    print(bold("  Step 2 — Key traversal"))
    print(bold(f"{'─'*55}"))

    visible_traits = trait_extraction_result.get("trait_extraction", {}).get("visible_traits", {})
    payload = {"visible_traits": visible_traits}

    r2 = requests.post(f"{BASE_URL}/identify/Species_tree_traversal/start", json=payload, timeout=30)
    r2.raise_for_status()
    s2 = r2.json()
    session_id = s2.get("session_id")

    rounds = 0
    while s2.get("status") == "question" and rounds < 20:
        q   = s2.get("question", "?")
        opts = s2.get("options", [])
        print(f"\n  Q: {q}")
        if opts:
            for i, o in enumerate(opts, 1):
                print(f"     {i}. {o}")
            # Auto-pick first option in non-interactive mode
            chosen = opts[0]
            print(f"  → Auto-selecting: {yellow(chosen)}")
        else:
            chosen = "yes"
            print(f"  → Auto-answering: {yellow(chosen)}")

        r2 = requests.post(
            f"{BASE_URL}/identify/Species_tree_traversal/answer",
            json={"session_id": session_id, "answer": chosen},
            timeout=30,
        )
        r2.raise_for_status()
        s2 = r2.json()
        rounds += 1

    step2_species = s2.get("species", "unknown")
    print(f"\n  Key conclusion: {green(step2_species)}")

    # ── Step 3 (trait comparison) ────────────────────────────────────────────
    print(bold(f"\n{'─'*55}"))
    print(bold("  Step 3 — Trait comparison"))
    print(bold(f"{'─'*55}"))

    r3 = requests.post(
        f"{BASE_URL}/identify/comparison/compare",
        json={"swedish_name": step2_species, "visible_traits": visible_traits},
        timeout=30,
    )
    r3.raise_for_status()
    s3 = r3.json()

    trait_match = s3.get("trait_match", {})
    match_score = trait_match.get("score", 0)
    lookalikes  = s3.get("lookalikes", [])
    print(f"  Trait match score : {yellow(f'{match_score*100:.1f}%')}")
    if lookalikes:
        la_names = [la.get("swedish_name", "?") for la in lookalikes]
        print(f"  Lookalikes        : {', '.join(la_names)}")

    # ── Step 4 (final) ───────────────────────────────────────────────────────
    print(bold(f"\n{'─'*55}"))
    print(bold("  Step 4 — Final result"))
    print(bold(f"{'─'*55}"))

    r4 = requests.post(
        f"{BASE_URL}/identify/prediction/finalize",
        json={"trait_extraction_result": trait_extraction_result, "Species_tree_traversal_result": s2, "comparison_result": s3},
        timeout=30,
    )
    r4.raise_for_status()
    s4 = r4.json()

    final_species = s4.get("final_species") or s4.get("top_prediction", "?")
    final_conf    = s4.get("final_confidence", s4.get("overall_confidence", 0))
    edibility     = s4.get("edibility", s4.get("safety_rating", "?"))

    print(f"  {'Final species':<18} {green(final_species)}")
    print(f"  {'Confidence':<18} {yellow(f'{final_conf*100:.1f}%')}")
    print(f"  {'Edibility':<18} {edibility}")

    total = time.time() - t0
    print(f"\n  {cyan(f'Full pipeline in {total:.2f}s')}")

    if args.json:
        print("\n── Raw JSON output ──")
        print_json(s4)

    return 0


def cmd_batch(args) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(red(f"Not a directory: {folder}"))
        return 1

    if not check_alive():
        print(red("API is not reachable. Run: make start"))
        return 1

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in sorted(folder.iterdir()) if p.suffix in exts]

    if not images:
        print(yellow(f"No images found in {folder}"))
        return 1

    print(bold(f"\nBatch test — {len(images)} images from {folder.name}/\n"))
    results = []

    for img in images:
        try:
            result = _step1(img)
            ml   = result.get("trait_extraction", {}).get("ml_prediction", {})
            top  = result.get("top_prediction") or ml.get("top_species", "?")
            conf = ml.get("confidence", 0)
            method = ml.get("method", "?")
            status = green("✓")
            results.append((img.name, top, conf, method, None))
            print(f"  {status} {img.name:<35} → {top:<25} {conf*100:.1f}%  [{method}]")
        except Exception as e:
            results.append((img.name, None, 0, None, str(e)))
            print(f"  {red('✗')} {img.name:<35} → ERROR: {e}")

    success = sum(1 for r in results if r[4] is None)
    print(bold(f"\n  {success}/{len(results)} successful"))
    return 0 if success == len(results) else 1


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        prog="test_cli.py",
        description="CLI test tool for the Mushroom Identification API",
    )
    parser.add_argument("--url", default=BASE_URL, help=f"API base URL (default: {BASE_URL})")
    sub = parser.add_subparsers(dest="command", required=True)

    # health
    sub.add_parser("health", help="Check if the API is running")

    # identify
    p_id = sub.add_parser("identify", help="Identify a mushroom from an image")
    p_id.add_argument("image", help="Path to image file")
    p_id.add_argument("--full", action="store_true", help="Run full 4-step pipeline")
    p_id.add_argument("--json", action="store_true", help="Also print raw JSON output")

    # batch
    p_bt = sub.add_parser("batch", help="Run Step 1 on all images in a folder")
    p_bt.add_argument("folder", help="Path to folder containing images")

    args = parser.parse_args()

    BASE_URL = args.url

    dispatch = {"health": cmd_health, "identify": cmd_identify, "batch": cmd_batch}
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()