"""
Step 3 — Trait Database Comparator

Given the species candidate produced by Step 2 (key.xml traversal) this module:
  1. Resolves the Swedish common name to a species record in species.csv.
  2. Loads the species' morphological trait profile from species_traits.xml.
  3. Compares Step 1 visible_traits against the database profile, producing a
     match score and a breakdown of matching / conflicting traits.
  4. Looks up all lookalike pairs from lookalikes.csv, loads both species'
     trait profiles, and highlights the key distinguishing features.

Public API
----------
  comparator = TraitDatabaseComparator(data_dir)

  result = comparator.compare(swedish_name, visible_traits)

  # Result shape:
  # {
  #   "status":             "ok" | "species_not_found",
  #   "candidate":          {species_id, swedish_name, english_name,
  #                          scientific_name, edible, toxicity_level},
  #   "name_match_score":   float  0-1,
  #   "trait_match": {
  #     "score":            float  0-1,
  #     "matched":          [{trait, visible_value, db_value, quality}],
  #     "conflicts":        [{trait, visible_value, db_value, severity}],
  #     "not_comparable":   [trait_name],
  #   },
  #   "lookalikes":         [{species metadata + distinguishing_features
  #                           + trait_differences + safety_alert}],
  #   "safety_alert":       bool   True if any CRITICAL lookalike or toxic species
  # }
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.dataset_utils import load_species_traits_xml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour synonym groups used for fuzzy trait matching
# ---------------------------------------------------------------------------

_COLOUR_GROUPS: List[frozenset] = [
    frozenset({"orange", "yellow-orange", "orange-yellow", "reddish-orange", "pale-orange"}),
    frozenset({"yellow", "pale-yellow", "golden", "golden-yellow", "cream-yellow"}),
    frozenset({"red", "red with white spots", "scarlet", "crimson", "reddish"}),
    frozenset({"white", "pure-white", "off-white", "cream", "whitish", "pale"}),
    frozenset({"brown", "dark-brown", "tan", "olive-brown", "chestnut", "cinnamon",
               "tan to dark brown", "dark brown", "light-brown", "pale brown",
               "buff", "umber", "tawny"}),
    frozenset({"grey", "gray", "grey-brown", "gray-brown", "greyish", "grayish", "silver-grey"}),
    frozenset({"black", "dark", "dark-grey", "dark-gray", "blackish"}),
    frozenset({"green", "olive", "olive-green", "greenish"}),
    frozenset({"blue", "blue-grey", "bluish", "blue-green", "blue-on-cut"}),
    frozenset({"pink", "pinkish", "rose", "lilac", "violet", "purple"}),
]


def _colour_group(c: str) -> Optional[frozenset]:
    """Return the colour group containing *c*, or None."""
    cl = c.lower().strip()
    for g in _COLOUR_GROUPS:
        for member in g:
            if cl in member or member in cl:
                return g
    return None


def _colours_match(visible: str, db_value: str) -> str:
    """
    Compare a visible colour string against a DB trait value string.
    Returns 'exact', 'partial', or 'conflict'.
    """
    v = visible.lower().strip()
    db = db_value.lower()

    if v in db or db in v:
        return "exact"

    vg = _colour_group(v)
    # DB value may be a pipe-separated list or range description
    for token in re.split(r"[|,/ ]+", db):
        token = token.strip()
        if not token:
            continue
        if v in token or token in v:
            return "exact"
        tg = _colour_group(token)
        if vg and tg and (vg & tg):
            return "partial"

    return "conflict"


# Shape synonyms
_SHAPE_SYNONYMS: Dict[str, List[str]] = {
    "funnel-shaped":  ["funnel", "funnel-shaped", "vase", "trumpet", "infundibuliform"],
    "convex":         ["convex", "rounded", "dome", "hemispherical"],
    "flat":           ["flat", "plane", "depressed", "flattened"],
    "bell-shaped":    ["bell", "campanulate", "conical", "umbonate"],
    "wavy":           ["wavy", "irregular", "undulate", "lobed", "wavy-edged"],
    "convex|flat":    ["convex", "flat", "convex|flat"],
    "smooth":         ["smooth"],
}


def _shapes_match(visible: str, db_value: str) -> str:
    v = visible.lower().strip()
    db = db_value.lower()
    if v in db or db in v:
        return "exact"
    synonyms = _SHAPE_SYNONYMS.get(v, [v])
    for syn in synonyms:
        if syn in db:
            return "partial"
    # check reverse
    for k, syns in _SHAPE_SYNONYMS.items():
        if v in syns:
            for syn in syns:
                if syn in db:
                    return "partial"
    return "conflict"


def _texture_match(visible: str, db_value: str) -> str:
    v = visible.lower()
    db = db_value.lower()
    if v in db:
        return "exact"
    texture_map = {
        "smooth":  ["smooth", "silky", "glabrous", "slippery"],
        "fibrous": ["fibrous", "silky", "streaked", "radially fibrous"],
        "scaly":   ["scaly", "squamulose", "woolly", "rough", "fibrous-scaly",
                    "shaggy", "granular"],
    }
    for syn in texture_map.get(v, [v]):
        if syn in db:
            return "partial" if syn != v else "exact"
    return "conflict"


def _ridges_match(has_ridges: bool, attachment_value: str) -> str:
    av = attachment_value.lower()
    ridge_words = {"decurrent", "ridges", "folds", "forked", "blunt"}
    gill_words  = {"free", "adnate", "attached", "notched", "sinuate",
                   "adnexed", "emarginate"}
    is_ridge_attachment = any(w in av for w in ridge_words)
    is_gill_attachment  = any(w in av for w in gill_words)

    if has_ridges and is_ridge_attachment:
        return "exact"
    if has_ridges and is_gill_attachment:
        return "conflict"
    if not has_ridges and is_gill_attachment:
        return "exact"
    if not has_ridges and is_ridge_attachment:
        return "conflict"
    return "partial"   # pores, spines, etc. — inconclusive


# ---------------------------------------------------------------------------
# Scoring weights for each comparable trait
# ---------------------------------------------------------------------------

_TRAIT_WEIGHTS: Dict[str, float] = {
    "cap_color":     0.30,
    "cap_shape":     0.25,
    "ridges":        0.20,
    "cap_texture":   0.15,
    "stem_color":    0.10,
}

_QUALITY_SCORE = {"exact": 1.0, "partial": 0.5, "conflict": 0.0}


# ---------------------------------------------------------------------------
# Name resolver: Swedish common name → species row
# ---------------------------------------------------------------------------
# Name aliases: key.xml Swedish common names → species_id in our DB
# These are needed where the key.xml name differs from the DB swedish_name.
# ---------------------------------------------------------------------------

_KEY_XML_ALIASES: Dict[str, str] = {
    # key.xml name (lowercase)          : species_id in species.csv
    "stensopp":                           "BO.ED",   # Karljohan = Porcini
    "stensopp (karljohanssvamp)":         "BO.ED",
    "karljohanssvamp":                    "BO.ED",
    "ängschampinjon":                     "AG.CA",   # Ängssvamp
    "stolt fjällskivling":                "MA.PR",   # Parasolsvamp
    "morkel":                             "MO.ES",
    "toppmurkla":                         "MO.ES",
    "stenmurkla":                         "GY.ES",   # False morel
    "jätteröksvamp":                      "LY.PE",   # Röksvamp / Puffball
    "honungsskivling":                    "AR.ME",   # Hallon
    "svart trumpetsvamp":                 "CR.CO",
    "grå kantarell":                      "CA.CN",
    "rödgul trumpetsvamp":                "CA.AU",
    "trattkantarell":                     "CA.TU",
    "smörsopp":                           "SU.LU",
    "citronslemskivling":                 "GO.GL",
    "björksopp":                          "BO.BA",
    "blek taggsvamp":                     "HY.RE",
    "rödgul taggsvamp":                   "HY.RU",
    "druvfingersvamp":                    "RA.BO",
    "fårticka":                           "AL.OV",
    "brödticka":                          "AL.CO",
    "gallsopp":                           "TY.FE",
    "krusig blomkålssvamp":               "SP.CR",
}


def _name_similarity(query: str, candidate: str) -> float:
    """Simple token overlap similarity between two Swedish name strings (0-1)."""
    q_tokens = set(re.split(r"[\s\(\)/,-]+", query.lower()))
    c_tokens = set(re.split(r"[\s\(\)/,-]+", candidate.lower()))
    q_tokens.discard(""); c_tokens.discard("")
    if not q_tokens:
        return 0.0
    overlap = q_tokens & c_tokens
    return len(overlap) / max(len(q_tokens), len(c_tokens))


def _find_best_species(
    query: str,
    species_list: List[Dict[str, str]],
    threshold: float = 0.35,
) -> Tuple[Optional[Dict[str, str]], float]:
    """
    Find the best-matching species row for a Swedish name query.
    First checks the explicit alias map, then falls back to token-overlap
    fuzzy matching. Returns (row, score) or (None, 0.0).
    """
    # Fast alias lookup first
    species_by_id: Dict[str, Dict[str, str]] = {r["species_id"]: r for r in species_list}
    alias_id = _KEY_XML_ALIASES.get(query.lower().strip())
    if alias_id and alias_id in species_by_id:
        return species_by_id[alias_id], 1.0

    best_row: Optional[Dict[str, str]] = None
    best_score = 0.0
    for row in species_list:
        score = max(
            _name_similarity(query, row["swedish_name"]),
            _name_similarity(query, row["english_name"]),
        )
        if score > best_score:
            best_score = score
            best_row = row
    if best_score < threshold:
        return None, 0.0
    return best_row, round(best_score, 3)


# ---------------------------------------------------------------------------
# Trait loader
# ---------------------------------------------------------------------------

def _load_traits(
    species_id: str,
    trait_rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    """
    Return nested dict: {category: {trait_name: trait_value}}
    for the given species_id.
    """
    result: Dict[str, Dict[str, str]] = {}
    for row in trait_rows:
        if row["species_id"] != species_id:
            continue
        cat = row["trait_category"]
        if cat not in result:
            result[cat] = {}
        result[cat][row["trait_name"]] = row["trait_value"]
    return result


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def _compare_visible_to_db(
    visible: Dict[str, Any],
    db_traits: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """
    Compare Step 1 visible_traits against the loaded DB trait profile.
    Returns a dict with score, matched, conflicts, not_comparable.
    """
    matched: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    not_comparable: List[str] = []

    cap   = db_traits.get("CAP",   {})
    gills = db_traits.get("GILLS", {})
    stem  = db_traits.get("STEM",  {})

    def _record(trait_key: str, visible_val: Any, db_val: str, quality: str,
                weight: float, is_conflict: bool) -> None:
        entry = {
            "trait":         trait_key,
            "visible_value": str(visible_val),
            "db_value":      db_val,
            "quality":       quality,
            "weight":        weight,
        }
        (conflicts if is_conflict else matched).append(entry)

    # --- cap colour ---
    dom = visible.get("dominant_color", "")
    if dom and "color" in cap:
        q = _colours_match(dom, cap["color"])
        _record("cap_color", dom, cap["color"], q,
                _TRAIT_WEIGHTS["cap_color"], q == "conflict")
    else:
        not_comparable.append("cap_color")

    # --- cap shape ---
    shape = visible.get("cap_shape", "")
    if shape and shape != "unknown" and "shape" in cap:
        q = _shapes_match(shape, cap["shape"])
        _record("cap_shape", shape, cap["shape"], q,
                _TRAIT_WEIGHTS["cap_shape"], q == "conflict")
    else:
        not_comparable.append("cap_shape")

    # --- surface texture ---
    texture = visible.get("surface_texture", "")
    if texture and "surface_texture" in cap:
        q = _texture_match(texture, cap["surface_texture"])
        _record("cap_texture", texture, cap["surface_texture"], q,
                _TRAIT_WEIGHTS["cap_texture"], q == "conflict")
    else:
        not_comparable.append("cap_texture")

    # --- ridges / gill attachment ---
    if "attachment" in gills:
        q = _ridges_match(bool(visible.get("has_ridges")), gills["attachment"])
        _record("ridges", visible.get("has_ridges"), gills["attachment"], q,
                _TRAIT_WEIGHTS["ridges"], q == "conflict")
    else:
        not_comparable.append("ridges")

    # --- stem colour (secondary cue) ---
    if visible.get("secondary_color") and "color" in stem:
        q = _colours_match(visible["secondary_color"], stem["color"])
        _record("stem_color", visible["secondary_color"], stem["color"], q,
                _TRAIT_WEIGHTS["stem_color"], q == "conflict")
    else:
        not_comparable.append("stem_color")

    # --- weighted score ---
    total_weight = 0.0
    weighted_sum = 0.0
    for entry in matched + conflicts:
        w = entry["weight"]
        total_weight += w
        weighted_sum += _QUALITY_SCORE[entry["quality"]] * w

    score = round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.0

    return {
        "score":          score,
        "matched":        matched,
        "conflicts":      conflicts,
        "not_comparable": not_comparable,
    }


# ---------------------------------------------------------------------------
# Main comparator class
# ---------------------------------------------------------------------------

class TraitDatabaseComparator:
    """
    Loads species.csv, species_traits.csv, and lookalikes.csv once at init
    and provides fast in-memory lookups for Step 3 comparisons.
    """

    def __init__(self, data_dir: str) -> None:
        data_path = Path(data_dir)
        self._species = self._load_csv(data_path / "species.csv")
        self._trait_rows = load_species_traits_xml(data_path / "species_traits.xml").to_dict("records")
        self._lookalike_rows = self._load_csv(data_path / "lookalikes.csv")
        # Index species by species_id for fast lookup
        self._species_by_id: Dict[str, Dict[str, str]] = {
            r["species_id"]: r for r in self._species
        }
        logger.info(
            "TraitDatabaseComparator loaded: %d species, %d trait rows, %d lookalike pairs",
            len(self._species), len(self._trait_rows), len(self._lookalike_rows),
        )

    @staticmethod
    def _load_csv(path: Path) -> List[Dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    # ------------------------------------------------------------------
    def compare(
        self,
        swedish_name: str,
        visible_traits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Full Step 3 comparison.

        Args:
            swedish_name:   Swedish species name from Step 2 conclusion.
            visible_traits: Step 1 visible_traits dict.

        Returns:
            Structured result dict (see module docstring).
        """
        # 1. Resolve name → species record
        species_row, name_score = _find_best_species(swedish_name, self._species)
        if species_row is None:
            return {
                "status":       "species_not_found",
                "query":        swedish_name,
                "message":      f"No species matched '{swedish_name}' in the database "
                                f"(best score below threshold).",
                "lookalikes":   [],
                "safety_alert": False,
            }

        species_id = species_row["species_id"]
        logger.debug("Step 3 candidate: %s → %s (score %.2f)", swedish_name, species_id, name_score)

        # 2. Load and compare traits
        db_traits = _load_traits(species_id, self._trait_rows)
        trait_match = _compare_visible_to_db(visible_traits, db_traits)

        # 3. Lookalike lookup
        lookalikes = self._build_lookalikes(species_id, visible_traits)

        # 4. Safety alert if species is toxic or any CRITICAL lookalike
        is_toxic = species_row.get("toxicity_level", "SAFE").upper() in {
            "TOXIC", "EXTREMELY_TOXIC", "PSYCHOACTIVE"
        }
        has_critical_lookalike = any(
            la.get("confusion_likelihood") == "CRITICAL" for la in lookalikes
        )
        safety_alert = is_toxic or has_critical_lookalike

        return {
            "status": "ok",
            "candidate": {
                "species_id":      species_id,
                "swedish_name":    species_row["swedish_name"],
                "english_name":    species_row["english_name"],
                "scientific_name": species_row["scientific_name"],
                "edible":          species_row.get("edible", "FALSE").upper() == "TRUE",
                "toxicity_level":  species_row.get("toxicity_level", "UNKNOWN"),
            },
            "name_match_score": name_score,
            "trait_match":      trait_match,
            "lookalikes":       lookalikes,
            "safety_alert":     safety_alert,
        }

    # ------------------------------------------------------------------
    def _build_lookalikes(
        self,
        species_id: str,
        visible_traits: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Find all lookalike pairs involving species_id and build rich
        comparison dicts with trait differences highlighted.
        """
        results = []
        for row in self._lookalike_rows:
            is_edible = row["edible_species_id"] == species_id
            is_toxic  = row["toxic_species_id"]  == species_id
            if not is_edible and not is_toxic:
                continue

            # The "other" species in the pair
            other_id = row["toxic_species_id"] if is_edible else row["edible_species_id"]
            if other_id == "NONE":
                continue

            other_row = self._species_by_id.get(other_id)
            if other_row is None:
                continue

            other_traits = _load_traits(other_id, self._trait_rows)
            candidate_traits = _load_traits(species_id, self._trait_rows)

            # Build a trait-diff summary (CAP and GILLS are most visible)
            trait_diffs = self._diff_traits(candidate_traits, other_traits)

            # Compare visible traits against the lookalike to see how similar
            other_match = _compare_visible_to_db(visible_traits, other_traits)

            results.append({
                "species_id":          other_id,
                "swedish_name":        other_row["swedish_name"],
                "english_name":        other_row["english_name"],
                "scientific_name":     other_row["scientific_name"],
                "edible":              other_row.get("edible", "FALSE").upper() == "TRUE",
                "toxicity_level":      other_row.get("toxicity_level", "UNKNOWN"),
                "confusion_likelihood": row.get("confusion_likelihood", "UNKNOWN"),
                "distinguishing_features": row.get("distinguishing_features", ""),
                "visual_similarity_to_image": other_match["score"],
                "key_trait_differences": trait_diffs,
                "safety_alert": other_row.get("toxicity_level", "SAFE").upper() in {
                    "TOXIC", "EXTREMELY_TOXIC",
                },
            })

        # Sort: most dangerous first, then by confusion likelihood
        likelihood_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}
        results.sort(key=lambda x: (
            not x["safety_alert"],
            likelihood_order.get(x["confusion_likelihood"], 4),
        ))
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _diff_traits(
        candidate: Dict[str, Dict[str, str]],
        other: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Compare CAP and GILLS trait dicts and return differences.
        Only includes traits that differ in a meaningful way.
        """
        diffs = []
        for category in ("CAP", "GILLS", "FLESH"):
            c_cat = candidate.get(category, {})
            o_cat = other.get(category, {})
            all_keys = set(c_cat) | set(o_cat)
            for key in sorted(all_keys):
                c_val = c_cat.get(key, "—")
                o_val = o_cat.get(key, "—")
                if c_val == o_val:
                    continue
                # Skip size ranges — too variable
                if key == "size_cm":
                    continue
                diffs.append({
                    "category":        category,
                    "trait":           key,
                    "candidate_value": c_val,
                    "lookalike_value": o_val,
                })
        return diffs
