"""Key-Tree Oracle — perfect traversal paths for A/B benchmarking.

Extracts every root-to-leaf path from key.xml, maps each decision to a
canonical ``species_id``, and exposes a deterministic 50/50 split so that
the benchmark can feed perfect ``pre_answers`` to the tree for the "oracle"
group while leaving the "non-oracle" group unchanged.
"""

from __future__ import annotations

import csv
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set

from models.trait_database_comparator import _KEY_XML_ALIASES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML → path extraction
# ---------------------------------------------------------------------------

def _collect_qa_paths(
    element: ET.Element,
    current_path: List[Dict[str, str]],
    all_paths: List[List[Dict[str, str]]],
) -> None:
    """Recursively collect root-to-leaf paths as lists of {question, answer} steps."""
    tag = element.tag

    if tag == "key":
        q = element.get("question", "")
        for child in element:
            _collect_qa_paths(child, [{"question": q, "answer": ""}], all_paths)

    elif tag == "condition":
        ans = element.get("answer", "")
        sub_q = element.get("question", "")

        # Update the last step's answer
        new_path = [dict(step) for step in current_path]
        if new_path:
            new_path[-1]["answer"] = ans

        # If there is a sub-question, push a new empty-answer step
        if sub_q:
            new_path.append({"question": sub_q, "answer": ""})

        # Only follow <decision> children (ignore <mixupdecision>)
        decision_children = [c for c in element if c.tag == "decision"]
        non_decision_children = [c for c in element if c.tag != "decision"]

        for child in decision_children:
            _collect_qa_paths(child, new_path, all_paths)

        for child in non_decision_children:
            _collect_qa_paths(child, new_path, all_paths)

    elif tag == "decision":
        name = element.get("namn", "")
        path_copy = [dict(step) for step in current_path]
        path_copy.append({"decision": name})
        all_paths.append(path_copy)


# ---------------------------------------------------------------------------
# Decision name → species_id mapping
# ---------------------------------------------------------------------------

def _build_decision_to_species_id(xml_path: str) -> Dict[str, str]:
    """Map every <decision namn=...> in key.xml to a species_id."""
    tree = ET.parse(xml_path)
    decisions: Set[str] = set()
    for elem in tree.iter("decision"):
        decisions.add(elem.get("namn", "").strip())

    # Load species.csv
    csv_path = Path(xml_path).parent / "species.csv"
    swedish_to_sid: Dict[str, str] = {}
    english_to_sid: Dict[str, str] = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("swedish_name"):
                    swedish_to_sid[row["swedish_name"].lower().strip()] = row["species_id"]
                if row.get("english_name"):
                    english_to_sid[row["english_name"].lower().strip()] = row["species_id"]

    mapping: Dict[str, str] = {}
    for d in decisions:
        d_lower = d.lower().strip()
        sid: Optional[str] = None

        # 1. Hard-coded alias table (most reliable)
        if d_lower in _KEY_XML_ALIASES:
            sid = _KEY_XML_ALIASES[d_lower]

        # 2. Exact Swedish name match
        if not sid and d_lower in swedish_to_sid:
            sid = swedish_to_sid[d_lower]

        # 3. Exact English name match
        if not sid and d_lower in english_to_sid:
            sid = english_to_sid[d_lower]

        if sid:
            mapping[d] = sid

    return mapping


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class OracleKeyTree:
    """Perfect key-tree oracle for benchmark A/B testing.

    Usage
    -----
    oracle = OracleKeyTree("data/raw/key.xml")
    pre_answers = oracle.get_pre_answers("CA.CI")   # Chanterelle path
    # pre_answers == {
    #     "Hur ser svampen ut?": "Undersidan har åsar eller ådror",
    #     "Vilken färg har svampen?": "Hela svampen är gul",
    # }
    """

    def __init__(self, xml_path: str, target_species_ids: Optional[List[str]] = None):
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"key.xml not found at {xml_path}")

        # Extract all root-to-leaf paths
        tree = ET.parse(str(self.xml_path))
        self._paths: List[List[Dict[str, str]]] = []
        _collect_qa_paths(tree.getroot(), [], self._paths)

        # Map decision names → species_id
        self._decision_to_sid = _build_decision_to_species_id(str(self.xml_path))

        # Build species_id → pre_answers dict (one primary path per species)
        self._oracle_answers: Dict[str, Dict[str, str]] = {}
        for path in self._paths:
            if not path or "decision" not in path[-1]:
                continue
            decision_name = path[-1]["decision"]
            sid = self._decision_to_sid.get(decision_name)
            if not sid:
                continue

            # Convert path steps to {question: answer} dict
            qa = {}
            for step in path:
                if "question" in step and "answer" in step:
                    qa[step["question"]] = step["answer"]

            # If a species maps to multiple decisions, keep the first one we see
            if sid not in self._oracle_answers:
                self._oracle_answers[sid] = qa
                logger.debug("Oracle path for %s via '%s': %d Q/A pairs", sid, decision_name, len(qa))

        # Filter to target species if provided (e.g. benchmark species only)
        if target_species_ids is not None:
            target_set = set(target_species_ids)
            available = {sid for sid in self._oracle_answers if sid in target_set}
        else:
            available = set(self._oracle_answers.keys())

        # Deterministic 50/50 split — sorted list, even indices = oracle
        sorted_sids = sorted(available)
        self._oracle_species = sorted_sids[::2]
        self._non_oracle_species = sorted_sids[1::2]

        logger.info(
            "OracleKeyTree ready: %d species with paths, %d oracle, %d non-oracle",
            len(sorted_sids), len(self._oracle_species), len(self._non_oracle_species),
        )
        logger.info("Oracle species: %s", self._oracle_species)
        logger.info("Non-oracle species: %s", self._non_oracle_species)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pre_answers(self, species_id: str) -> Optional[Dict[str, str]]:
        """Return the perfect question→answer dict for *species_id*, or None."""
        if species_id not in self._oracle_species:
            return None
        return dict(self._oracle_answers.get(species_id, {}))

    @property
    def oracle_species_ids(self) -> List[str]:
        return list(self._oracle_species)

    @property
    def non_oracle_species_ids(self) -> List[str]:
        return list(self._non_oracle_species)

    @property
    def all_mapped_species_ids(self) -> List[str]:
        return sorted(self._oracle_species + self._non_oracle_species)
