"""Tree Path Validator — validates an LLM-outputted navigation path against key.xml.

The LLM produces a list of {question, answer} steps. This module checks whether
those steps form a valid root-to-leaf path in the dichotomous key and returns
the corresponding species decision.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple


def _collect_paths(element: ET.Element, current_path: List[Tuple[str, str]], all_paths: List[dict]) -> None:
    """Recursively collect all root-to-leaf paths as structured records."""
    tag = element.tag

    if tag == "key":
        q = element.get("question", "")
        for child in element:
            _collect_paths(child, [(q, "")], all_paths)

    elif tag == "condition":
        ans = element.get("answer", "")
        sub_q = element.get("question", "")

        new_path = list(current_path)
        if new_path:
            # Update the last step's answer
            last_q, _ = new_path[-1]
            new_path[-1] = (last_q, ans)

        # If there is a sub-question, push a new step
        if sub_q:
            new_path.append((sub_q, ""))

        decision_children = [c for c in element if c.tag == "decision"]
        non_decision = [c for c in element if c.tag != "decision"]

        for child in decision_children:
            _collect_paths(child, new_path, all_paths)
        for child in non_decision:
            _collect_paths(child, new_path, all_paths)

    elif tag == "decision":
        name = element.get("namn", "").strip()
        edibility = element.get("ätlighet", "").strip()
        all_paths.append({
            "steps": list(current_path),
            "decision": name,
            "edibility": edibility,
        })


class TreePathValidator:
    """Validate LLM-outputted tree navigation paths against key.xml.

    Usage
    -----
    validator = TreePathValidator("data/raw/key.xml")
    result = validator.validate([
        {"question": "Hur ser svampen ut?", "answer": "Undersidan har åsar eller ådror"},
        {"question": "Vilken färg har svampen?", "answer": "Hela svampen är gul"},
    ])
    # result == {"valid": True, "decision": "Kantarell", "edibility": "*", "matched_steps": 2}
    """

    def __init__(self, xml_path: str):
        tree = ET.parse(xml_path)
        self._root = tree.getroot()
        self._paths: List[dict] = []
        _collect_paths(self._root, [], self._paths)

    def _step_matches(self, q_llm: str, a_llm: str, q_xml: str, a_xml: str) -> bool:
        """Check if an LLM step matches an XML step."""
        q_llm = q_llm.strip().lower()
        a_llm = a_llm.strip().lower()
        q_xml = q_xml.strip().lower()
        a_xml = a_xml.strip().lower()

        # Question must match exactly or be a close substring
        q_match = q_llm == q_xml or q_llm in q_xml or q_xml in q_llm

        # Answer must match exactly or be a close substring
        a_match = a_llm == a_xml or a_llm in a_xml or a_xml in a_llm

        return q_match and a_match

    def validate(
        self,
        llm_path: List[Dict[str, str]],
    ) -> Dict[str, any]:
        """Check whether *llm_path* matches any valid root-to-leaf path.

        Args:
            llm_path: List of {"question": str, "answer": str} dicts from the LLM.

        Returns:
            Dict with keys: ``valid`` (bool), ``decision`` (str or None),
            ``edibility`` (str), ``matched_steps`` (int), ``best_path`` (list).
        """
        if not llm_path:
            return {"valid": False, "decision": None, "edibility": "", "matched_steps": 0, "best_path": []}

        # Normalize LLM path
        normalized_llm = []
        for step in llm_path:
            q = step.get("question", "").strip()
            a = step.get("answer", "").strip()
            if q and a:
                normalized_llm.append((q, a))

        if not normalized_llm:
            return {"valid": False, "decision": None, "edibility": "", "matched_steps": 0, "best_path": []}

        # Find all paths where EVERY LLM step matches sequentially
        candidates = []
        for path in self._paths:
            path_steps = path["steps"]
            if len(normalized_llm) > len(path_steps):
                continue  # LLM gave more steps than this path has

            all_match = True
            for i, (q_llm, a_llm) in enumerate(normalized_llm):
                if i >= len(path_steps):
                    all_match = False
                    break
                q_xml, a_xml = path_steps[i]
                if not self._step_matches(q_llm, a_llm, q_xml, a_xml):
                    all_match = False
                    break

            if all_match:
                candidates.append(path)

        if not candidates:
            return {"valid": False, "decision": None, "edibility": "", "matched_steps": 0, "best_path": []}

        # Pick the candidate with the most steps (most complete match)
        best = max(candidates, key=lambda p: len(p["steps"]))

        # Valid only if the LLM provided at least as many steps as the best path
        # (or the path is complete — we allow LLM to stop at the decision)
        is_complete = len(normalized_llm) >= len(best["steps"]) - 1

        return {
            "valid": is_complete,
            "decision": best["decision"] if is_complete else None,
            "edibility": best.get("edibility", "") if is_complete else "",
            "matched_steps": len(normalized_llm),
            "best_path": best["steps"],
        }

    def _walk_tree(self, steps: List[Tuple[str, str]]) -> Tuple[Optional[ET.Element], int]:
        """Walk the XML tree following *steps*, return (node_reached, matched_count).

        Starts at the root <key> element, follows child <condition> elements
        whose ``answer`` matches each step sequentially.
        """
        root = self._root
        current = root
        matched = 0

        for q_llm, a_llm in steps:
            found = False
            for child in current:
                if child.tag != "condition":
                    continue
                ans = child.get("answer", "").strip().lower()
                a_norm = a_llm.strip().lower()
                if ans == a_norm or ans in a_norm or a_norm in ans:
                    current = child
                    matched += 1
                    found = True
                    break
            if not found:
                break

        return current, matched

    def _get_question_and_options(self, element: ET.Element) -> Tuple[str, List[str]]:
        """Extract the next question and available answer options from an element."""
        question = element.get("question", "").strip()
        options = []
        for child in element:
            if child.tag == "condition":
                ans = child.get("answer", "").strip()
                if ans:
                    options.append(ans)
        return question, options

    def get_partial_context(self, llm_path: List[Dict[str, str]]) -> Dict[str, any]:
        """Return partial tree context for an incomplete LLM path.

        Walks the tree as far as the LLM's answers match, then returns:
        - partial_path: matched Q/A steps
        - question: next question at the stuck node
        - options: available answers at the stuck node
        - depth: how many steps were matched
        """
        if not llm_path:
            return {"partial_path": [], "question": "", "options": [], "depth": 0}

        steps = []
        for step in llm_path:
            q = step.get("question", "").strip()
            a = step.get("answer", "").strip()
            if q and a:
                steps.append((q, a))

        node, matched = self._walk_tree(steps)
        partial = steps[:matched]

        if matched == 0:
            # Didn't match any step — return root question
            root_q = self._root.get("question", "").strip()
            root_opts = []
            for child in self._root:
                if child.tag == "condition":
                    ans = child.get("answer", "").strip()
                    if ans:
                        root_opts.append(ans)
            return {"partial_path": [], "question": root_q, "options": root_opts, "depth": 0}

        question, options = self._get_question_and_options(node)
        return {
            "partial_path": [{"question": q, "answer": a} for q, a in partial],
            "question": question,
            "options": options,
            "depth": matched,
        }

    def get_all_paths(self) -> List[dict]:
        """Return all parsed paths for debugging or prompt injection."""
        return list(self._paths)
