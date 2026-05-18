"""
Parse key.xml into LLM-friendly text representations.

The module provides two views of the decision tree:
  1. Text tree — indented, human-readable nested structure.
  2. Decision paths — every root-to-leaf path as a flat list of Q→A steps.

Both are generated once at startup and injected into the LLM system prompt so
that the model can reason about the tree without hard-coded traversal code.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# XML → internal representation
# ---------------------------------------------------------------------------

def _edibility(attrib: Dict[str, str]) -> str:
    for k, v in attrib.items():
        if "tlighet" in k.lower():
            return v
    return "?"


def _parse_node(node: ET.Element) -> Dict[str, Any]:
    """Recursively parse an XML node into a plain dict."""
    tag = node.tag
    attrib = dict(node.attrib)
    children = [_parse_node(c) for c in node]
    return {"tag": tag, "attrib": attrib, "children": children}


def _build_text_tree(node: Dict[str, Any], depth: int = 0) -> List[str]:
    """Recursively build indented text lines from parsed node dict."""
    indent = "  " * depth
    lines: List[str] = []
    tag = node["tag"]
    attrib = node["attrib"]

    if tag == "key":
        q = attrib.get("question", "")
        lines.append(f"KEY: {q}")
        for child in node["children"]:
            lines.extend(_build_text_tree(child, depth + 1))

    elif tag == "condition":
        ans = attrib.get("answer", "")
        sub_q = attrib.get("question", "")
        lines.append(f"{indent}→ {ans}")
        if sub_q:
            lines.append(f"{indent}  Q: {sub_q}")
        for child in node["children"]:
            lines.extend(_build_text_tree(child, depth + 1))

    elif tag == "decision":
        name = attrib.get("namn", "")
        ed = _edibility(attrib)
        url = attrib.get("url", "")
        lines.append(f"{indent}★ DECISION: {name} (edibility={ed})")
        if url:
            lines.append(f"{indent}  URL: {url}")
        for child in node["children"]:
            lines.extend(_build_text_tree(child, depth + 1))

    elif tag == "mixupdecision":
        name = attrib.get("namn", "")
        note = attrib.get("skiljetecken", "")
        lines.append(f"{indent}  [Lookalike] {name}")
        if note:
            lines.append(f"{indent}    Distinguishing feature: {note}")

    return lines


def _collect_paths(
    node: Dict[str, Any],
    current_path: List[Dict[str, str]],
    all_paths: List[List[Dict[str, str]]],
) -> None:
    """Collect every root-to-leaf path as a list of {question, answer} steps."""
    tag = node["tag"]
    attrib = node["attrib"]

    if tag == "key":
        q = attrib.get("question", "")
        for child in node["children"]:
            _collect_paths(child, [{"question": q, "answer": ""}], all_paths)

    elif tag == "condition":
        ans = attrib.get("answer", "")
        sub_q = attrib.get("question", "")
        # Update the last step's answer
        if current_path:
            current_path[-1]["answer"] = ans
        # If there is a sub-question, push a new step
        if sub_q:
            current_path.append({"question": sub_q, "answer": ""})
        # Check if any child is a decision (leaf)
        has_decision = any(c["tag"] == "decision" for c in node["children"])
        if has_decision:
            for child in node["children"]:
                if child["tag"] == "decision":
                    _collect_paths(child, list(current_path), all_paths)
        else:
            for child in node["children"]:
                _collect_paths(child, list(current_path), all_paths)

    elif tag == "decision":
        name = attrib.get("namn", "")
        ed = _edibility(attrib)
        lookalikes = []
        for child in node["children"]:
            if child["tag"] == "mixupdecision":
                lookalikes.append({
                    "name": child["attrib"].get("namn", ""),
                    "distinguishing": child["attrib"].get("skiljetecken", ""),
                })
        path_copy = list(current_path)
        path_copy.append({"decision": name, "edibility": ed, "lookalikes": lookalikes})
        all_paths.append(path_copy)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class KeyTreeParser:
    """
    Loads key.xml and produces LLM-friendly text representations.

    Usage
    -----
    parser = KeyTreeParser("data/raw/key.xml")
    print(parser.text_tree)       # indented nested view
    print(parser.decision_paths)  # flat list of root-to-leaf paths
    """

    def __init__(self, xml_path: str):
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"key.xml not found at {xml_path}")

        with open(self.xml_path, encoding="utf-8") as fh:
            root = ET.fromstring(fh.read())

        self._root_dict = _parse_node(root)

    @property
    def text_tree(self) -> str:
        """Indented, human-readable tree."""
        return "\n".join(_build_text_tree(self._root_dict))

    @property
    def decision_paths(self) -> str:
        """
        Every root-to-leaf path as a numbered list of Q→A steps ending in a
        species decision.  This is the most compact form for LLM context.
        """
        paths: List[List[Dict[str, str]]] = []
        _collect_paths(self._root_dict, [], paths)
        lines: List[str] = []
        for i, path in enumerate(paths, 1):
            lines.append(f"Path {i}:")
            for step in path:
                if "decision" in step:
                    lines.append(f"  → DECISION: {step['decision']} (edibility={step['edibility']})")
                    for la in step.get("lookalikes", []):
                        lines.append(f"     [Lookalike] {la['name']}: {la['distinguishing']}")
                else:
                    lines.append(f"  Q: {step['question']}")
                    lines.append(f"  A: {step['answer']}")
            lines.append("")
        return "\n".join(lines)

    @property
    def compact_paths(self) -> str:
        """Even more compact: one line per path."""
        paths: List[List[Dict[str, str]]] = []
        _collect_paths(self._root_dict, [], paths)
        lines: List[str] = []
        for i, path in enumerate(paths, 1):
            parts = []
            for step in path:
                if "decision" in step:
                    parts.append(f"→ {step['decision']}")
                else:
                    parts.append(f"{step['answer']}")
            lines.append(f"{i}. {' | '.join(parts)}")
        return "\n".join(lines)

    def get_prompt_injection(self, max_chars: int = 12000) -> str:
        """
        Return a single string suitable for stuffing into an LLM system prompt.
        Prefers compact paths; falls back to text tree if still too long.
        """
        compact = self.compact_paths
        if len(compact) <= max_chars:
            return (
                "=== SVAMPGUIDEN DECISION TREE ===\n"
                "The following are ALL possible identification paths from root to leaf.\n"
                "Each path ends in a species decision. Use these paths to reason about\n"
                "which species matches the observed traits.\n\n"
                f"{compact}\n"
                "=== END TREE ==="
            )
        text = self.text_tree
        if len(text) <= max_chars:
            return (
                "=== SVAMPGUIDEN DECISION TREE ===\n\n"
                f"{text}\n"
                "=== END TREE ==="
            )
        # Hard truncate with a warning
        truncated = text[:max_chars]
        return (
            "=== SVAMPGUIDEN DECISION TREE (TRUNCATED) ===\n\n"
            f"{truncated}\n... [truncated]\n"
            "=== END TREE ==="
        )
