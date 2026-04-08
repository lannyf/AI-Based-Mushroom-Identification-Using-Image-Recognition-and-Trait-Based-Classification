"""
Step 2 — Species Tree Traversal (key.xml)

Implements the decision-tree traversal described in Sys.txt:
  1. Parse key.xml into a traversable tree.
  2. Auto-answer questions that can be resolved from Step 1 visible_traits.
  3. When information is missing, return the question and options to the caller
     so the UI can ask the user.
  4. Accept the user's answer and continue traversal until a conclusion
     (species decision) is reached.

Public API
----------
  engine = KeyTreeEngine(xml_path)

  # Start a new session using Step 1 output
  result = engine.start_session(session_id, visible_traits)

  # Provide a user answer and continue
  result = engine.answer(session_id, answer)

  # Result shape — always one of:
  #   {"status": "question",    "session_id": ..., "question": ...,
  #    "options": [...], "auto_answered": [...]}
  #
  #   {"status": "conclusion",  "session_id": ...,
  #    "species": ..., "edibility": ..., "edibility_label": ...,
  #    "url": ..., "lookalikes": [...], "path": [...]}
  #
  #   {"status": "error",       "session_id": ..., "message": ...}
"""

from __future__ import annotations

import logging
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edibility labels
# ---------------------------------------------------------------------------

_EDIBILITY: Dict[str, str] = {
    "*":   "edible",
    "+":   "inedible / eat with caution",
    "++":  "POISONOUS / deadly",
    "0":   "inedible",
    "(*)": "edible with caution",
    "?":   "unknown",
}


# ---------------------------------------------------------------------------
# Tree node types
# ---------------------------------------------------------------------------

@dataclass
class DecisionNode:
    species: str
    edibility: str
    url: str
    lookalikes: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ConditionNode:
    answer: str
    sub_question: str          # follow-up question text (may be empty)
    children: List[Any]        # List[ConditionNode | DecisionNode]


@dataclass
class QuestionNode:
    """Root of the current traversal state: a question with its branches."""
    question: str
    conditions: List[ConditionNode]


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------

def _edibility_value(attrib: Dict[str, str]) -> str:
    for key, val in attrib.items():
        if "tlighet" in key.lower():
            return val
    return "?"


def _parse_condition(node: ET.Element) -> ConditionNode:
    answer = node.attrib.get("answer", "")
    sub_question = node.attrib.get("question", "")
    children: List[Any] = []
    for child in node:
        if child.tag == "condition":
            children.append(_parse_condition(child))
        elif child.tag == "decision":
            children.append(_parse_decision(child))
    return ConditionNode(answer=answer, sub_question=sub_question, children=children)


def _parse_decision(node: ET.Element) -> DecisionNode:
    lookalikes = []
    for child in node:
        if child.tag == "mixupdecision":
            lookalikes.append({
                "name":      child.attrib.get("namn", ""),
                "edibility": _edibility_value(child.attrib),
                "note":      child.attrib.get("skiljetecken", ""),
            })
    return DecisionNode(
        species=node.attrib.get("namn", ""),
        edibility=_edibility_value(node.attrib),
        url=node.attrib.get("url", ""),
        lookalikes=lookalikes,
    )


def parse_key_xml(xml_path: str) -> QuestionNode:
    """Parse key.xml into a QuestionNode tree rooted at the first question."""
    with open(xml_path, encoding="utf-8") as fh:
        content = fh.read()
    root = ET.fromstring(content)
    question = root.attrib.get("question", "")
    conditions = [
        _parse_condition(child)
        for child in root
        if child.tag == "condition"
    ]
    return QuestionNode(question=question, conditions=conditions)


# ---------------------------------------------------------------------------
# Trait → answer auto-mapper
# ---------------------------------------------------------------------------

def _try_auto_answer(question: str, options: List[str],
                     traits: Dict[str, Any]) -> Optional[str]:
    """
    Try to derive the answer to *question* from Step 1 visible_traits.

    Returns the matching option string, or None if the question cannot be
    answered automatically.
    """
    dom   = traits.get("dominant_color", "").lower()
    sec   = traits.get("secondary_color", "").lower()
    shape = traits.get("cap_shape", "").lower()
    has_r = traits.get("has_ridges", False)
    cr    = traits.get("colour_ratios", {})
    dark  = cr.get("dark", 0)
    oy    = cr.get("orange_yellow", 0)

    # ------------------------------------------------------------------ #
    #  Q: "Hur ser svampen ut?" (underside / overall structure)           #
    # ------------------------------------------------------------------ #
    if "hur ser svampen ut" in question.lower():
        if has_r:
            target = "Undersidan har åsar eller ådror"
            if target in options:
                return target
        if dom in {"brown", "olive-brown", "tan"} and shape in {"convex", "flat"}:
            target = "Undersidan har rör"
            if target in options:
                return target
        return None  # ask user for this critical question

    # ------------------------------------------------------------------ #
    #  Q: "Vilken färg har svampen?" (whole mushroom colour)              #
    # ------------------------------------------------------------------ #
    if "vilken färg har svampen" in question.lower():
        yellow_opt   = next((o for o in options if "gul"  in o.lower()), None)
        grey_opt     = next((o for o in options if "grå"  in o.lower()), None)
        black_opt    = next((o for o in options if "svart" in o.lower()), None)
        orange_opt   = next((o for o in options if "orange" in o.lower()), None)

        if dom in {"orange", "yellow", "orange-yellow"} and oy > 0.2:
            if orange_opt and "orange fot" in orange_opt.lower():
                return orange_opt  # Rödgul trumpetsvamp
            if yellow_opt:
                return yellow_opt
        if dom in {"black", "dark", "grey"} and dark > 0.2:
            if black_opt:
                return black_opt
        if dom in {"brown", "grey", "olive-brown", "tan"}:
            if grey_opt:
                return grey_opt
        return None

    # ------------------------------------------------------------------ #
    #  Q: "Vad har hatten för färg?" (dark vs light cap)                  #
    # ------------------------------------------------------------------ #
    if "vad har hatten för färg" in question.lower():
        dark_opt  = next((o for o in options if "mörk"  in o.lower()), None)
        light_opt = next((o for o in options if "ljus"  in o.lower()), None)
        if dom in {"black", "dark", "grey", "olive-brown"}:
            return dark_opt
        if dom in {"white", "tan", "yellow", "orange-yellow"}:
            return light_opt
        return None

    # ------------------------------------------------------------------ #
    #  Q: "Vilken färg har hatten?" (specific cap colour)                 #
    # ------------------------------------------------------------------ #
    if "vilken färg har hatten" in question.lower():
        colour_map = {
            "gul":          {"yellow", "orange-yellow"},
            "gulorange":    {"orange", "orange-yellow"},
            "vinröd":       {"red"},
            "chokladbrun":  {"brown", "olive-brown"},
            "kastanjebrun": {"brown", "tan"},
            "gul till rödgul": {"yellow", "orange"},
            "brun":         {"brown", "tan", "olive-brown"},
            "orange":       {"orange", "orange-yellow"},
        }
        for opt in options:
            ol = opt.lower()
            for keyword, colours in colour_map.items():
                if keyword in ol and dom in colours:
                    return opt
        return None

    # ------------------------------------------------------------------ #
    #  Q: "Vilken färg och form har svampen?" (spine fungi)               #
    # ------------------------------------------------------------------ #
    if "vilken färg och form har svampen" in question.lower():
        if dom in {"orange", "yellow", "orange-yellow"}:
            rg_opt = next((o for o in options if "rödgul" in o.lower()
                           or "gulröd" in o.lower()), None)
            if rg_opt:
                return rg_opt
        return None

    # All other questions: cannot auto-answer from image
    return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class TraversalSession:
    session_id: str
    current: QuestionNode              # the question currently being asked
    visible_traits: Dict[str, Any]     # Step 1 output
    path: List[str] = field(default_factory=list)       # answers chosen so far
    auto_answered: List[Dict] = field(default_factory=list)  # auto-resolved Q→A


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class KeyTreeEngine:
    """
    Traversal engine for the key.xml species decision tree.

    One engine instance is created at startup and shared across all sessions.
    Session state is stored in ``_sessions`` keyed by session_id.
    """

    def __init__(self, xml_path: str) -> None:
        self.root: QuestionNode = parse_key_xml(xml_path)
        self._sessions: Dict[str, TraversalSession] = {}
        logger.info("KeyTreeEngine ready — root question: %s", self.root.question)

    # ------------------------------------------------------------------
    def start_session(
        self,
        session_id: Optional[str],
        visible_traits: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Begin a new traversal session using Step 1 visible_traits.

        Returns the first question the user needs to answer (or a conclusion
        if the tree can be traversed entirely from the image data).
        """
        sid = session_id or str(uuid.uuid4())
        session = TraversalSession(
            session_id=sid,
            current=self.root,
            visible_traits=visible_traits,
        )
        self._sessions[sid] = session
        logger.debug("New session %s", sid)
        return self._advance(session)

    # ------------------------------------------------------------------
    def answer(self, session_id: str, answer: str) -> Dict[str, Any]:
        """
        Provide the user's answer for the current question and continue.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return {"status": "error", "session_id": session_id,
                    "message": "Session not found. Call /step2/start first."}

        result = self._step(session, answer)
        if result["status"] == "conclusion":
            # Clean up session on conclusion
            del self._sessions[session_id]
        return result

    # ------------------------------------------------------------------
    def _advance(self, session: TraversalSession) -> Dict[str, Any]:
        """
        Try to auto-answer the current question; if not possible return it
        to the caller.
        """
        options = [c.answer for c in session.current.conditions]
        auto = _try_auto_answer(
            session.current.question,
            options,
            session.visible_traits,
        )
        if auto:
            logger.debug("Auto-answer: '%s' → '%s'", session.current.question, auto)
            session.auto_answered.append({
                "question": session.current.question,
                "answer":   auto,
                "source":   "image_analysis",
            })
            return self._step(session, auto)

        return {
            "status":       "question",
            "session_id":   session.session_id,
            "question":     session.current.question,
            "options":      options,
            "path":         list(session.path),
            "auto_answered": list(session.auto_answered),
        }

    # ------------------------------------------------------------------
    def _step(self, session: TraversalSession, answer: str) -> Dict[str, Any]:
        """
        Match *answer* against the current question's conditions and advance.
        """
        matched: Optional[ConditionNode] = None
        for cond in session.current.conditions:
            if cond.answer.strip().lower() == answer.strip().lower():
                matched = cond
                break

        if matched is None:
            opts = [c.answer for c in session.current.conditions]
            return {
                "status":     "error",
                "session_id": session.session_id,
                "message":    f"Answer '{answer}' does not match any option.",
                "valid_options": opts,
            }

        session.path.append(answer)

        # Check children for a decision
        decisions = [c for c in matched.children if isinstance(c, DecisionNode)]
        sub_conditions = [c for c in matched.children if isinstance(c, ConditionNode)]

        if decisions:
            dec = decisions[0]
            return {
                "status":          "conclusion",
                "session_id":      session.session_id,
                "species":         dec.species,
                "edibility":       dec.edibility,
                "edibility_label": _EDIBILITY.get(dec.edibility, dec.edibility),
                "url":             dec.url,
                "lookalikes": [
                    {
                        "name":           la["name"],
                        "edibility":      la["edibility"],
                        "edibility_label": _EDIBILITY.get(la["edibility"], la["edibility"]),
                        "distinguishing_feature": la["note"],
                    }
                    for la in dec.lookalikes
                ],
                "path":         list(session.path),
                "auto_answered": list(session.auto_answered),
            }

        if sub_conditions and matched.sub_question:
            session.current = QuestionNode(
                question=matched.sub_question,
                conditions=sub_conditions,
            )
            return self._advance(session)

        # Edge case: condition has sub-conditions but no follow-up question
        # (shouldn't happen in this XML but handle gracefully)
        if sub_conditions:
            session.current = QuestionNode(
                question=session.current.question,
                conditions=sub_conditions,
            )
            return self._advance(session)

        return {
            "status":     "error",
            "session_id": session.session_id,
            "message":    f"Tree path ended without a decision after answer '{answer}'.",
        }

    # ------------------------------------------------------------------
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return current state of a session (for debugging)."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return {
            "session_id":   session.session_id,
            "question":     session.current.question,
            "options":      [c.answer for c in session.current.conditions],
            "path":         list(session.path),
            "auto_answered": list(session.auto_answered),
        }
