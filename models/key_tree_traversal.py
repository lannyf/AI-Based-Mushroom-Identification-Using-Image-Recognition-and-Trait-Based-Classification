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
  result = engine.start_session(session_id, visible_traits, ml_hint=None, pre_answers=None)

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

_PRECHECK_SKIP_THRESHOLD = 0.85

_TREE_EXACT_MATCHES: Dict[str, Dict[str, str]] = {
    "Chanterelle": {"tree_species": "Kantarell", "swedish_name": "Kantarell"},
    "Black Trumpet": {"tree_species": "Svart trumpetsvamp", "swedish_name": "Svart trumpetsvamp"},
    "Porcini": {"tree_species": "Stensopp (karljohanssvamp)", "swedish_name": "Karljohan"},
    "Other Boletus": {"tree_species": "Brunsopp", "swedish_name": "Brunsopp"},
}

_TREE_LOOKALIKE_ONLY: Dict[str, Dict[str, str]] = {
    "Amanita virosa": {
        "swedish_name": "Änglsvamp",
        "tree_alias": "Vit flugsvamp",
        "edibility": "++",
        "edibility_label": "POISONOUS / deadly",
        "reason": "Species appears only as a toxic lookalike in key.xml, not as a traversable decision.",
    },
    "False Chanterelle": {
        "swedish_name": "Falsk kantarell",
        "tree_alias": "Narrkantarell (falsk kantarell)",
        "edibility": "+",
        "edibility_label": "inedible / eat with caution",
        "reason": "Species appears only as a lookalike under Kantarell, not as a traversable decision.",
    },
}

_TREE_UNSUPPORTED: Dict[str, Dict[str, str]] = {
    "Fly Agaric": {
        "swedish_name": "Flugsvamp",
        "edibility": "+",
        "edibility_label": "inedible / eat with caution",
        "reason": "Species is not present in key.xml.",
    },
}


def _tree_compatibility(
    visible_traits: Dict[str, Any],
    ml_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ml_species = ""
    ml_confidence = 0.0
    if ml_hint is not None:
        ml_species = str(ml_hint.get("top_species", "")).strip()
        ml_confidence = float(ml_hint.get("confidence", 0.0) or 0.0)

    compatibility: Dict[str, Any] = {
        "ml_top_species": ml_species,
        "ml_confidence": ml_confidence,
        "tree_support": "unknown",
        "tree_policy": "full_traversal",
        "tree_species": None,
        "swedish_name": None,
        "reason": "",
    }

    if ml_species in _TREE_EXACT_MATCHES:
        info = _TREE_EXACT_MATCHES[ml_species]
        compatibility.update({
            "tree_support": "exact_decision",
            "tree_policy": "full_traversal",
            "tree_species": info["tree_species"],
            "swedish_name": info["swedish_name"],
            "reason": "Species has an exact traversable decision in key.xml.",
        })
        return compatibility

    if ml_species in _TREE_LOOKALIKE_ONLY:
        info = _TREE_LOOKALIKE_ONLY[ml_species]
        compatibility.update({
            "tree_support": "lookalike_only",
            "tree_policy": "skip_tree" if ml_confidence >= _PRECHECK_SKIP_THRESHOLD else "branch_only",
            "tree_species": info.get("tree_alias"),
            "swedish_name": info["swedish_name"],
            "edibility": info["edibility"],
            "edibility_label": info["edibility_label"],
            "reason": info["reason"],
        })
        return compatibility

    if ml_species in _TREE_UNSUPPORTED:
        info = _TREE_UNSUPPORTED[ml_species]
        compatibility.update({
            "tree_support": "unsupported",
            "tree_policy": "skip_tree" if ml_confidence >= _PRECHECK_SKIP_THRESHOLD else "branch_only",
            "tree_species": None,
            "swedish_name": info["swedish_name"],
            "edibility": info["edibility"],
            "edibility_label": info["edibility_label"],
            "reason": info["reason"],
        })
        return compatibility

    return compatibility


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

def derive_key_answers(visible_traits: Dict[str, Any]) -> Dict[str, str]:
    """
    Derive key.xml auto-answers from part-aware structured traits.
    Returns a dict {question_text: answer_text} for high-confidence mappings only.
    """
    from config import trait_config as tc

    answers: Dict[str, str] = {}
    if not tc.ENABLE_PART_AWARE_KEY_AUTOANSWERS:
        return answers

    conf = visible_traits.get("trait_confidence", {})

    # morphology_case -> root question
    if conf.get("morphology_case", 0.0) >= tc.PART_AWARE_MIN_AUTOANSWER_CONFIDENCE:
        morphology = visible_traits.get("morphology_case")
        if morphology == "coral":
            answers["Hur ser svampen ut?"] = "Den är busklik med många grenar"
        elif morphology == "puffball":
            answers["Hur ser svampen ut?"] = "Päronformad eller rund"

    # hymenophore_type -> root question
    hconf = visible_traits.get("hymenophore_confidence", conf.get("hymenophore_type", 0.0))
    if hconf >= tc.PART_AWARE_MIN_AUTOANSWER_CONFIDENCE:
        hymenophore = visible_traits.get("hymenophore_type", "unknown")
        if hymenophore == "ridges":
            answers["Hur ser svampen ut?"] = "Undersidan har åsar eller ådror"
        elif hymenophore == "gills":
            answers["Hur ser svampen ut?"] = "Undersidan har skivor"
        elif hymenophore == "pores":
            answers["Hur ser svampen ut?"] = "Undersidan har rör"
        elif hymenophore == "teeth":
            answers["Hur ser svampen ut?"] = "Undersidan har taggar"

    return answers


def _try_auto_answer(
    question: str,
    options: List[str],
    traits: Dict[str, Any],
    ml_hint: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
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
    brown = cr.get("brown", 0)
    red   = cr.get("red", 0)
    white = cr.get("white", 0)

    ml_species = ""
    ml_conf = 0.0
    if ml_hint is not None:
        ml_species = str(ml_hint.get("top_species", ""))
        ml_conf = float(ml_hint.get("confidence", 0.0) or 0.0)

    if ml_conf >= 0.75:
        ml_question_hints = {
            "Hur ser svampen ut?": {
                "Fly Agaric": "Undersidan har skivor",
                "Amanita virosa": "Undersidan har skivor",
                "False Chanterelle": "Undersidan har skivor",
                "Chanterelle": "Undersidan har åsar eller ådror",
                "Black Trumpet": "Undersidan har åsar eller ådror",
                "Porcini": "Undersidan har rör",
                "Other Boletus": "Undersidan har rör",
            },
            "Vilken färg har svampen?": {
                "Chanterelle": "Hela svampen är gul",
                "Black Trumpet": "Hela svampen är svart eller mörkgrå",
            },
            "Hur ser rören ut?": {
                "Porcini": "Rören kan lätt lossas från hatten hos utväxta exemplar",
                "Other Boletus": "Rören kan lätt lossas från hatten hos utväxta exemplar",
            },
            "Har den ring?": {
                "Porcini": "Den har ingen ring",
                "Other Boletus": "Den har ingen ring",
            },
            "Vad stämmer bäst angående utseende?": {
                "Porcini": "Kraftig fot med vitt ådernät",
                "Other Boletus": "Brun hatt och fot",
            },
        }
        hinted = ml_question_hints.get(question, {}).get(ml_species)
        if hinted in options:
            return hinted

    # Try part-aware derived answers first (Phase 4)
    derived = derive_key_answers(traits)
    if question in derived:
        ans = derived[question]
        if ans in options:
            return ans

    # ------------------------------------------------------------------ #
    #  Q: "Hur ser svampen ut?" (underside / overall structure)           #
    # ------------------------------------------------------------------ #
    if "hur ser svampen ut" in question.lower():
        ridge_target = "Undersidan har åsar eller ådror"
        pore_target = "Undersidan har rör"

        # The raw ridge detector is intentionally noisy on full-scene photos.
        # Only auto-answer when the visual profile is clearly bolete-like or
        # chanterelle/trumpet-like; otherwise let the caller ask the user.
        if pore_target in options:
            bolete_like = (
                dom in {"brown", "olive-brown", "tan"}
                and sec in {"yellow", "yellow-green", "olive-brown", "orange-yellow"}
                and brown >= 0.09
                and red < 0.25
                and white < 0.15
            )
            bolete_photo_variant = (
                dom in {"orange", "red"}
                and brown >= 0.25
                and sec in {"orange", "red", "yellow-green", "orange-yellow"}
                and white < 0.08
            )
            if bolete_like or bolete_photo_variant:
                return pore_target

        if ridge_target in options and has_r:
            chanterelle_like = (
                (dom in {"yellow", "orange", "orange-yellow"} and oy >= brown)
                or (sec in {"yellow", "orange", "orange-yellow"} and oy >= 0.16 and dark < 0.18)
                or (dom in {"black", "grey"} and dark >= 0.15)
                or (sec in {"black", "grey"} and dark >= 0.15 and brown < 0.15)
            )
            if chanterelle_like:
                return ridge_target

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
    tree_compatibility: Dict[str, Any]
    ml_hint: Optional[Dict[str, Any]] = field(default=None)
    pre_answers: Dict[str, str] = field(default_factory=dict)  # user-supplied answers
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
        ml_hint: Optional[Dict[str, Any]] = None,
        pre_answers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Begin a new traversal session using Step 1 visible_traits.

        Args:
            session_id: Optional existing session ID.
            visible_traits: Structured traits from visual_trait_extractor.extract().
            ml_hint: Optional CNN hint dict with ``top_species`` and ``confidence``.
            pre_answers: Optional user-supplied answers keyed by question text.
                         Only used when visible_traits cannot provide a conclusive
                         auto-answer.

        Returns the first question the user needs to answer (or a conclusion
        if the tree can be traversed entirely from the image data).
        """
        sid = session_id or str(uuid.uuid4())
        compatibility = _tree_compatibility(visible_traits, ml_hint)
        if compatibility["tree_policy"] == "skip_tree":
            logger.debug(
                "Skipping tree traversal for %s (%.3f): %s",
                compatibility["ml_top_species"],
                compatibility["ml_confidence"],
                compatibility["reason"],
            )
            return {
                "status": "conclusion",
                "session_id": sid,
                "species": compatibility["swedish_name"],
                "edibility": compatibility["edibility"],
                "edibility_label": compatibility["edibility_label"],
                "url": "",
                "lookalikes": [],
                "path": [],
                "auto_answered": [],
                "tree_compatibility": compatibility,
                "source": "ml_precheck",
                "message": (
                    f"Skipped species tree traversal because "
                    f"'{compatibility['ml_top_species']}' cannot be confirmed exactly by key.xml."
                ),
            }
        session = TraversalSession(
            session_id=sid,
            current=self.root,
            visible_traits=visible_traits,
            tree_compatibility=compatibility,
            ml_hint=ml_hint,
            pre_answers=pre_answers or {},
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
        Try to auto-answer the current question from visible_traits first;
        if not possible, fall back to pre_answers; if still not possible,
        return the question to the caller.
        """
        options = [c.answer for c in session.current.conditions]

        # 1. Try auto-answer from image-derived visible_traits
        auto = _try_auto_answer(
            session.current.question,
            options,
            session.visible_traits,
            session.ml_hint,
        )
        if auto:
            logger.debug("Auto-answer: '%s' → '%s'", session.current.question, auto)
            session.auto_answered.append({
                "question": session.current.question,
                "answer":   auto,
                "source":   "image_analysis",
            })
            return self._step(session, auto)

        # 2. Fall back to user-supplied pre_answers (complementary only)
        pre = session.pre_answers.get(session.current.question)
        if pre and pre in options:
            logger.debug("Pre-answer: '%s' → '%s'", session.current.question, pre)
            session.auto_answered.append({
                "question": session.current.question,
                "answer":   pre,
                "source":   "user_pre_answer",
            })
            return self._step(session, pre)

        # 3. Cannot resolve — ask the user
        return {
            "status":       "question",
            "session_id":   session.session_id,
            "question":     session.current.question,
            "options":      options,
            "path":         list(session.path),
            "auto_answered": list(session.auto_answered),
            "tree_compatibility": session.tree_compatibility,
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
                "tree_compatibility": session.tree_compatibility,
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
            "tree_compatibility": session.tree_compatibility,
        }
