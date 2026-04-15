"""Pydantic request/response schemas for the mushroom identification API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class Step2StartRequest(BaseModel):
    session_id: Optional[str] = None
    visible_traits: Dict[str, Any]


class Step2AnswerRequest(BaseModel):
    session_id: str
    answer: str


class Step3CompareRequest(BaseModel):
    swedish_name: str
    visible_traits: Dict[str, Any]


class Step4FinalizeRequest(BaseModel):
    trait_extraction_result: Dict[str, Any]
    Species_tree_traversal_result: Dict[str, Any]
    comparison_result: Dict[str, Any]
