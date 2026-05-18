"""Pydantic request/response schemas for the mushroom identification API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class Step2StartRequest(BaseModel):
    session_id: Optional[str] = None
    visible_traits: Dict[str, Any]
    ml_hint: Optional[Dict[str, Any]] = None
    pre_answers: Optional[Dict[str, str]] = None


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


class LLMPredictRequest(BaseModel):
    visible_traits: Dict[str, Any]


class UnifiedIdentifyRequest(BaseModel):
    """Request body for the unified identification endpoint.

    Accepts two images: an above-view photo (showing cap + stem)
    and a below-view photo (showing underside + stem).
    Both are sent as multipart/form-data, not JSON.
    """
    # This schema is used for documentation only;
    # the actual endpoint reads UploadFile fields directly.
    pass


class UnifiedIdentifyResponse(BaseModel):
    case: Dict[str, Any]
    traits: Dict[str, Any]
    cnn: Dict[str, Any]
    tree: Dict[str, Any]
    database: Dict[str, Any]
    llm: Dict[str, Any]
    agreement: str
    final_recommendation: Dict[str, Any]
    processing_time_ms: float
