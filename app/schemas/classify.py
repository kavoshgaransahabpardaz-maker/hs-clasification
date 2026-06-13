"""
Public API contract for /v1/classify.

These shapes are stable across versions — the calling product depends on them,
never on the DB schema or internal models.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    code: str
    description: str
    score: float
    supporting_ruling_id: Optional[int] = None
    supporting_ruling_ref: Optional[str] = None


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Free-text product description")
    jurisdiction: str = Field(..., description="Target market: 'EU' or 'UK'")
    max_candidates: int = Field(10, ge=1, le=50)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    request_id: int
    predicted_code: Optional[str]
    confidence: Optional[float]
    status: str  # auto_resolved | needs_review
    profile: Optional[Dict[str, Any]]
    candidates: List[Candidate]
    cited_ruling_ids: List[int]
    pipeline_version: str


class ReviewRequest(BaseModel):
    reviewed_code: str
    reviewer_note: Optional[str] = None


class ReviewResponse(BaseModel):
    request_id: int
    status: str
    reviewed_code: str
