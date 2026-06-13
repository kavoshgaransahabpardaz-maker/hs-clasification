"""
Pipeline orchestrator — ties together all six pipeline steps and persists
each classification as a ClassificationRequest row (audit trail).

Pipeline steps (spec section 5):
  1. extraction   (LLM → profile)                       M4 ✓
  2. retrieval    (pgvector cosine → top-K candidates)  M4 ✓
  3. rules        (legal note exclusions + GRI)          M5 ✓
  4. validation   (code validity, 24h live cache)        M5 ✓
  5. ranking      (max-sim + support → one score/code)  M5 ✓
  6. confidence   (raw score → status; calibrated M6)   M5 ✓

The pipeline is synchronous to match SQLAlchemy sync sessions and to keep
loaders / eval scripts simple.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion.embedder import Embedder
from app.models.classification_request import ClassificationRequest
from app.models.enums import RequestStatus
from app.pipeline.confidence import compute_confidence
from app.pipeline.extraction import ProductProfile, extract_profile
from app.pipeline.ranking import RankedCode, rank_candidates, ranked_to_api_candidates
from app.pipeline.retrieval import CandidateResult, retrieve_candidates
from app.pipeline.rules import apply_rules
from app.pipeline.validation import validate_candidates
from app.schemas.classify import Candidate, ClassifyResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intermediate pipeline result (internal; not part of the public API)
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    profile: ProductProfile
    # Raw retrieval hits (CandidateResult list — one row per ruling)
    raw_candidates: list[CandidateResult]
    # Ranked, deduplicated by code (RankedCode list — one row per code)
    ranked: list[RankedCode]
    predicted_code: str | None = None
    confidence: float | None = None
    status: str = RequestStatus.NEEDS_REVIEW
    cited_ruling_ids: list[int] = field(default_factory=list)
    gri_explanation: str | None = None

    # Backwards-compat alias used by eval harness and M4 tests
    @property
    def candidates(self) -> list[CandidateResult]:
        return self.raw_candidates


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    text: str,
    jurisdiction: str,
    session: Session,
    embedder: Embedder | None = None,
    max_candidates: int | None = None,
    confidence_threshold: float | None = None,
) -> PipelineResult:
    """
    Run the full six-step classification pipeline.

    Does NOT persist — call persist_request() separately.
    """
    if max_candidates is None:
        max_candidates = settings.retrieval_top_k
    if confidence_threshold is None:
        confidence_threshold = settings.confidence_threshold

    # ── Step 1: Extraction ──────────────────────────────────────────────────
    profile = extract_profile(text)
    query = profile.query or text[:300]

    # ── Step 2: Retrieval ───────────────────────────────────────────────────
    raw_candidates = retrieve_candidates(
        query=query,
        jurisdiction=jurisdiction,
        session=session,
        embedder=embedder,
        top_k=max_candidates,
    )

    if not raw_candidates:
        logger.info("No candidates retrieved for jurisdiction=%s — returning empty result", jurisdiction)
        return PipelineResult(
            profile=profile,
            raw_candidates=[],
            ranked=[],
            predicted_code=None,
            confidence=None,
            status=RequestStatus.NEEDS_REVIEW,
        )

    # ── Step 3: Rules (legal note exclusions + GRI) ─────────────────────────
    pruned_candidates, gri_explanation = apply_rules(
        profile=profile,
        candidates=raw_candidates,
        jurisdiction=jurisdiction,
        session=session,
    )

    # ── Step 4: Validation (confirm codes are currently valid) ───────────────
    valid_candidates = validate_candidates(
        candidates=pruned_candidates,
        jurisdiction=jurisdiction,
        session=session,
    )

    # ── Step 5: Ranking (one score per code) ────────────────────────────────
    ranked = rank_candidates(valid_candidates)

    # ── Step 6: Confidence ──────────────────────────────────────────────────
    confidence, status = compute_confidence(ranked, threshold=confidence_threshold)

    predicted_code = ranked[0].code if ranked else None

    # Collect all ruling IDs cited (from all codes, not just top-1).
    all_cited_ids: list[int] = []
    for rc in ranked:
        all_cited_ids.extend(rc.supporting_ruling_ids)

    return PipelineResult(
        profile=profile,
        raw_candidates=raw_candidates,
        ranked=ranked,
        predicted_code=predicted_code,
        confidence=confidence,
        status=status,
        cited_ruling_ids=all_cited_ids,
        gri_explanation=gri_explanation,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_request(
    text: str,
    jurisdiction: str,
    result: PipelineResult,
    session: Session,
) -> ClassificationRequest:
    """Write a ClassificationRequest row for the audit trail."""
    row = ClassificationRequest(
        input_text=text,
        target_jurisdiction=jurisdiction,
        profile=result.profile.model_dump(),
        predicted_code=result.predicted_code,
        confidence=result.confidence,
        status=result.status,
        candidates=ranked_to_api_candidates(result.ranked),
        cited_ruling_ids=result.cited_ruling_ids,
        pipeline_version=settings.pipeline_version,
    )
    session.add(row)
    session.flush()
    return row


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------


def build_response(
    request_row: ClassificationRequest, result: PipelineResult
) -> ClassifyResponse:
    """Convert the internal PipelineResult + DB row into the public API shape."""
    from app.schemas.classify import Candidate

    candidates_out = [
        Candidate(
            code=rc.code,
            description=rc.best_description,
            score=rc.score,
            supporting_ruling_id=rc.supporting_ruling_ids[0] if rc.supporting_ruling_ids else None,
            supporting_ruling_ref=rc.supporting_ruling_refs[0] if rc.supporting_ruling_refs else None,
        )
        for rc in result.ranked
    ]

    return ClassifyResponse(
        request_id=request_row.id,
        predicted_code=result.predicted_code,
        confidence=result.confidence,
        status=result.status,
        profile=result.profile.model_dump(),
        candidates=candidates_out,
        cited_ruling_ids=result.cited_ruling_ids,
        pipeline_version=settings.pipeline_version,
    )
