"""
FastAPI application entry-point.

API contract (stable — versioned under /v1):
  GET  /health
  POST /v1/classify
  GET  /v1/classify/{id}
  POST /v1/classify/{id}/review   (M7)

Startup sequence:
  1. Load the fitted calibrator from disk (if present) and inject it into the
     confidence pipeline.  If no calibrator file exists, the uncalibrated
     sigmoid from M5 is used and a warning is logged.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Generator

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models.classification_request import ClassificationRequest
from app.pipeline.classifier import (
    build_response,
    persist_request,
    run_pipeline,
)
from app.schemas.classify import (
    ClassifyRequest,
    ClassifyResponse,
    ReviewRequest,
    ReviewResponse,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load calibrator at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load the M6 calibrator on startup so every request uses it.

    If data/calibrator.pkl doesn't exist yet (pre-M6 or first boot),
    the uncalibrated sigmoid from confidence.py is kept and a warning is
    logged.  Run `python -m app.pipeline.calibration` to create the file.
    """
    from app.pipeline.calibration import load_calibrator_if_exists
    from app.pipeline.confidence import set_calibrator

    cal = load_calibrator_if_exists()
    if cal is not None:
        set_calibrator(cal.predict_single)
        log.info(
            "Calibrator loaded (method=%s, n_train=%d, n_digits=%d)",
            cal.method, cal.n_train_samples, cal.n_digits,
        )
    else:
        log.warning(
            "No calibrator found — using uncalibrated sigmoid (M5 behaviour). "
            "Run `python -m app.pipeline.calibration` after M6 to fix this."
        )

    yield  # application runs here


app = FastAPI(
    title="HS Code Classification Service",
    version=settings.pipeline_version,
    lifespan=_lifespan,
    description=(
        "Classifies product descriptions into EU/UK commodity codes with "
        "calibrated confidence.  Decision support only — not a legal determination."
    ),
)


# ---------------------------------------------------------------------------
# DB session dependency
# ---------------------------------------------------------------------------


def _get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline_version": settings.pipeline_version}


@app.post("/v1/classify", response_model=ClassifyResponse, status_code=200)
def classify(
    body: ClassifyRequest,
    db: Session = Depends(_get_db),
) -> ClassifyResponse:
    """
    Classify a product description into an EU or UK commodity code.

    Returns candidates with similarity scores.  Until M6 (calibration),
    `confidence` is null and `status` is always `needs_review`.
    """
    jurisdiction = body.jurisdiction.upper()
    if jurisdiction not in ("EU", "UK"):
        raise HTTPException(status_code=422, detail="jurisdiction must be 'EU' or 'UK'")

    result = run_pipeline(
        text=body.text,
        jurisdiction=jurisdiction,
        session=db,
        max_candidates=body.max_candidates,
        confidence_threshold=body.confidence_threshold,
    )
    row = persist_request(body.text, jurisdiction, result, db)
    return build_response(row, result)


@app.get("/v1/classify/{request_id}", response_model=ClassifyResponse)
def get_classification(
    request_id: int,
    db: Session = Depends(_get_db),
) -> ClassifyResponse:
    """Re-fetch a previously submitted classification by id."""
    row = db.scalar(
        select(ClassificationRequest).where(ClassificationRequest.id == request_id)
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

    from app.schemas.classify import Candidate

    candidates = [
        Candidate(
            code=c["code"],
            description=c.get("description", ""),
            score=c.get("score", 0.0),
            supporting_ruling_id=c.get("supporting_ruling_id"),
            supporting_ruling_ref=c.get("supporting_ruling_ref"),
        )
        for c in (row.candidates or [])
    ]

    return ClassifyResponse(
        request_id=row.id,
        predicted_code=row.predicted_code,
        confidence=row.confidence,
        status=row.status,
        profile=row.profile,
        candidates=candidates,
        cited_ruling_ids=row.cited_ruling_ids or [],
        pipeline_version=row.pipeline_version,
    )


@app.post("/v1/classify/{request_id}/review", response_model=ReviewResponse)
def review(
    request_id: int,
    body: ReviewRequest,
    db: Session = Depends(_get_db),
) -> ReviewResponse:
    """
    Record a human review decision.

    M7 will also embed the reviewed ruling and insert it as an INTERNAL ruling
    so future retrieval improves.  For now, just writes reviewed_code.
    """
    from datetime import datetime, timezone

    row = db.scalar(
        select(ClassificationRequest).where(ClassificationRequest.id == request_id)
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

    row.reviewed_code = body.reviewed_code
    row.reviewer_note = body.reviewer_note
    row.status = "reviewed"
    row.reviewed_at = datetime.now(timezone.utc)

    # TODO M7: insert INTERNAL ruling, embed it, re-index

    return ReviewResponse(
        request_id=row.id,
        status=row.status,
        reviewed_code=body.reviewed_code,
    )
