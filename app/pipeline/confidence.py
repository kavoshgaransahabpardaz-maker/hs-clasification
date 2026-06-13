"""
Confidence layer — maps ranking scores to a [0, 1] confidence and status.

Step 6 of the pipeline.

M5 (this file): raw, uncalibrated confidence.
  score → sigmoid normalisation → confidence ∈ [0, 1]
  confidence ≥ threshold → auto_resolved, else needs_review

M6 (calibration.py): replaces the sigmoid with an isotonic / Platt / binned
  calibrator fitted on the gold set.  After M6, a stated 0.90 confidence means
  ~90% empirical accuracy on held-out data.

⚠  SPEC GUARDRAIL: "Do not trust the confidence threshold before calibration
   (M6)."  The status field at M5 is correct in its logic, but the threshold
   is not yet trustworthy.  The API response includes pipeline_version so
   callers can distinguish M5 from M6 behaviour.

Interface contract (stable across M5 → M6):
  compute_confidence(ranked, threshold) → (confidence: float, status: str)
  The M6 calibrator is injected by swapping _score_to_confidence; callers
  remain unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

from app.models.enums import RequestStatus
from app.pipeline.ranking import RankedCode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sigmoid normalisation (M5 raw confidence)
# ---------------------------------------------------------------------------

# Sigmoid centre and steepness — tuned so that:
#   score ≈ 0.90 → confidence ≈ 0.75  (high-similarity, multi-ruling support)
#   score ≈ 0.70 → confidence ≈ 0.50
#   score ≈ 0.50 → confidence ≈ 0.25
_SIGMOID_CENTRE: float = 0.75
_SIGMOID_STEEPNESS: float = 10.0


def _sigmoid_confidence(score: float) -> float:
    """Map a raw ranking score → confidence via sigmoid."""
    return 1.0 / (1.0 + math.exp(-_SIGMOID_STEEPNESS * (score - _SIGMOID_CENTRE)))


# The active score→confidence mapping; replaced in M6 by a fitted calibrator.
_score_to_confidence: Callable[[float], float] = _sigmoid_confidence


def set_calibrator(calibrator_fn: Callable[[float], float]) -> None:
    """
    Replace the default sigmoid with a fitted calibrator (called in M6).

    calibrator_fn: float → float, must map [0, ∞) → [0, 1].
    """
    global _score_to_confidence
    _score_to_confidence = calibrator_fn
    logger.info("Confidence calibrator updated.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_confidence(
    ranked: list[RankedCode],
    threshold: float | None = None,
) -> tuple[float | None, str]:
    """
    Derive a confidence score and classification status from the ranked list.

    Args:
        ranked:    Output of rank_candidates().  May be empty.
        threshold: Override the default from settings.  If None, uses
                   settings.confidence_threshold.

    Returns:
        (confidence, status) where:
          confidence ∈ [0.0, 1.0] or None if no candidates
          status     ∈ {auto_resolved, needs_review}

    ⚠  At M5 the confidence is not calibrated — see module docstring.
    """
    from app.config import settings

    if threshold is None:
        threshold = settings.confidence_threshold

    if not ranked:
        return None, RequestStatus.NEEDS_REVIEW

    top = ranked[0]
    confidence = round(_score_to_confidence(top.score), 4)

    # Margin check: if the top-2 scores are very close, downgrade to needs_review
    # regardless of the absolute confidence.
    if len(ranked) >= 2:
        margin = top.score - ranked[1].score
        if margin < 0.02:
            logger.debug(
                "Confidence: margin %.4f between top-2 codes (%s vs %s) is too small "
                "— forcing needs_review",
                margin,
                ranked[0].code,
                ranked[1].code,
            )
            return confidence, RequestStatus.NEEDS_REVIEW

    status = (
        RequestStatus.AUTO_RESOLVED
        if confidence >= threshold
        else RequestStatus.NEEDS_REVIEW
    )

    logger.debug(
        "Confidence: score=%.4f → conf=%.4f, status=%s (threshold=%.2f)",
        top.score,
        confidence,
        status,
        threshold,
    )

    return confidence, status
