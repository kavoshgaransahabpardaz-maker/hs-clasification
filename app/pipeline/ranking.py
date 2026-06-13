"""
Ranking layer — collapse to one score per code.

Step 5 of the pipeline, runs AFTER validation.

Multiple retrieval hits may point to the same HS code (e.g. ten EBTI rulings
all classify "fresh tomatoes" as 0702000007).  This step groups them and
computes a single composite score:

  score = max_similarity + SUPPORT_WEIGHT * log(1 + support_count)

  max_similarity   — highest cosine similarity among rulings for this code.
  support_count    — how many distinct rulings corroborate this code.
  SUPPORT_WEIGHT   — small constant (default 0.05) so support gives a modest
                     bonus without overriding similarity.

The resulting ranked list is sorted descending by score.  The top-1 entry
becomes the predicted_code.

Later (M6 self-consistency): add LLM self-consistency voting across N samples
and merge the vote distribution into the score.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any

from app.pipeline.retrieval import CandidateResult

logger = logging.getLogger(__name__)

# Weight given to the log-support bonus (keep small so similarity dominates).
SUPPORT_WEIGHT: float = 0.05


# ---------------------------------------------------------------------------
# Ranked code dataclass
# ---------------------------------------------------------------------------


@dataclass
class RankedCode:
    """One entry in the ranked candidate list — one row per distinct code."""

    code: str
    score: float
    max_similarity: float
    support_count: int
    # Best-matching ruling description for this code (highest similarity).
    best_description: str
    # All ruling IDs that support this code (cited in the audit trail).
    supporting_ruling_ids: list[int] = field(default_factory=list)
    supporting_ruling_refs: list[str | None] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank_candidates(candidates: list[CandidateResult]) -> list[RankedCode]:
    """
    Group *candidates* by code and compute one composite score per code.

    Returns a list of RankedCode sorted by score descending.
    Empty input returns empty list.
    """
    if not candidates:
        return []

    # Group by code, preserving insertion order (candidates are already sorted
    # by similarity descending from retrieval).
    groups: dict[str, list[CandidateResult]] = {}
    for c in candidates:
        groups.setdefault(c.code, []).append(c)

    ranked: list[RankedCode] = []
    for code, group in groups.items():
        max_sim = max(c.similarity for c in group)
        support_count = len(group)
        score = max_sim + SUPPORT_WEIGHT * math.log(1 + support_count)

        # Pick the ruling with the highest similarity as the representative.
        best = max(group, key=lambda c: c.similarity)

        ranked.append(
            RankedCode(
                code=code,
                score=round(score, 6),
                max_similarity=round(max_sim, 4),
                support_count=support_count,
                best_description=best.description,
                supporting_ruling_ids=[c.ruling_id for c in group],
                supporting_ruling_refs=[c.ruling_ref for c in group],
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)

    logger.debug(
        "Ranking: %d candidates → %d distinct codes; top=%s (score=%.4f)",
        len(candidates),
        len(ranked),
        ranked[0].code if ranked else "none",
        ranked[0].score if ranked else 0.0,
    )

    return ranked


def ranked_to_api_candidates(ranked: list[RankedCode]) -> list[dict[str, Any]]:
    """
    Convert a ranked list into the API candidates JSON format.

    This replaces the raw CandidateResult serialisation used in M4.
    """
    return [
        {
            "code": r.code,
            "description": r.best_description,
            "score": r.score,
            "supporting_ruling_id": r.supporting_ruling_ids[0] if r.supporting_ruling_ids else None,
            "supporting_ruling_ref": r.supporting_ruling_refs[0] if r.supporting_ruling_refs else None,
            "support_count": r.support_count,
        }
        for r in ranked
    ]
