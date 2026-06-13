"""
Validation layer — confirms the winning code is currently valid.

Step 4 of the pipeline, runs AFTER rules and BEFORE ranking.

Two check layers (in order):

  1. DB check (fast, offline)
     validate_code() from db.py — queries nomenclature_node for a currently-
     valid row.  This is always available and handles 99% of cases.

  2. Live API check (network, cached 24h)
     For UK: GET /api/v2/commodities/{10-digit} → 200 = valid, 404 = invalid.
     For EU:  no official endpoint; relies on the DB check only.
     The live check is the source of truth since the nomenclature updates daily.

Candidates whose codes fail both checks are dropped.  If ALL candidates are
invalid (should be very rare after good ingestion), the original list is
returned unchanged with a warning — the ranking step will then give a low
score, resulting in `needs_review`.

Spec guardrail: "Never emit a code that is not currently valid in the live
nomenclature."
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.db import validate_code
from app.pipeline.retrieval import CandidateResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 24-hour live-API validity cache
# Keyed by "{jurisdiction}:{code}" → (is_valid: bool, checked_at: float)
# ---------------------------------------------------------------------------

_LIVE_CACHE: dict[str, tuple[bool, float]] = {}
CACHE_TTL_SECONDS: float = 86_400  # 24 h

UK_TARIFF_API = "https://www.trade-tariff.service.gov.uk"


def _cache_get(key: str) -> bool | None:
    entry = _LIVE_CACHE.get(key)
    if entry is None:
        return None
    valid, ts = entry
    if time.monotonic() - ts > CACHE_TTL_SECONDS:
        del _LIVE_CACHE[key]
        return None
    return valid


def _cache_set(key: str, valid: bool) -> None:
    _LIVE_CACHE[key] = (valid, time.monotonic())


def clear_validation_cache() -> None:
    """Flush the live-API cache (useful in tests)."""
    _LIVE_CACHE.clear()


# ---------------------------------------------------------------------------
# Live UK validation
# ---------------------------------------------------------------------------


def _validate_uk_live(code: str) -> bool:
    """
    Check the UK Trade Tariff live API.

    Returns True if 200, False if 404.  Raises on other HTTP errors so the
    caller can fall back to the DB check.
    """
    import httpx

    url = f"{UK_TARIFF_API}/api/v2/commodities/{code}"
    try:
        resp = httpx.get(url, timeout=5.0, follow_redirects=True)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True
    except httpx.HTTPStatusError:
        raise
    except Exception as exc:
        raise RuntimeError(f"UK live validation request failed: {exc}") from exc


def _validate_live(code: str, jurisdiction: str) -> bool | None:
    """
    Attempt a live validity check with 24h caching.

    Returns True/False, or None if the check is unavailable (EU, or network
    error) so the caller can fall back to the DB result.
    """
    cache_key = f"{jurisdiction}:{code}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    result: bool | None = None
    try:
        if jurisdiction == "UK":
            result = _validate_uk_live(code)
        else:
            # EU has no official single-code validation endpoint.
            # Rely on the DB check for EU codes.
            return None
    except Exception as exc:
        logger.debug("Live validation unavailable for %s %s: %s", jurisdiction, code, exc)
        return None

    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_candidates(
    candidates: list[CandidateResult],
    jurisdiction: str,
    session: Session,
    use_live_api: bool = True,
) -> list[CandidateResult]:
    """
    Filter *candidates* to those with currently-valid codes.

    Each code is checked:
      1. DB (nomenclature_node) — fast, always available.
      2. Live API (UK only, 24h TTL) — authoritative.

    A code is considered valid if the DB OR live check passes.
    Returns the original list unchanged if ALL candidates fail (fail-safe).
    """
    valid: list[CandidateResult] = []
    invalid_codes: list[str] = []

    for candidate in candidates:
        code = candidate.code

        # 1. DB check (primary)
        db_valid = validate_code(code, jurisdiction, session)

        if db_valid:
            valid.append(candidate)
            continue

        # 2. Live check (only if DB says invalid, to resolve stale nomenclature)
        if use_live_api:
            live_valid = _validate_live(code, jurisdiction)
            if live_valid is True:
                logger.debug(
                    "Code %s not in local nomenclature_node but valid on live API — accepting",
                    code,
                )
                valid.append(candidate)
                continue
            if live_valid is False:
                invalid_codes.append(code)
                continue
            # live_valid is None (unavailable) → fall through to reject

        invalid_codes.append(code)

    if invalid_codes:
        logger.info(
            "Validation: rejected %d invalid codes: %s", len(invalid_codes), invalid_codes
        )

    if not valid:
        logger.warning(
            "Validation: ALL %d candidates failed — returning original list to avoid "
            "empty response.  Check nomenclature_node freshness.",
            len(candidates),
        )
        return candidates

    return valid
