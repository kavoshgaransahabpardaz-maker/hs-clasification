"""
Rules layer — legal note exclusion + GRI tie-breaking.

Step 3 of the pipeline, runs AFTER retrieval and BEFORE validation/ranking.

Two sub-steps:

  A. Exclusion screening
     Load legal_note rows for the chapters of all candidate codes.
     Use a single LLM call to determine which candidates are definitively
     excluded by the notes for this product.

  B. GRI tie-break (only when multiple codes survive A)
     If ≥2 distinct codes remain, ask the LLM to apply the General Rules for
     the Interpretation of the HS (GRIs) and select one winner.

     ⚠  LLM CONSTRAINTS (guardrails from the spec):
       • The LLM MUST choose only from the provided candidate codes.
       • It MUST state which GRI article or chapter/section note it applied.
       • It MUST NOT invent any code not in the candidate list.

Both LLM calls are constrained-output JSON with a fixed schema.
On any LLM/JSON failure the candidates pass through unchanged so
downstream ranking can still produce a result.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.enums import NoteScope
from app.models.legal_note import LegalNote
from app.pipeline.extraction import ProductProfile
from app.pipeline.retrieval import CandidateResult

logger = logging.getLogger(__name__)

# Only trigger GRI tie-break when the top-2 scores are within this margin.
GRI_SCORE_MARGIN = 0.10


# ---------------------------------------------------------------------------
# Legal note loader
# ---------------------------------------------------------------------------


def _load_chapter_notes(
    chapter_codes: list[str], jurisdiction: str, session: Session
) -> list[LegalNote]:
    """Return all legal notes for the given chapter codes and jurisdiction."""
    if not chapter_codes:
        return []
    return list(
        session.scalars(
            select(LegalNote).where(
                LegalNote.jurisdiction == jurisdiction,
                LegalNote.scope == NoteScope.CHAPTER,
                LegalNote.scope_code.in_(chapter_codes),
            )
        ).all()
    )


def _load_section_notes(
    chapter_codes: list[str], jurisdiction: str, session: Session
) -> list[LegalNote]:
    """Return section notes — look up which sections the chapters belong to."""
    # For simplicity at M5, load all section notes for the jurisdiction and
    # let the LLM decide relevance.  A more targeted approach would require
    # mapping chapter → section, which is populated in M2.
    return list(
        session.scalars(
            select(LegalNote).where(
                LegalNote.jurisdiction == jurisdiction,
                LegalNote.scope == NoteScope.SECTION,
            )
        ).all()
    )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _format_candidates(candidates: list[CandidateResult]) -> str:
    seen: set[str] = set()
    lines = []
    for c in candidates:
        if c.code not in seen:
            seen.add(c.code)
            lines.append(f"  [{c.code}] {c.description[:200]}")
    return "\n".join(lines) if lines else "  (none)"


def _format_notes(notes: list[LegalNote]) -> str:
    if not notes:
        return "  (no chapter/section notes found)"
    parts = []
    for n in notes:
        prefix = f"Chapter {n.scope_code}" if n.scope == NoteScope.CHAPTER else f"Section {n.scope_code}"
        parts.append(f"  [{prefix} — {n.note_type}]\n  {n.text[:600]}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call: exclusion + GRI
# ---------------------------------------------------------------------------

_RULES_SYSTEM = """\
You are a customs classification legal expert applying WCO Harmonised System rules.

You will be given:
  1. A product description and structured profile.
  2. A list of candidate HS/CN codes with their descriptions.
  3. Relevant chapter and section legal notes.

Your task:
  A. Identify which candidate codes are DEFINITIVELY EXCLUDED by the notes for \
this specific product.
     Only exclude a code if the note clearly and unambiguously excludes this \
type of product.
     When in doubt, keep the candidate.

  B. If exactly one code survives, that is the GRI winner.
     If multiple codes survive, apply the GRI to select the most appropriate:
       GRI 1 — classify by headings/notes as written
       GRI 2 — incomplete/unfinished articles; mixtures
       GRI 3(a) — most specific description
       GRI 3(b) — essential character
       GRI 3(c) — last in tariff order
       GRI 4 — most akin to similar goods

     The gri_winner MUST be one of the surviving candidate codes.
     Do NOT invent a code that is not in the candidate list.

Respond with valid JSON only (no markdown):
{
  "excluded_codes": ["code1", "code2"],
  "gri_winner": "code_or_null",
  "gri_article": "GRI 3(a) / Note 1 to Chapter 84 / etc.",
  "reasoning": "one or two sentence explanation"
}\
"""


def _call_rules_llm(
    profile: ProductProfile,
    candidates: list[CandidateResult],
    notes: list[LegalNote],
) -> dict[str, Any]:
    """
    Single LLM call for both exclusion screening and GRI tie-break.

    Returns the parsed JSON dict, or {} on any failure.
    """
    from openai import OpenAI

    candidate_codes = list(dict.fromkeys(c.code for c in candidates))  # preserve order, unique

    user_msg = (
        f"Product description: {profile.query or 'N/A'}\n"
        f"Profile: material={profile.material!r}, function={profile.function!r}, "
        f"form_state={profile.form_state!r}, intended_use={profile.intended_use!r}\n\n"
        f"Candidate codes:\n{_format_candidates(candidates)}\n\n"
        f"Legal notes:\n{_format_notes(notes)}"
    )[:6000]  # stay within context

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _RULES_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or "{}"
        result: dict[str, Any] = json.loads(raw)

        # Validate: only allow exclusion of codes that were in our candidate list.
        excluded = [
            c for c in result.get("excluded_codes", []) if c in candidate_codes
        ]
        result["excluded_codes"] = excluded

        # Validate: gri_winner must be in the candidate list.
        winner = result.get("gri_winner")
        if winner and winner not in candidate_codes:
            logger.warning(
                "Rules LLM returned gri_winner %r not in candidates — ignoring", winner
            )
            result["gri_winner"] = None

        return result

    except json.JSONDecodeError as exc:
        logger.warning("Rules LLM: JSON parse error — %s", exc)
    except Exception as exc:
        logger.warning("Rules LLM call failed — %s", exc)

    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_rules(
    profile: ProductProfile,
    candidates: list[CandidateResult],
    jurisdiction: str,
    session: Session,
) -> tuple[list[CandidateResult], str | None]:
    """
    Apply chapter/section note exclusions and GRI tie-breaking to *candidates*.

    Returns:
        (surviving_candidates, gri_explanation)

    If all candidates are excluded (over-aggressive LLM), returns the original
    list unchanged (fail-safe).  The gri_explanation is logged + stored on the
    request for audit.
    """
    if not candidates:
        return candidates, None

    # Collect unique chapter codes from all candidates (first 2 digits).
    chapter_codes = list(dict.fromkeys(c.code[:2] for c in candidates))

    # Load legal notes.
    notes = _load_chapter_notes(chapter_codes, jurisdiction, session)
    # Section notes are expensive to load if there are many; only load if there
    # are actual chapter notes suggesting cross-chapter ambiguity.
    if len(chapter_codes) > 1:
        notes += _load_section_notes(chapter_codes, jurisdiction, session)

    if not notes:
        logger.debug("No legal notes found for chapters %s — skipping rules step", chapter_codes)
        return candidates, None

    # LLM call for exclusion + GRI.
    llm_result = _call_rules_llm(profile, candidates, notes)

    excluded_codes: set[str] = set(llm_result.get("excluded_codes", []))
    gri_winner: str | None = llm_result.get("gri_winner")
    reasoning: str = llm_result.get("reasoning", "")
    gri_article: str = llm_result.get("gri_article", "")

    explanation = (
        f"[{gri_article}] {reasoning}".strip() if (gri_article or reasoning) else None
    )

    # Filter out excluded candidates.
    surviving = [c for c in candidates if c.code not in excluded_codes]

    # Fail-safe: never exclude everything.
    if not surviving:
        logger.warning(
            "Rules layer excluded ALL candidates for chapters %s — reverting", chapter_codes
        )
        return candidates, explanation

    # If the LLM identified a GRI winner, surface it by re-ordering so it appears first.
    if gri_winner:
        winner_candidates = [c for c in surviving if c.code == gri_winner]
        other_candidates = [c for c in surviving if c.code != gri_winner]
        surviving = winner_candidates + other_candidates

    logger.debug(
        "Rules: %d→%d candidates; excluded=%s; gri_winner=%s",
        len(candidates),
        len(surviving),
        excluded_codes or "none",
        gri_winner or "none",
    )

    return surviving, explanation
