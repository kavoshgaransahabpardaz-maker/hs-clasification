"""
Shared utilities for all ingestion loaders.

  • RetryClient   — httpx wrapper with exponential back-off.
  • upsert_node   — insert-or-update a NomenclatureNode row.
  • upsert_note   — insert-or-update a LegalNote row.
  • parse_date    — tolerant ISO date parser used across loaders.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.legal_note import LegalNote
from app.models.nomenclature_node import NomenclatureNode
from app.models.enums import (
    Jurisdiction,
    NomenclatureLevel,
    NoteScope,
    NoteType,
)

logger = logging.getLogger(__name__)

# How long to wait between API calls (seconds) — be polite to public endpoints.
DEFAULT_RATE_DELAY: float = 0.3


class RetryClient:
    """
    Thin httpx.Client wrapper that retries on 429 / 5xx with exponential
    back-off.  Use as a context manager.
    """

    def __init__(
        self,
        base_url: str = "",
        rate_delay: float = DEFAULT_RATE_DELAY,
        max_retries: int = 4,
        timeout: float = 30.0,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Accept": "application/json"},
            follow_redirects=True,
        )
        self.rate_delay = rate_delay
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_delay)
                resp = self._client.get(url, **kwargs)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", backoff * 2))
                    logger.warning("Rate limited on %s; sleeping %.1fs", url, retry_after)
                    time.sleep(retry_after)
                    backoff *= 2
                    continue
                if resp.status_code >= 500:
                    logger.warning(
                        "HTTP %s on %s (attempt %d)", resp.status_code, url, attempt + 1
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                return resp
            except httpx.TransportError as exc:
                last_exc = exc
                logger.warning("Transport error on %s: %s (attempt %d)", url, exc, attempt + 1)
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(
            f"Failed to GET {url} after {self.max_retries} attempts"
        ) from last_exc

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.get(url, **kwargs).json()

    def get_text(self, url: str, **kwargs: Any) -> str:
        return self.get(url, **kwargs).text

    # ------------------------------------------------------------------
    def __enter__(self) -> "RetryClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self._client.close()

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------


def upsert_node(
    session: Session,
    *,
    code: str,
    level: NomenclatureLevel,
    jurisdiction: Jurisdiction,
    description: str,
    path: str,
    parent_id: Optional[int] = None,
    valid_from: Optional[date] = None,
    valid_to: Optional[date] = None,
) -> NomenclatureNode:
    """
    Insert or update a NomenclatureNode.  Natural key = (jurisdiction, code).

    Returns the persisted node (with id populated after flush).
    """
    node = session.scalar(
        select(NomenclatureNode).where(
            NomenclatureNode.code == code,
            NomenclatureNode.jurisdiction == jurisdiction,
        )
    )
    if node is None:
        node = NomenclatureNode(
            code=code,
            level=level,
            jurisdiction=jurisdiction,
            description=description,
            path=path,
            parent_id=parent_id,
            valid_from=valid_from,
            valid_to=valid_to,
        )
        session.add(node)
    else:
        node.description = description
        node.level = level
        node.path = path
        if parent_id is not None:
            node.parent_id = parent_id
        if valid_from is not None:
            node.valid_from = valid_from
        if valid_to is not None:
            node.valid_to = valid_to

    session.flush()  # populate node.id for child path computation
    return node


def upsert_note(
    session: Session,
    *,
    jurisdiction: Jurisdiction,
    scope: NoteScope,
    scope_code: str,
    note_type: NoteType,
    text: str,
) -> LegalNote:
    """Insert or update a LegalNote. Natural key = (jurisdiction, scope, scope_code, note_type)."""
    note = session.scalar(
        select(LegalNote).where(
            LegalNote.jurisdiction == jurisdiction,
            LegalNote.scope == scope,
            LegalNote.scope_code == scope_code,
            LegalNote.note_type == note_type,
        )
    )
    if note is None:
        note = LegalNote(
            jurisdiction=jurisdiction,
            scope=scope,
            scope_code=scope_code,
            note_type=note_type,
            text=text,
        )
        session.add(note)
    else:
        note.text = text

    return note


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_date(value: Optional[str]) -> Optional[date]:
    """Parse an ISO date string (YYYY-MM-DD) tolerantly; return None on failure."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def code_level(code: str, jurisdiction: Jurisdiction) -> NomenclatureLevel:
    """
    Infer the NomenclatureLevel from a commodity code string.

    UK uses 10-digit codes; EU uses 8-digit (CN8) or 10-digit (TARIC10).
    Trailing zeros differentiate levels for UK.
    """
    n = len(code)
    if n <= 2:
        return NomenclatureLevel.CHAPTER
    if n == 4:
        return NomenclatureLevel.HEADING
    if n == 6:
        return NomenclatureLevel.SUBHEADING
    if n == 8:
        return NomenclatureLevel.CN8
    if n >= 10:
        # UK: last 2 digits are suffix; commodity if suffix != "00"
        if jurisdiction == Jurisdiction.UK and code[8:] != "00":
            return NomenclatureLevel.COMMODITY
        return NomenclatureLevel.COMMODITY
    return NomenclatureLevel.COMMODITY
