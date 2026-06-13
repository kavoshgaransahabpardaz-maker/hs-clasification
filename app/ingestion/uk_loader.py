"""
UK nomenclature ingestion — two modes:

BULK (cold start, recommended first run)
  Downloads the full commodities CSV from the Department for Business & Trade
  Data API.  Gives every commodity code + description in one request.
  Licence: Open Government Licence v3.

DELTA (daily refresh)
  Walks the GOV.UK Trade Tariff Public API v2 (JSON:API format) to pick up
  section/chapter notes and any changes since the last bulk load.

Usage:
    python -m app.ingestion.uk_loader [--mode bulk|delta|full]

"full" (default) = bulk CSV first, then API delta for notes/headings.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.ingestion.base import (
    RetryClient,
    parse_date,
    upsert_node,
    upsert_note,
)
from app.models.enums import (
    Jurisdiction,
    NomenclatureLevel,
    NoteScope,
    NoteType,
)
from app.models.nomenclature_node import NomenclatureNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UK_TARIFF_API = "https://www.trade-tariff.service.gov.uk"
DBT_API_BASE = "https://data.api.trade.gov.uk/v1"
DATASET_ID = "uk-tariff-2021-01-01"

# Commit every N rows during bulk load to bound memory usage.
BULK_COMMIT_EVERY = 500


# ---------------------------------------------------------------------------
# Helper: JSON:API included-object index
# ---------------------------------------------------------------------------


def _included_index(included: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Build a (type, id) → object map for a JSON:API 'included' array."""
    return {(obj["type"], obj["id"]): obj for obj in included}


# ---------------------------------------------------------------------------
# Section notes
# ---------------------------------------------------------------------------


def load_uk_section_notes(session: Session, client: RetryClient) -> None:
    """
    Fetch all sections and persist:
      • A NomenclatureNode row (level=section) for each section.
      • A LegalNote row for sections that carry a section_note.
    """
    logger.info("UK — fetching sections list")
    resp = client.get_json("/api/v2/sections")
    sections = resp.get("data", [])

    for item in sections:
        attrs = item.get("attributes", {})
        position = str(attrs.get("position", item["id"]))
        title = attrs.get("title", "")

        # Sections don't have HS codes — use position number as the code value.
        node = upsert_node(
            session,
            code=f"S{position.zfill(2)}",  # e.g. "S01", "S16"
            level=NomenclatureLevel.SECTION,
            jurisdiction=Jurisdiction.UK,
            description=title,
            path=f"S{position.zfill(2)}",
        )

        # Fetch detail to get the section note (not in the list response).
        detail_url = f"/api/v2/sections/{item['id']}"
        try:
            detail = client.get_json(detail_url)
            detail_attrs = detail.get("data", {}).get("attributes", {})
            note_text: str | None = detail_attrs.get("section_note") or detail_attrs.get("formatted_section_note")
            if note_text and note_text.strip():
                upsert_note(
                    session,
                    jurisdiction=Jurisdiction.UK,
                    scope=NoteScope.SECTION,
                    scope_code=position.zfill(2),
                    note_type=NoteType.OTHER,
                    text=note_text.strip(),
                )
        except Exception as exc:
            logger.warning("Could not fetch section detail %s: %s", item["id"], exc)

        session.flush()

    session.commit()
    logger.info("UK — sections done (%d)", len(sections))


# ---------------------------------------------------------------------------
# Chapter notes
# ---------------------------------------------------------------------------


def load_uk_chapter_notes(session: Session, client: RetryClient) -> None:
    """
    Fetch all chapters and persist:
      • A NomenclatureNode (level=chapter).
      • A LegalNote for chapters with a chapter_note.

    Iterates chapters 01–99 directly using the canonical 2-digit chapter code
    as the URL path parameter.  This avoids the list endpoint's internal IDs
    (e.g. 29417) which are not the same as chapter numbers and cause 404s.
    """
    logger.info("UK — fetching chapters 01–99 from detail endpoint")
    count = 0
    for n in range(1, 100):
        chapter_code = str(n).zfill(2)
        try:
            resp = client.get_json(f"/api/v2/chapters/{chapter_code}")
        except Exception as exc:
            # Chapters that don't exist in the UK tariff return 404 — normal.
            logger.debug("Chapter %s not found (skipped): %s", chapter_code, exc)
            continue

        attrs = resp.get("data", {}).get("attributes", {})
        description = attrs.get("description", attrs.get("formatted_description", ""))

        upsert_node(
            session,
            code=chapter_code,
            level=NomenclatureLevel.CHAPTER,
            jurisdiction=Jurisdiction.UK,
            description=description,
            path=chapter_code,
            valid_from=parse_date(attrs.get("validity_start_date")),
            valid_to=parse_date(attrs.get("validity_end_date")),
        )

        note_text: str | None = (
            attrs.get("chapter_note") or attrs.get("formatted_chapter_note")
        )
        if note_text and note_text.strip():
            upsert_note(
                session,
                jurisdiction=Jurisdiction.UK,
                scope=NoteScope.CHAPTER,
                scope_code=chapter_code,
                note_type=NoteType.OTHER,
                text=note_text.strip(),
            )

        session.flush()
        count += 1

    session.commit()
    logger.info("UK — chapters done (%d found of 99)", count)


# ---------------------------------------------------------------------------
# Headings (API-based — used by delta mode and to fill gaps in bulk CSV)
# ---------------------------------------------------------------------------


def load_uk_heading(
    heading_code: str, session: Session, client: RetryClient
) -> None:
    """
    Fetch a single 4-digit heading and its child commodities from the API.

    The response embeds commodities in `included`, linked via relationships.
    We walk the tree recursively to build parent_id links.
    """
    try:
        resp = client.get_json(f"/api/v2/headings/{heading_code}")
    except Exception as exc:
        logger.warning("Could not fetch heading %s: %s", heading_code, exc)
        return

    data = resp.get("data", {})
    included = _included_index(resp.get("included", []))

    h_attrs = data.get("attributes", {})
    heading_desc = h_attrs.get("description", h_attrs.get("formatted_description", ""))

    chapter_code = heading_code[:2]
    chapter_node = session.scalar(
        select(NomenclatureNode).where(
            NomenclatureNode.code == chapter_code,
            NomenclatureNode.jurisdiction == Jurisdiction.UK,
        )
    )
    chapter_id = chapter_node.id if chapter_node else None

    heading_node = upsert_node(
        session,
        code=heading_code,
        level=NomenclatureLevel.HEADING,
        jurisdiction=Jurisdiction.UK,
        description=heading_desc,
        path=f"{chapter_code}.{heading_code}",
        parent_id=chapter_id,
        valid_from=parse_date(h_attrs.get("validity_start_date")),
        valid_to=parse_date(h_attrs.get("validity_end_date")),
    )

    # Walk child commodities embedded in the response
    _ingest_commodity_children(
        data.get("relationships", {}).get("children", {}).get("data", []),
        included,
        parent_node=heading_node,
        parent_path=f"{chapter_code}.{heading_code}",
        session=session,
    )

    session.flush()


def _ingest_commodity_children(
    children_refs: list[dict[str, str]],
    included: dict[tuple[str, str], dict[str, Any]],
    parent_node: Any,
    parent_path: str,
    session: Session,
) -> None:
    """Recursively ingest commodity children from a JSON:API included array."""
    for ref in children_refs:
        obj = included.get((ref["type"], ref["id"]))
        if obj is None:
            continue

        attrs = obj.get("attributes", {})
        item_id: str = attrs.get("goods_nomenclature_item_id", "")
        if not item_id:
            continue

        description = attrs.get("description", attrs.get("formatted_description", ""))
        code = item_id.rstrip("0") or item_id  # keep meaningful digits
        # Use the full 10-digit id for commodities to preserve specificity.
        code = item_id[:10]
        suffix = attrs.get("producline_suffix", "80")

        # Determine level from code length
        n = len(item_id.rstrip("0") or "00")
        if n <= 4:
            level = NomenclatureLevel.HEADING
        elif n <= 6:
            level = NomenclatureLevel.SUBHEADING
        else:
            level = NomenclatureLevel.COMMODITY

        path = f"{parent_path}.{item_id[:10]}"

        child_node = upsert_node(
            session,
            code=item_id[:10],
            level=level,
            jurisdiction=Jurisdiction.UK,
            description=description,
            path=path,
            parent_id=parent_node.id,
            valid_from=parse_date(attrs.get("validity_start_date")),
            valid_to=parse_date(attrs.get("validity_end_date")),
        )

        # Recurse into grandchildren if present
        grandchildren = obj.get("relationships", {}).get("children", {}).get("data", [])
        if grandchildren:
            _ingest_commodity_children(
                grandchildren, included, child_node, path, session
            )


# ---------------------------------------------------------------------------
# Bulk CSV load (cold start — fastest way to populate all commodity codes)
# ---------------------------------------------------------------------------


def _get_latest_dbt_version(client: RetryClient) -> str:
    """Return the latest version string from the DBT Data API."""
    url = f"{DBT_API_BASE}/datasets/{DATASET_ID}/versions"
    resp = client.get_json(url)
    versions = resp.get("versions", resp if isinstance(resp, list) else [])
    if not versions:
        raise ValueError("No versions returned by DBT API")
    # versions are typically sorted newest-first or by created_at
    latest = versions[0]
    if isinstance(latest, dict):
        return str(latest.get("version", latest.get("id", "1")))
    return str(latest)


def load_uk_bulk_csv(session: Session, client: RetryClient) -> int:
    """
    Download the full UK commodities CSV and upsert all rows.

    Returns the number of rows processed.

    CSV columns expected (Open Government Licence v3 dataset):
      goods_nomenclature_item_id, producline_suffix,
      goods_nomenclature_description_text,
      validity_start_date, validity_end_date,
      parent_goods_nomenclature_item_id
    """
    logger.info("UK — fetching latest DBT dataset version")
    try:
        version = _get_latest_dbt_version(client)
    except Exception as exc:
        logger.warning("Could not determine DBT version, using 'latest': %s", exc)
        version = "latest"

    csv_url = (
        f"{DBT_API_BASE}/datasets/{DATASET_ID}/versions/{version}"
        "/tables/commodities/data?format=csv"
    )
    logger.info("UK — downloading bulk CSV from %s", csv_url)
    csv_text = client.get_text(csv_url)

    reader = csv.DictReader(io.StringIO(csv_text))
    count = 0
    for row in reader:
        item_id: str = (
            row.get("goods_nomenclature_item_id")
            or row.get("commodity_code")
            or ""
        ).strip()
        if not item_id or len(item_id) < 4:
            continue

        description: str = (
            row.get("goods_nomenclature_description_text")
            or row.get("description")
            or ""
        ).strip()

        valid_from = parse_date(
            row.get("validity_start_date") or row.get("start_date")
        )
        valid_to = parse_date(
            row.get("validity_end_date") or row.get("end_date")
        )

        chapter_code = item_id[:2]
        # Determine level from suffix: suffix "80" = commodity, "00" = intermediate
        suffix = (row.get("producline_suffix") or "80").strip()
        if len(item_id) == 10:
            level = NomenclatureLevel.COMMODITY if suffix != "00" else NomenclatureLevel.SUBHEADING
        elif len(item_id) == 8:
            level = NomenclatureLevel.CN8
        elif len(item_id) == 6:
            level = NomenclatureLevel.SUBHEADING
        elif len(item_id) == 4:
            level = NomenclatureLevel.HEADING
        else:
            level = NomenclatureLevel.CHAPTER

        path = _build_path(item_id)

        upsert_node(
            session,
            code=item_id,
            level=level,
            jurisdiction=Jurisdiction.UK,
            description=description,
            path=path,
            valid_from=valid_from,
            valid_to=valid_to,
        )

        count += 1
        if count % BULK_COMMIT_EVERY == 0:
            session.commit()
            logger.info("UK bulk CSV — %d rows committed", count)

    session.commit()
    logger.info("UK bulk CSV — done, %d rows total", count)
    return count


def _build_path(item_id: str) -> str:
    """
    Build a dot-separated materialized path for a UK commodity code.

    Example: "0702000007" → "07.0702.070200.0702000007"
    """
    parts: list[str] = []
    if len(item_id) >= 2:
        parts.append(item_id[:2])
    if len(item_id) >= 4:
        parts.append(item_id[:4])
    if len(item_id) >= 6:
        parts.append(item_id[:6])
    if len(item_id) >= 8:
        parts.append(item_id[:8])
    if len(item_id) >= 10 and item_id not in (item_id[:8] + "00",):
        parts.append(item_id[:10])
    # deduplicate while preserving order
    seen: set[str] = set()
    unique_parts: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)
    return ".".join(unique_parts)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_uk_ingestion(session: Session, mode: str = "full") -> None:
    """
    Run UK ingestion.

    mode="bulk"  — bulk CSV only (commodity codes, no notes).
    mode="delta" — API-based sections, chapters, headings only.
    mode="full"  — bulk CSV first, then API delta (recommended).
    """
    with RetryClient(base_url=UK_TARIFF_API) as api_client:
        if mode in ("bulk", "full"):
            load_uk_bulk_csv(session, api_client)

        if mode in ("delta", "full"):
            load_uk_section_notes(session, api_client)
            load_uk_chapter_notes(session, api_client)

    logger.info("UK ingestion complete (mode=%s)", mode)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="UK nomenclature ingestion")
    parser.add_argument(
        "--mode",
        choices=["bulk", "delta", "full"],
        default="full",
        help="Ingestion mode (default: full)",
    )
    args = parser.parse_args()

    with get_session() as session:
        run_uk_ingestion(session, mode=args.mode)


if __name__ == "__main__":
    _cli()
    sys.exit(0)
