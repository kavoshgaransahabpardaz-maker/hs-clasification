"""
EU TARIC nomenclature ingestion.

Source: EU Customs Tariff (TARIC) open dataset on data.europa.eu.

The loader performs two steps:
  1.  Discovery  — queries the EU Open Data Portal SPARQL endpoint to find
                   the current download URL for the TARIC nomenclature file.
  2.  Ingest     — downloads the CSV/XML and upserts nomenclature_node +
                   legal_note rows (CN8 and TARIC10).

Note: the EU has no official "classify from description" endpoint, so
candidate generation for EU relies entirely on your own pgvector retrieval.

Usage:
    python -m app.ingestion.eu_loader
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from typing import Any

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EU_DATA_PORTAL = "https://data.europa.eu"

# Discovery searches tried in order.  The Publications Office publishes the
# Combined Nomenclature (CN) dataset which contains all commodity codes.
_DISCOVERY_SEARCHES: list[str] = [
    # Primary: search for Combined Nomenclature by Publications Office
    (
        "https://data.europa.eu/api/hub/search/datasets"
        "?q=combined+nomenclature+CN&filter=publisher:publications-office-of-the-eu&limit=5"
    ),
    # Secondary: broader search for the annual CN regulation dataset
    (
        "https://data.europa.eu/api/hub/search/datasets"
        "?q=combined+nomenclature+customs+tariff&limit=5"
    ),
    # Tertiary: TARIC keyword
    (
        "https://data.europa.eu/api/hub/search/datasets"
        "?q=TARIC+nomenclature+commodity&filter=publisher:publications-office-of-the-eu&limit=5"
    ),
]

# The old DDS2 portal (ec.europa.eu/taxation_customs/dds2) was decommissioned.
# If auto-discovery fails, download the CN CSV manually from:
#   https://taxation-customs.ec.europa.eu/customs-4/calculation-customs-duties/customs-tariff/eu-customs-tariff-taric_en
# and pass it with --eu-csv-path  (or docker compose --profile ingest with a volume mount).
TARIC_MANUAL_DOWNLOAD_GUIDANCE = (
    "https://taxation-customs.ec.europa.eu/customs-4/calculation-customs-duties"
    "/customs-tariff/eu-customs-tariff-taric_en"
)

BULK_COMMIT_EVERY = 500


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def _discover_taric_download_url(client: RetryClient) -> str | None:
    """
    Query the EU Open Data Portal API to find the current CN/TARIC CSV download.

    Tries multiple search queries in order.  Returns the first distribution URL
    whose format is CSV/text, or None if all strategies fail.
    """
    for search_url in _DISCOVERY_SEARCHES:
        try:
            resp = client.get_json(search_url)
        except Exception as exc:
            logger.debug("EU discovery search failed (%s): %s", search_url, exc)
            continue

        results: list[dict[str, Any]] = (
            resp.get("result", {}).get("results", [])
            or resp.get("datasets", [])
            or []
        )

        for dataset in results:
            distributions = dataset.get("distributions", []) or dataset.get(
                "result", {}
            ).get("distributions", [])
            for dist in distributions:
                fmt = (dist.get("format") or "").upper()
                url = dist.get("downloadURL") or dist.get("accessURL") or ""
                if fmt in ("CSV", "TEXT/CSV", "CSV/ZIP") and url:
                    logger.info("EU CN/TARIC download URL discovered: %s", url)
                    return url

    logger.warning("Could not discover EU CN/TARIC CSV download URL from portal")
    return None


# ---------------------------------------------------------------------------
# CN/TARIC CSV parsing
# ---------------------------------------------------------------------------

# Expected column names in the TARIC CSV export.  The actual names vary
# slightly between portal exports; we try several aliases.
_COL_ALIASES: dict[str, list[str]] = {
    "code": [
        "goods_nomenclature_item_id",
        "cn_code",
        "taric_code",
        "commodity_code",
        "code",
        "nomenclature_code",
    ],
    "description": [
        "goods_nomenclature_description_text",
        "description",
        "description_text",
        "product_description",
    ],
    "parent": [
        "parent_goods_nomenclature_item_id",
        "parent_code",
        "parent",
    ],
    "valid_from": ["validity_start_date", "start_date", "valid_from"],
    "valid_to": ["validity_end_date", "end_date", "valid_to"],
    "level": ["number_indents", "indent", "level", "hierarchy_level"],
}


def _resolve_column(row: dict[str, str], field: str) -> str:
    for alias in _COL_ALIASES.get(field, []):
        if alias in row:
            return row[alias]
    return ""


def _cn_level(code: str) -> NomenclatureLevel:
    """Infer EU NomenclatureLevel from code length."""
    n = len(code.rstrip("0") or "0")
    if n <= 2:
        return NomenclatureLevel.CHAPTER
    if n <= 4:
        return NomenclatureLevel.HEADING
    if n <= 6:
        return NomenclatureLevel.SUBHEADING
    if n <= 8:
        return NomenclatureLevel.CN8
    return NomenclatureLevel.COMMODITY  # TARIC10


def _cn_path(code: str) -> str:
    """
    Build materialized path for an EU CN/TARIC code.

    Example: "0702000007" → "07.0702.070200.07020000.0702000007"
    """
    parts: list[str] = []
    for length in (2, 4, 6, 8, 10):
        segment = code[:length]
        # Skip if the segment is all zeros beyond chapter level
        if length > 2 and segment.lstrip("0") == "":
            break
        # Avoid repeating the same value (e.g. code="01" at length 2 and 4)
        if not parts or parts[-1] != segment:
            parts.append(segment)
        if len(code) <= length:
            break
    return ".".join(parts)


def _ingest_csv_rows(reader: csv.DictReader, session: Session) -> int:  # type: ignore[type-arg]
    count = 0
    for row in reader:
        code = _resolve_column(row, "code").strip()
        if not code or len(code) < 2:
            continue

        description = _resolve_column(row, "description").strip()
        valid_from = parse_date(_resolve_column(row, "valid_from"))
        valid_to = parse_date(_resolve_column(row, "valid_to"))
        level = _cn_level(code)
        path = _cn_path(code)

        upsert_node(
            session,
            code=code,
            level=level,
            jurisdiction=Jurisdiction.EU,
            description=description,
            path=path,
            valid_from=valid_from,
            valid_to=valid_to,
        )

        count += 1
        if count % BULK_COMMIT_EVERY == 0:
            session.commit()
            logger.info("EU TARIC CSV — %d rows committed", count)

    session.commit()
    return count


# ---------------------------------------------------------------------------
# Section / chapter notes for EU
# ---------------------------------------------------------------------------

# The EU Combined Nomenclature (CN) section and chapter notes are published
# annually as part of the Official Journal (EUR-Lex).  The TARIC export
# sometimes includes them; if not, we fetch the summary page from DG TAXUD.

EU_CN_NOTES_BASE = "https://ec.europa.eu/taxation_customs/dds2/taric"


def load_eu_chapter_notes(session: Session, client: RetryClient) -> None:
    """
    Attempt to load EU chapter notes from the TARIC consultation interface.

    The endpoint at /taric/chapter_notes returns HTML/JSON depending on the
    Accept header.  We request JSON where supported.

    If the endpoint is unavailable, log a warning and continue — notes can be
    re-ingested separately without reprocessing the full commodity list.
    """
    logger.info("EU — attempting to load chapter notes from TARIC interface")
    for chapter_num in range(1, 98):
        chapter_code = str(chapter_num).zfill(2)
        url = f"{EU_CN_NOTES_BASE}/chapter_notes/{chapter_code}"
        try:
            resp = client.get(url, headers={"Accept": "application/json, text/html"})
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                data = resp.json()
                note_text = data.get("note") or data.get("text") or data.get("content", "")
            else:
                # Fall back to raw text (strip HTML tags roughly)
                import re
                note_text = re.sub(r"<[^>]+>", " ", resp.text).strip()

            if note_text and len(note_text) > 20:
                upsert_note(
                    session,
                    jurisdiction=Jurisdiction.EU,
                    scope=NoteScope.CHAPTER,
                    scope_code=chapter_code,
                    note_type=NoteType.OTHER,
                    text=note_text,
                )
                session.flush()
        except Exception as exc:
            logger.debug("No EU chapter note for chapter %s: %s", chapter_code, exc)

    session.commit()
    logger.info("EU — chapter notes load complete")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_eu_ingestion(session: Session) -> None:
    """
    Run EU TARIC nomenclature ingestion.

    Steps:
      1. Discover the current TARIC CSV download URL.
      2. Download and ingest the CSV.
      3. Load chapter/section notes.
    """
    with RetryClient(rate_delay=0.5) as client:
        # Step 1: discover download URL
        csv_url = _discover_taric_download_url(client)
        if csv_url is None:
            raise RuntimeError(
                "EU CN/TARIC CSV auto-discovery failed.\n\n"
                "The old DDS2 portal (ec.europa.eu/taxation_customs/dds2) has been "
                "decommissioned by the EU Commission.\n\n"
                "Manual steps:\n"
                f"  1. Visit: {TARIC_MANUAL_DOWNLOAD_GUIDANCE}\n"
                "  2. Download the Combined Nomenclature CSV.\n"
                "  3. Re-run ingestion with the file:\n"
                "       docker compose --profile ingest run --rm \\\n"
                "         -v /path/to/cn.csv:/data/cn.csv \\\n"
                "         ingest python -m app.ingestion --steps nomenclature "
                "--jurisdiction eu --eu-csv-path /data/cn.csv\n"
                "  Or outside Docker:\n"
                "       python -m app.ingestion --steps nomenclature "
                "--jurisdiction eu --eu-csv-path /path/to/cn.csv"
            )

        # Step 2: download and parse CSV
        logger.info("EU — downloading CN/TARIC nomenclature from %s", csv_url)
        try:
            csv_text = client.get_text(csv_url)
            reader = csv.DictReader(io.StringIO(csv_text))
            count = _ingest_csv_rows(reader, session)
            logger.info("EU — nomenclature ingestion done: %d rows", count)
        except Exception as exc:
            logger.error("EU CN/TARIC CSV download/parse failed: %s", exc)
            raise

        # Step 3: chapter notes
        load_eu_chapter_notes(session, client)

    logger.info("EU ingestion complete")


def run_eu_ingestion_from_file(session: Session, csv_path: str) -> None:
    """
    Ingest EU nomenclature from a locally downloaded CSV file.

    Use this if the automatic discovery fails:
        python -m app.ingestion.eu_loader --csv-path /path/to/taric.csv
    """
    import pathlib

    logger.info("EU — ingesting from local file: %s", csv_path)
    text = pathlib.Path(csv_path).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    count = _ingest_csv_rows(reader, session)
    logger.info("EU — local file ingestion done: %d rows", count)

    with RetryClient(rate_delay=0.5) as client:
        load_eu_chapter_notes(session, client)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="EU TARIC nomenclature ingestion")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to a locally downloaded TARIC CSV (skips auto-discovery)",
    )
    args = parser.parse_args()

    with get_session() as session:
        if args.csv_path:
            run_eu_ingestion_from_file(session, args.csv_path)
        else:
            run_eu_ingestion(session)


if __name__ == "__main__":
    _cli()
    sys.exit(0)
