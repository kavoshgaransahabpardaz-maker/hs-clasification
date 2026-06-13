"""
EBTI (European Binding Tariff Information) rulings ingestion.

Source: EU Open Data Portal — dataset "European Binding Tariff Information (EBTI-3)".
Reference: https://taxation-customs.ec.europa.eu/customs-4/calculation-customs-duties/
            customs-tariff/eu-customs-tariff-taric/european-binding-tariff-information-ebti_en

Each row is a BTI decision:  product_description → assigned CN8/TARIC10 code,
validity ~3 years.  These rows become the EU retrieval corpus AND (for the
reserved is_eval=True fraction) the gold evaluation labels.

Eval split: deterministic hash-based — ~15% of rows are marked is_eval=True.
This fraction is stable across re-ingestions (same ruling always lands in the
same split) and can be tuned via EBTI_EVAL_FRACTION env var.

Usage:
    python -m app.ingestion.ebti_loader
    python -m app.ingestion.ebti_loader --csv-path /path/to/ebti.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import sys
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session
from app.ingestion.base import RetryClient, parse_date
from app.models.enums import Jurisdiction, RulingSource
from app.models.ruling import Ruling

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The data.europa.eu /api/hub/search endpoint returns 400; try multiple URL formats.
_EBTI_DISCOVERY_SEARCHES: list[str] = [
    # New EU Open Data Portal search API
    "https://data.europa.eu/api/hub/search/datasets?q=EBTI+binding+tariff&limit=5",
    # Without filter parameter (old format with filter returns 400)
    "https://data.europa.eu/api/hub/search/datasets?q=EBTI+tariff+information&limit=5",
    # CKAN-compatible action API (used by some EU portals)
    "https://data.europa.eu/api/action/package_search?q=EBTI&rows=5",
]

# Fraction of rulings held out as evaluation labels (not used for retrieval).
EBTI_EVAL_FRACTION = 0.15

BULK_COMMIT_EVERY = 200


# ---------------------------------------------------------------------------
# Eval split helper
# ---------------------------------------------------------------------------


def _is_eval(reference: str, fraction: float = EBTI_EVAL_FRACTION) -> bool:
    """
    Deterministically assign this ruling to the eval set.

    Uses MD5(reference) mod 100 < fraction*100, so:
      • Re-ingesting the same dataset never changes a row's split.
      • ~fraction of rows become eval labels.
    """
    digest = int(hashlib.md5(reference.encode()).hexdigest(), 16)
    return (digest % 100) < int(fraction * 100)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def _discover_ebti_csv_url(client: RetryClient) -> str | None:
    """Query the EU Open Data Portal to find the current EBTI CSV download URL."""
    for search_url in _EBTI_DISCOVERY_SEARCHES:
        try:
            resp = client.get_json(search_url)
        except Exception as exc:
            logger.debug("EBTI discovery search failed (%s): %s", search_url, exc)
            continue

        results: list[dict[str, Any]] = (
            resp.get("result", {}).get("results", [])
            or resp.get("results", {}).get("results", [])
            or resp.get("datasets", [])
            or []
        )
        for dataset in results:
            title = (dataset.get("title") or "").lower()
            if "ebti" not in title and "binding tariff" not in title:
                continue
            for dist in dataset.get("distributions", []):
                fmt = (dist.get("format") or "").upper()
                url = dist.get("downloadURL") or dist.get("accessURL") or ""
                if fmt in ("CSV", "TEXT/CSV") and url:
                    logger.info("EBTI download URL discovered: %s", url)
                    return url

    logger.warning("Could not discover EBTI CSV URL from EU Open Data Portal")
    return None


# ---------------------------------------------------------------------------
# CSV column resolution
# ---------------------------------------------------------------------------

# EBTI CSV columns vary slightly between portal exports.
_COL_ALIASES: dict[str, list[str]] = {
    "reference": [
        "bti_application_reference",
        "bti_reference",
        "reference",
        "ruling_reference",
        "id",
    ],
    "description": [
        "product_description",
        "goods_description",
        "description",
        "description_of_goods",
    ],
    "code": [
        "goods_nomenclature_item_id",
        "nomenclature_code",       # EBTI-3 bulk export
        "cn_code",
        "taric_code",
        "commodity_code",
        "code",
        "tariff_code",
    ],
    "justification": [
        "classification_justification",  # EBTI-3 bulk export
        "justification",
        "justification_text",
        "notes",
        "reasoning",
    ],
    "valid_from": [
        "start_date_of_validity",  # EBTI-3 bulk export
        "validity_start_date",
        "date_of_issue",
        "start_date",
        "valid_from",
    ],
    "valid_to": [
        "end_date_of_validity",    # EBTI-3 bulk export
        "validity_end_date",
        "date_of_expiry",
        "end_date",
        "valid_to",
    ],
}


def _col(row: dict[str, str], field: str) -> str:
    for alias in _COL_ALIASES.get(field, []):
        if alias in row:
            return (row[alias] or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Upsert ruling
# ---------------------------------------------------------------------------


def _upsert_ruling(
    session: Session,
    *,
    reference: str,
    jurisdiction: Jurisdiction,
    product_description: str,
    assigned_code: str,
    justification: str | None,
    valid_from: date | None,
    valid_to: date | None,
    is_eval: bool,
    source: RulingSource,
) -> Ruling:
    """Insert or update a Ruling by reference."""
    existing = session.scalar(
        select(Ruling).where(
            Ruling.reference == reference,
            Ruling.jurisdiction == jurisdiction,
        )
    )
    if existing is None:
        ruling = Ruling(
            source=source,
            reference=reference,
            jurisdiction=jurisdiction,
            product_description=product_description,
            assigned_code=assigned_code,
            justification=justification,
            valid_from=valid_from,
            valid_to=valid_to,
            is_eval=is_eval,
        )
        session.add(ruling)
    else:
        existing.product_description = product_description
        existing.assigned_code = assigned_code
        if justification:
            existing.justification = justification
        existing.valid_from = valid_from
        existing.valid_to = valid_to
        # Never flip an eval row back to non-eval on refresh.
        existing.is_eval = existing.is_eval or is_eval
        ruling = existing

    session.flush()
    return ruling


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------


def _ingest_csv_rows(reader: csv.DictReader, session: Session) -> tuple[int, int]:  # type: ignore[type-arg]
    """
    Parse and upsert all EBTI ruling rows.

    Returns (total_rows, eval_rows).
    """
    # Normalise headers to lowercase so aliases match regardless of export casing.
    if reader.fieldnames:
        reader.fieldnames = [f.lower().strip() for f in reader.fieldnames]

    total = 0
    eval_count = 0

    for row in reader:
        reference = _col(row, "reference")
        description = _col(row, "description")
        code = _col(row, "code").rstrip("*")  # EBTI pads short codes with asterisks

        if not description or not code:
            continue

        # Fall back to a synthetic reference if the dataset lacks one.
        if not reference:
            reference = f"EBTI-{code}-{hashlib.md5(description.encode()).hexdigest()[:8]}"

        is_eval = _is_eval(reference)
        if is_eval:
            eval_count += 1

        _upsert_ruling(
            session,
            reference=reference,
            jurisdiction=Jurisdiction.EU,
            product_description=description,
            assigned_code=code,
            justification=_col(row, "justification") or None,
            valid_from=parse_date(_col(row, "valid_from")),
            valid_to=parse_date(_col(row, "valid_to")),
            is_eval=is_eval,
            source=RulingSource.EBTI,
        )

        total += 1
        if total % BULK_COMMIT_EVERY == 0:
            session.commit()
            logger.info("EBTI — %d rows committed (%d eval)", total, eval_count)

    session.commit()
    return total, eval_count


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_ebti_ingestion(session: Session) -> tuple[int, int]:
    """
    Auto-discover and ingest the EBTI dataset.

    Returns (total_rows, eval_rows).
    Raises RuntimeError if the download fails — caller should retry or use
    run_ebti_ingestion_from_file() with a manually downloaded CSV.
    """
    with RetryClient(rate_delay=0.3) as client:
        csv_url = _discover_ebti_csv_url(client)
        if csv_url is None:
            logger.warning(
                "Could not discover EBTI CSV URL — skipping EU rulings.\n"
                "Download the dataset manually from:\n"
                "  https://data.europa.eu/data/datasets/ebti-3\n"
                "Then re-run with:\n"
                "  --ebti-csv-path /path/to/ebti.csv"
            )
            return 0, 0
        logger.info("EBTI — downloading from %s", csv_url)
        csv_text = client.get_text(csv_url)

    reader = csv.DictReader(io.StringIO(csv_text))
    total, eval_count = _ingest_csv_rows(reader, session)
    logger.info(
        "EBTI ingestion done: %d rulings (%d eval, %.0f%%)",
        total,
        eval_count,
        100 * eval_count / max(total, 1),
    )
    return total, eval_count


def run_ebti_ingestion_from_file(session: Session, csv_path: str) -> tuple[int, int]:
    """Ingest EBTI rulings from a locally downloaded CSV."""
    import pathlib

    logger.info("EBTI — ingesting from local file: %s", csv_path)
    text = pathlib.Path(csv_path).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    total, eval_count = _ingest_csv_rows(reader, session)
    logger.info("EBTI — done: %d rulings (%d eval)", total, eval_count)
    return total, eval_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="EBTI EU rulings ingestion")
    parser.add_argument("--csv-path", default=None, help="Path to a local EBTI CSV")
    args = parser.parse_args()

    with get_session() as session:
        if args.csv_path:
            run_ebti_ingestion_from_file(session, args.csv_path)
        else:
            run_ebti_ingestion(session)


if __name__ == "__main__":
    _cli()
    sys.exit(0)
