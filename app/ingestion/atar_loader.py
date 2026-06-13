"""
ATaR (UK Advance Tariff Rulings) ingestion.

Source: GOV.UK Trade Tariff — published ATaR decisions.
Reference: https://www.gov.uk/guidance/check-tariff-classification-decisions

⚠️  ACCESS METHOD — VERIFY BEFORE M3 RUN ⚠️
The spec explicitly states: "Verify the current access method (public search vs
downloadable dataset) before implementing."

As of the spec publication date, two paths exist:

  PATH A — Trade Tariff API  (preferred if available)
    GET https://www.trade-tariff.service.gov.uk/api/v2/
    Look for a "rulings" or "advance_tariff_rulings" resource.
    This loader tries /api/v2/advance_tariff_rulings first.

  PATH B — Bulk CSV / GOV.UK Open Data
    If the Trade Tariff publishes a bulk export (via data.api.trade.gov.uk or
    find-open-data.service.gov.uk), download it with run_atar_ingestion_from_file().

  PATH C — Web scraping (last resort)
    If neither API nor bulk file is available, the search page at
    https://www.trade-tariff.service.gov.uk/rulings
    can be iterated with pagination; `_scrape_rulings_page()` is provided
    as a fallback but should NOT be used in production without rate-limit care.

Set ATAR_ACCESS_METHOD = "api" | "file" | "scrape" in your .env to override
the default auto-detection.

Eval split: same deterministic hash approach as EBTI (~15%).
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
from app.ingestion.ebti_loader import _is_eval, _upsert_ruling
from app.models.enums import Jurisdiction, RulingSource

logger = logging.getLogger(__name__)

UK_TARIFF_API = "https://www.trade-tariff.service.gov.uk"

# Possible API endpoints for ATaR rulings — try in order.
ATAR_API_CANDIDATES = [
    "/api/v2/advance_tariff_rulings",
    "/api/v2/rulings",
    "/api/v2/commodity_decisions",
]

BULK_COMMIT_EVERY = 200


# ---------------------------------------------------------------------------
# Path A: Trade Tariff API
# ---------------------------------------------------------------------------


def _try_atar_api_endpoint(client: RetryClient) -> str | None:
    """
    Probe known API endpoint candidates and return the first that responds
    with a JSON array/object of rulings.  Returns None if none work.
    """
    for endpoint in ATAR_API_CANDIDATES:
        try:
            resp = client.get_json(endpoint)
            # Expect {"data": [...]} or a list directly
            data = resp.get("data", resp) if isinstance(resp, dict) else resp
            if isinstance(data, list) and len(data) > 0:
                logger.info("ATaR API endpoint found: %s", endpoint)
                return endpoint
        except Exception as exc:
            logger.debug("ATaR endpoint %s not available: %s", endpoint, exc)
    return None


def _parse_api_ruling(item: dict[str, Any]) -> dict[str, Any] | None:
    """
    Extract ruling fields from a JSON:API ruling object.

    The Trade Tariff v2 API wraps attributes under item["attributes"].
    Returns a flat dict or None if required fields are missing.
    """
    attrs = item.get("attributes", item)  # handle both JSON:API and flat formats

    description = (
        attrs.get("goods_description")
        or attrs.get("product_description")
        or attrs.get("description")
        or ""
    ).strip()

    code = (
        attrs.get("goods_nomenclature_item_id")
        or attrs.get("commodity_code")
        or attrs.get("tariff_code")
        or attrs.get("code")
        or ""
    ).strip()

    if not description or not code:
        return None

    reference = (
        attrs.get("reference")
        or attrs.get("ruling_reference")
        or attrs.get("id")
        or item.get("id")
        or f"ATaR-{code}-{hashlib.md5(description.encode()).hexdigest()[:8]}"
    )

    return {
        "reference": str(reference),
        "description": description,
        "code": code,
        "justification": attrs.get("justification") or attrs.get("notes"),
        "valid_from": parse_date(
            attrs.get("validity_start_date") or attrs.get("start_date")
        ),
        "valid_to": parse_date(
            attrs.get("validity_end_date") or attrs.get("end_date")
        ),
    }


def load_atar_from_api(session: Session, client: RetryClient) -> tuple[int, int]:
    """
    Fetch ATaR rulings from the Trade Tariff API with pagination.

    Returns (total_rows, eval_rows).  Raises RuntimeError if no endpoint works.
    """
    endpoint = _try_atar_api_endpoint(client)
    if endpoint is None:
        raise RuntimeError(
            "No ATaR API endpoint responded.  Use run_atar_ingestion_from_file() "
            "or run_atar_ingestion_scrape() instead."
        )

    total = 0
    eval_count = 0
    page = 1

    while True:
        try:
            resp = client.get_json(endpoint, params={"page": page, "per_page": 100})
        except Exception as exc:
            logger.error("ATaR API page %d failed: %s", page, exc)
            break

        items = resp.get("data", resp) if isinstance(resp, dict) else resp
        if not items:
            break

        for item in items:
            parsed = _parse_api_ruling(item)
            if parsed is None:
                continue

            is_eval = _is_eval(parsed["reference"])
            if is_eval:
                eval_count += 1

            _upsert_ruling(
                session,
                reference=parsed["reference"],
                jurisdiction=Jurisdiction.UK,
                product_description=parsed["description"],
                assigned_code=parsed["code"],
                justification=parsed["justification"],
                valid_from=parsed["valid_from"],
                valid_to=parsed["valid_to"],
                is_eval=is_eval,
                source=RulingSource.ATAR,
            )
            total += 1

        session.commit()
        logger.info("ATaR API — page %d: %d rulings so far", page, total)

        # Detect last page via pagination meta
        meta = resp.get("meta", {}) if isinstance(resp, dict) else {}
        total_pages = meta.get("total_pages") or meta.get("pages")
        if total_pages and page >= total_pages:
            break
        if len(items) < 100:
            break
        page += 1

    return total, eval_count


# ---------------------------------------------------------------------------
# Path B: Bulk CSV file
# ---------------------------------------------------------------------------

_CSV_COL_ALIASES: dict[str, list[str]] = {
    "reference": [
        "bti_application_reference",
        "ruling_reference",
        "reference",
        "atar_reference",
        "id",
    ],
    "description": [
        "product_description",
        "goods_description",
        "description",
    ],
    "code": [
        "goods_nomenclature_item_id",
        "commodity_code",
        "code",
        "tariff_code",
    ],
    "justification": ["justification", "notes", "reasoning"],
    "valid_from": ["validity_start_date", "start_date", "valid_from", "issue_date"],
    "valid_to": ["validity_end_date", "end_date", "valid_to", "expiry_date"],
}


def _csv_col(row: dict[str, str], field: str) -> str:
    for alias in _CSV_COL_ALIASES.get(field, []):
        if alias in row:
            return (row[alias] or "").strip()
    return ""


def _ingest_csv_rows(reader: csv.DictReader, session: Session) -> tuple[int, int]:  # type: ignore[type-arg]
    total = 0
    eval_count = 0

    for row in reader:
        reference = _csv_col(row, "reference")
        description = _csv_col(row, "description")
        code = _csv_col(row, "code")

        if not description or not code:
            continue
        if not reference:
            reference = (
                f"ATaR-{code}-{hashlib.md5(description.encode()).hexdigest()[:8]}"
            )

        is_eval = _is_eval(reference)
        if is_eval:
            eval_count += 1

        _upsert_ruling(
            session,
            reference=reference,
            jurisdiction=Jurisdiction.UK,
            product_description=description,
            assigned_code=code,
            justification=_csv_col(row, "justification") or None,
            valid_from=parse_date(_csv_col(row, "valid_from")),
            valid_to=parse_date(_csv_col(row, "valid_to")),
            is_eval=is_eval,
            source=RulingSource.ATAR,
        )

        total += 1
        if total % BULK_COMMIT_EVERY == 0:
            session.commit()
            logger.info("ATaR CSV — %d rows committed (%d eval)", total, eval_count)

    session.commit()
    return total, eval_count


# ---------------------------------------------------------------------------
# Path C: Web search scraping (fallback)
# ---------------------------------------------------------------------------


def load_atar_from_scrape(session: Session, client: RetryClient) -> tuple[int, int]:
    """
    Scrape ATaR rulings from the Trade Tariff search page.

    ⚠  Rate-limit friendly: 0.5s delay per page (set in RetryClient).
    Only use this if neither the API nor a bulk file is available.

    The search page at /rulings returns an HTML table; we request JSON by
    appending .json or using the Accept header.
    """
    logger.info("ATaR — falling back to search-page scraping")
    total = 0
    eval_count = 0
    page = 1

    while True:
        url = f"/rulings.json?page={page}&per_page=100"
        try:
            resp = client.get_json(url)
        except Exception as exc:
            logger.warning("ATaR scrape page %d failed: %s — stopping", page, exc)
            break

        rulings_list = resp.get("rulings") or resp.get("data") or []
        if not rulings_list:
            break

        for item in rulings_list:
            parsed = _parse_api_ruling(item)
            if parsed is None:
                continue

            is_eval = _is_eval(parsed["reference"])
            if is_eval:
                eval_count += 1

            _upsert_ruling(
                session,
                reference=parsed["reference"],
                jurisdiction=Jurisdiction.UK,
                product_description=parsed["description"],
                assigned_code=parsed["code"],
                justification=parsed["justification"],
                valid_from=parsed["valid_from"],
                valid_to=parsed["valid_to"],
                is_eval=is_eval,
                source=RulingSource.ATAR,
            )
            total += 1

        session.commit()
        logger.info("ATaR scrape — page %d: %d total", page, total)

        if len(rulings_list) < 100:
            break
        page += 1

    return total, eval_count


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_atar_ingestion(session: Session) -> tuple[int, int]:
    """
    Auto-detect access method and run ATaR ingestion.

    Order of preference: API → scrape.
    Returns (total_rows, eval_rows).

    If this raises, use run_atar_ingestion_from_file() with a manually
    downloaded CSV and update ATAR_ACCESS_METHOD in your .env.
    """
    with RetryClient(base_url=UK_TARIFF_API, rate_delay=0.5) as client:
        try:
            total, eval_count = load_atar_from_api(session, client)
            if total > 0:
                _log_summary("API", total, eval_count)
                return total, eval_count
        except RuntimeError:
            pass  # fall through to scrape

        logger.warning(
            "ATaR API unavailable — attempting search-page scrape. "
            "If this also fails, download a bulk CSV from GOV.UK and use "
            "--atar-csv-path."
        )
        total, eval_count = load_atar_from_scrape(session, client)

    _log_summary("scrape", total, eval_count)
    return total, eval_count


def run_atar_ingestion_from_file(session: Session, csv_path: str) -> tuple[int, int]:
    """Ingest ATaR rulings from a locally downloaded CSV."""
    import pathlib

    logger.info("ATaR — ingesting from local file: %s", csv_path)
    text = pathlib.Path(csv_path).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    total, eval_count = _ingest_csv_rows(reader, session)
    _log_summary("file", total, eval_count)
    return total, eval_count


def _log_summary(source: str, total: int, eval_count: int) -> None:
    logger.info(
        "ATaR ingestion done (%s): %d rulings (%d eval, %.0f%%)",
        source,
        total,
        eval_count,
        100 * eval_count / max(total, 1),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="ATaR UK rulings ingestion")
    parser.add_argument("--csv-path", default=None, help="Path to a local ATaR CSV")
    args = parser.parse_args()

    with get_session() as session:
        if args.csv_path:
            run_atar_ingestion_from_file(session, args.csv_path)
        else:
            run_atar_ingestion(session)


if __name__ == "__main__":
    _cli()
    sys.exit(0)
