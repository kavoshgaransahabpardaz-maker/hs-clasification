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
    python -m app.ingestion.eu_loader --csv-path /path/to/Nomenclature.xlsx
    python -m app.ingestion.eu_loader --csv-path /path/to/taric.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import date
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
# If auto-discovery fails, download the CN XLSX manually from:
#   https://taxation-customs.ec.europa.eu/customs-4/calculation-customs-duties/customs-tariff/eu-customs-tariff-taric_en
# and pass it with --csv-path  (or docker compose --profile ingest with a volume mount).
TARIC_MANUAL_DOWNLOAD_GUIDANCE = (
    "https://taxation-customs.ec.europa.eu/customs-4/calculation-customs-duties"
    "/customs-tariff/eu-customs-tariff-taric_en"
)

BULK_COMMIT_EVERY = 500

# Hier. Pos. value → NomenclatureLevel mapping (from EU CN XLSX export)
_HIER_POS_TO_LEVEL: dict[int, NomenclatureLevel] = {
    2: NomenclatureLevel.CHAPTER,
    4: NomenclatureLevel.HEADING,
    6: NomenclatureLevel.SUBHEADING,
    8: NomenclatureLevel.CN8,
    10: NomenclatureLevel.COMMODITY,
}

# XLSX namespace
_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


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
# XLSX reader (stdlib only — no openpyxl dependency)
# ---------------------------------------------------------------------------


def _xlsx_col_index(ref: str) -> int:
    """Convert XLSX column letter(s) to 0-based index: 'A'→0, 'B'→1, 'AA'→26."""
    letters = "".join(c for c in ref if c.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _xlsx_cell_value(cell: ET.Element) -> str:
    """Extract text value from an XLSX cell element (inline string or numeric)."""
    ns = _XLSX_NS
    if cell.get("t") == "inlineStr":
        is_el = cell.find(f"{{{ns}}}is")
        if is_el is not None:
            t_el = is_el.find(f"{{{ns}}}t")
            return t_el.text or "" if t_el is not None else ""
    else:
        v_el = cell.find(f"{{{ns}}}v")
        return v_el.text or "" if v_el is not None else ""
    return ""


def _read_xlsx(path: str) -> list[dict[str, str]]:
    """
    Read sheet1 of an XLSX file into a list of row dicts.

    Uses stdlib only (zipfile + xml.etree.ElementTree).
    Handles sparse rows correctly — XLSX may omit empty cells from the XML,
    so we use each cell's column reference (e.g. 'A1', 'C3') to place values.
    """
    ns = _XLSX_NS
    with zipfile.ZipFile(path) as z:
        with z.open("xl/worksheets/sheet1.xml") as f:
            tree = ET.parse(f)

    rows = tree.findall(f".//{{{ns}}}row")
    if not rows:
        return []

    # Determine header names and max column index from the first row.
    header_cells = rows[0].findall(f"{{{ns}}}c")
    max_col = 0
    for c in header_cells:
        ref = c.get("r", "")
        if ref:
            max_col = max(max_col, _xlsx_col_index(ref))

    headers: list[str] = [""] * (max_col + 1)
    for c in header_cells:
        ref = c.get("r", "")
        col = _xlsx_col_index(ref) if ref else 0
        if col < len(headers):
            headers[col] = _xlsx_cell_value(c)

    result: list[dict[str, str]] = []
    for row in rows[1:]:
        row_data: list[str] = [""] * len(headers)
        for c in row.findall(f"{{{ns}}}c"):
            ref = c.get("r", "")
            col = _xlsx_col_index(ref) if ref else 0
            if col < len(row_data):
                row_data[col] = _xlsx_cell_value(c)
        result.append(dict(zip(headers, row_data)))

    return result


def _parse_eu_date(value: str | None) -> date | None:
    """
    Parse dates from the EU CN XLSX export (DD-MM-YYYY) or ISO format (YYYY-MM-DD).
    """
    if not value:
        return None
    value = value.strip()
    # Try DD-MM-YYYY (format used in the EU Nomenclature XLSX)
    parts = value.split("-")
    if len(parts) == 3:
        try:
            if len(parts[0]) == 2 and len(parts[2]) == 4:
                # DD-MM-YYYY
                return date(int(parts[2]), int(parts[1]), int(parts[0]))
        except ValueError:
            pass
    # Fallback to ISO parser
    return parse_date(value)


def _ingest_xlsx_rows(rows: list[dict[str, str]], session: Session) -> int:
    """
    Ingest EU CN nomenclature from rows parsed from the official Nomenclature XLSX.

    Expected columns (EU CN XLSX export from taxation-customs.ec.europa.eu):
      Goods code   — 10-digit code + space + 2-char producline suffix (e.g. "0101000000 80")
      Start date   — DD-MM-YYYY
      End date     — DD-MM-YYYY (empty if still valid)
      Language     — "EN" / "FR" / … (we keep EN only)
      Hier. Pos.   — effective code length: 2=chapter, 4=heading, 6=subheading, 8=CN8, 10=TARIC10
      Indent       — visual dashes (ignored)
      Description  — commodity description text
    """
    count = 0
    for row in rows:
        # Skip non-English rows
        lang = row.get("Language", "").strip().upper()
        if lang and lang != "EN":
            continue

        goods_code_raw = row.get("Goods code", "").strip()
        if not goods_code_raw:
            continue

        # Strip the producline suffix ("0101000000 80" → "0101000000")
        full_code = goods_code_raw.split()[0] if " " in goods_code_raw else goods_code_raw

        hier_pos_str = row.get("Hier. Pos.", "").strip()
        try:
            hier_pos = int(hier_pos_str)
        except (ValueError, TypeError):
            hier_pos = 0

        # Trim the 10-digit code to the significant length given by Hier. Pos.
        if 2 <= hier_pos <= len(full_code):
            code = full_code[:hier_pos]
        else:
            # Fallback: strip trailing zeros (e.g. "0101000000" → "0101")
            code = full_code.rstrip("0") or full_code[:2]

        description = row.get("Description", "").strip()
        valid_from = _parse_eu_date(row.get("Start date"))
        valid_to = _parse_eu_date(row.get("End date"))

        level = _HIER_POS_TO_LEVEL.get(hier_pos, _cn_level(code))
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
            logger.info("EU TARIC XLSX — %d rows committed", count)

    session.commit()
    return count


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

def load_eu_chapter_notes(session: Session, client: RetryClient) -> None:
    """
    EU chapter notes are published in the Official Journal (EUR-Lex) and are
    not available via a machine-readable API endpoint.

    The old DDS2 endpoint (ec.europa.eu/taxation_customs/dds2/taric/chapter_notes/*)
    was decommissioned by the EU Commission and now returns 502 for all chapters.

    Chapter notes can be added manually via the admin interface if needed.
    This function is a no-op kept for API compatibility.
    """
    logger.info(
        "EU — chapter notes skipped (DDS2 portal decommissioned; "
        "add notes manually if required)"
    )


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
                "  2. Download the Combined Nomenclature XLSX or CSV.\n"
                "  3. Re-run ingestion with the file:\n"
                "       docker compose --profile ingest run --rm \\\n"
                "         -v /path/to/Nomenclature.xlsx:/data/cn.xlsx \\\n"
                "         ingest python -m app.ingestion --steps nomenclature "
                "--jurisdiction eu --eu-csv-path /data/cn.xlsx\n"
                "  Or outside Docker:\n"
                "       python -m app.ingestion --steps nomenclature "
                "--jurisdiction eu --eu-csv-path /path/to/Nomenclature.xlsx"
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
    Ingest EU nomenclature from a locally downloaded file.

    Accepts both XLSX (the official EU Nomenclature export format) and CSV.

    Usage:
        python -m app.ingestion.eu_loader --csv-path /path/to/Nomenclature\\ EN.xlsx
        python -m app.ingestion.eu_loader --csv-path /path/to/taric.csv
    """
    import pathlib

    path = pathlib.Path(csv_path)
    logger.info("EU — ingesting from local file: %s", csv_path)

    if path.suffix.lower() in (".xlsx", ".xls"):
        logger.info("EU — detected XLSX format, using built-in reader")
        rows = _read_xlsx(str(path))
        logger.info("EU — XLSX loaded: %d data rows", len(rows))
        count = _ingest_xlsx_rows(rows, session)
    else:
        text = path.read_text(encoding="utf-8-sig")
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
        help="Path to EU Nomenclature XLSX or CSV (skips auto-discovery)",
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
