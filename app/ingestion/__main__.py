"""
Unified ingestion CLI — runs M2 (nomenclature) and M3 (rulings + embeddings).

    # Full pipeline: nomenclature + rulings + embeddings for both jurisdictions
    python -m app.ingestion

    # Nomenclature only
    python -m app.ingestion --steps nomenclature

    # Rulings + embeddings only (assumes nomenclature already loaded)
    python -m app.ingestion --steps rulings

    # Embeddings only (assumes rulings already loaded)
    python -m app.ingestion --steps embeddings

    # Single jurisdiction
    python -m app.ingestion --jurisdiction uk

    # With local files (bypasses auto-discovery)
    python -m app.ingestion \\
        --eu-csv-path /data/taric.csv \\
        --ebti-csv-path /data/ebti.csv \\
        --atar-csv-path /data/atar.csv
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.db import get_session, init_db
from app.ingestion.atar_loader import run_atar_ingestion, run_atar_ingestion_from_file
from app.ingestion.ebti_loader import run_ebti_ingestion, run_ebti_ingestion_from_file
from app.ingestion.embed_rulings import embed_unembedded_rulings, verify_embedding_coverage
from app.ingestion.embedder import OpenAIEmbedder
from app.ingestion.eu_loader import run_eu_ingestion, run_eu_ingestion_from_file
from app.ingestion.uk_loader import run_uk_ingestion

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="HS Code Classification — full ingestion pipeline"
    )
    parser.add_argument(
        "--steps",
        choices=["all", "nomenclature", "rulings", "embeddings"],
        default="all",
        help="Pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--jurisdiction",
        choices=["uk", "eu", "all"],
        default="all",
        help="Jurisdiction scope (default: all)",
    )
    parser.add_argument(
        "--uk-mode",
        choices=["bulk", "delta", "full"],
        default="full",
        help="UK nomenclature mode (default: full = bulk CSV + API notes)",
    )
    # Local-file overrides for each source
    parser.add_argument("--eu-csv-path", default=None, help="Local TARIC CSV")
    parser.add_argument("--ebti-csv-path", default=None, help="Local EBTI CSV")
    parser.add_argument("--atar-csv-path", default=None, help="Local ATaR CSV")

    args = parser.parse_args()

    run_nomenclature = args.steps in ("all", "nomenclature")
    run_rulings = args.steps in ("all", "rulings")
    run_embeddings = args.steps in ("all", "rulings", "embeddings")

    do_uk = args.jurisdiction in ("uk", "all")
    do_eu = args.jurisdiction in ("eu", "all")

    # Ensure schema + pgvector extension exist.
    init_db()

    # ------------------------------------------------------------------
    # M2 — Nomenclature
    # ------------------------------------------------------------------
    if run_nomenclature:
        with get_session() as session:
            if do_uk:
                log.info("=== M2: UK nomenclature (mode=%s) ===", args.uk_mode)
                run_uk_ingestion(session, mode=args.uk_mode)

            if do_eu:
                log.info("=== M2: EU TARIC nomenclature ===")
                if args.eu_csv_path:
                    run_eu_ingestion_from_file(session, args.eu_csv_path)
                else:
                    run_eu_ingestion(session)

    # ------------------------------------------------------------------
    # M3 — Rulings
    # ------------------------------------------------------------------
    if run_rulings:
        with get_session() as session:
            if do_eu:
                log.info("=== M3: EBTI EU rulings ===")
                if args.ebti_csv_path:
                    run_ebti_ingestion_from_file(session, args.ebti_csv_path)
                else:
                    run_ebti_ingestion(session)

            if do_uk:
                log.info("=== M3: ATaR UK rulings ===")
                if args.atar_csv_path:
                    run_atar_ingestion_from_file(session, args.atar_csv_path)
                else:
                    run_atar_ingestion(session)

    # ------------------------------------------------------------------
    # M3 — Embeddings
    # ------------------------------------------------------------------
    if run_embeddings:
        log.info("=== M3: Embedding rulings ===")
        embedder = OpenAIEmbedder()
        with get_session() as session:
            from app.models.enums import Jurisdiction

            jur: Jurisdiction | None = None
            if args.jurisdiction == "uk":
                jur = Jurisdiction.UK
            elif args.jurisdiction == "eu":
                jur = Jurisdiction.EU

            embed_unembedded_rulings(session, embedder, jurisdiction=jur)

            coverage = verify_embedding_coverage(session, embedder.model_name)
            log.info(
                "Coverage: %d/%d rulings embedded (%d missing)",
                coverage["embedded"],
                coverage["total_rulings"],
                coverage["missing"],
            )
            if coverage["missing"] > 0:
                log.warning(
                    "%d rulings have no embedding — check API key / rerun.",
                    coverage["missing"],
                )

    log.info("=== Ingestion complete ===")


if __name__ == "__main__":
    main()
    sys.exit(0)
