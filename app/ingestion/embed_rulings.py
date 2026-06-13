"""
Embed all ruling rows that do not yet have a matching embedding row.

This is the M3 "3.5 Embedding the corpus" step from the spec.

  • Queries ruling rows missing from the embedding table for the current model.
  • Batches product_description strings through OpenAIEmbedder (256 texts/call).
  • Writes one embedding row per ruling (object_type='ruling').
  • Re-running is safe: already-embedded rulings are skipped.
  • Re-embed everything: pass --reembed to drop existing vectors first.

Usage:
    python -m app.ingestion.embed_rulings
    python -m app.ingestion.embed_rulings --reembed          # force re-embed all
    python -m app.ingestion.embed_rulings --jurisdiction eu  # EU only
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_session
from app.ingestion.embedder import Embedder, OpenAIEmbedder
from app.models.embedding import Embedding
from app.models.enums import Jurisdiction, ObjectType
from app.models.ruling import Ruling

logger = logging.getLogger(__name__)

EMBED_COMMIT_EVERY = 50  # flush + commit every N rulings


# ---------------------------------------------------------------------------
# Core embedding logic
# ---------------------------------------------------------------------------


def embed_unembedded_rulings(
    session: Session,
    embedder: Embedder,
    jurisdiction: Jurisdiction | None = None,
    batch_size: int = 256,
) -> int:
    """
    Find ruling rows without an embedding for the current model and embed them.

    Returns the number of new embedding rows written.

    Args:
        session:      Active SQLAlchemy session.
        embedder:     Embedder implementation to use.
        jurisdiction: Limit to a single jurisdiction; None = both.
        batch_size:   Texts per OpenAI call (must be ≤ 2048).
    """
    # Subquery: ruling IDs already embedded with this model
    already_embedded = select(Embedding.object_id).where(
        Embedding.object_type == ObjectType.RULING,
        Embedding.model == embedder.model_name,
    )

    query = select(Ruling).where(Ruling.id.not_in(already_embedded))
    if jurisdiction is not None:
        query = query.where(Ruling.jurisdiction == jurisdiction)

    rulings: Sequence[Ruling] = session.scalars(query).all()
    total_needed = len(rulings)

    if total_needed == 0:
        logger.info("All rulings already embedded with model '%s'.", embedder.model_name)
        return 0

    logger.info(
        "Embedding %d rulings (model=%s, jurisdiction=%s)",
        total_needed,
        embedder.model_name,
        jurisdiction or "all",
    )

    written = 0
    for batch_start in range(0, total_needed, batch_size):
        batch = rulings[batch_start : batch_start + batch_size]
        texts = [r.product_description for r in batch]

        try:
            vectors = embedder.embed(texts)
        except Exception as exc:
            logger.error(
                "Embedding batch %d–%d failed: %s — skipping batch",
                batch_start,
                batch_start + len(batch) - 1,
                exc,
            )
            continue

        if len(vectors) != len(batch):
            logger.error(
                "Expected %d vectors, got %d — skipping batch",
                len(batch),
                len(vectors),
            )
            continue

        for ruling, vector in zip(batch, vectors):
            session.add(
                Embedding(
                    object_type=ObjectType.RULING,
                    object_id=ruling.id,
                    jurisdiction=ruling.jurisdiction,
                    model=embedder.model_name,
                    vector=vector,
                )
            )
            written += 1

        if written % EMBED_COMMIT_EVERY == 0 or batch_start + batch_size >= total_needed:
            session.commit()
            logger.info(
                "Embeddings — %d/%d written (%.0f%%)",
                written,
                total_needed,
                100 * written / total_needed,
            )

    logger.info("Embedding complete: %d new rows written.", written)
    return written


def reembed_all_rulings(
    session: Session,
    embedder: Embedder,
    jurisdiction: Jurisdiction | None = None,
) -> int:
    """
    Delete existing embedding rows for the current model and re-embed everything.

    Use when the embedding model changes (the model column tracks provenance,
    so you can also keep old embeddings and run a model migration).
    """
    stmt = delete(Embedding).where(
        Embedding.object_type == ObjectType.RULING,
        Embedding.model == embedder.model_name,
    )
    if jurisdiction is not None:
        stmt = stmt.where(Embedding.jurisdiction == jurisdiction)

    deleted = session.execute(stmt).rowcount
    session.commit()
    logger.info("Deleted %d existing embedding rows for model '%s'.", deleted, embedder.model_name)

    return embed_unembedded_rulings(session, embedder, jurisdiction)


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------


def verify_embedding_coverage(session: Session, model_name: str) -> dict[str, int]:
    """
    Return counts to verify that every ruling has a matching embedding.

    Returns {"total_rulings": N, "embedded": M, "missing": N-M}.
    """
    total_rulings: int = session.scalar(
        select(func.count(Ruling.id))
    ) or 0

    embedded: int = session.scalar(
        select(func.count(Embedding.id)).where(
            Embedding.object_type == ObjectType.RULING,
            Embedding.model == model_name,
        )
    ) or 0

    return {
        "total_rulings": total_rulings,
        "embedded": embedded,
        "missing": total_rulings - embedded,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Embed ruling product descriptions")
    parser.add_argument(
        "--jurisdiction",
        choices=["uk", "eu"],
        default=None,
        help="Only embed rulings for this jurisdiction (default: both)",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Delete existing embeddings for this model and re-embed everything",
    )
    args = parser.parse_args()

    jurisdiction: Jurisdiction | None = None
    if args.jurisdiction == "uk":
        jurisdiction = Jurisdiction.UK
    elif args.jurisdiction == "eu":
        jurisdiction = Jurisdiction.EU

    embedder = OpenAIEmbedder()

    with get_session() as session:
        if args.reembed:
            written = reembed_all_rulings(session, embedder, jurisdiction)
        else:
            written = embed_unembedded_rulings(session, embedder, jurisdiction)

        coverage = verify_embedding_coverage(session, embedder.model_name)
        logger.info(
            "Coverage: %d/%d rulings embedded (%d missing)",
            coverage["embedded"],
            coverage["total_rulings"],
            coverage["missing"],
        )
        if coverage["missing"] > 0:
            logger.warning(
                "%d rulings have no embedding — rerun this script to fill gaps.",
                coverage["missing"],
            )


if __name__ == "__main__":
    _cli()
    sys.exit(0)
