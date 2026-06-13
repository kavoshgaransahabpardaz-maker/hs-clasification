"""
Retrieval step — pgvector cosine search over the ruling corpus.

The HNSW index on embedding.vector makes this sub-millisecond at runtime.
Recall@K is the ceiling on the whole pipeline — optimise this step first.

Query flow:
  1. Embed the query string (from extraction.ProductProfile.query).
  2. Cosine search over embedding rows: object_type='ruling', jurisdiction=target,
     model=current_model.
  3. Join to ruling; filter out expired rulings.
  4. Return top-K (ruling, similarity) pairs sorted descending by similarity.

Note on eval filtering:
  In production all rulings (eval and non-eval) are searchable.
  The eval harness passes exclude_eval=True so eval-label rulings don't appear
  as their own nearest neighbours and inflate recall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Sequence

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion.embedder import Embedder, OpenAIEmbedder
from app.models.embedding import Embedding
from app.models.enums import Jurisdiction, ObjectType
from app.models.ruling import Ruling

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateResult:
    ruling_id: int
    ruling_ref: str | None
    code: str
    description: str
    similarity: float
    jurisdiction: str


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------


def retrieve_candidates(
    query: str,
    jurisdiction: str,
    session: Session,
    embedder: Embedder | None = None,
    top_k: int | None = None,
    exclude_eval: bool = False,
) -> list[CandidateResult]:
    """
    Embed *query* and return the top-K most similar rulings by cosine similarity.

    Args:
        query:        Clean retrieval query string (from extraction step).
        jurisdiction: "EU" or "UK".
        session:      Active DB session.
        embedder:     Embedder to use; defaults to OpenAIEmbedder.
        top_k:        Number of candidates; defaults to settings.retrieval_top_k.
        exclude_eval: If True, exclude is_eval=True rulings (use during eval).

    Returns:
        List of CandidateResult, sorted by similarity descending.
        Empty list if no embeddings exist for the jurisdiction.
    """
    if embedder is None:
        embedder = OpenAIEmbedder()
    if top_k is None:
        top_k = settings.retrieval_top_k

    if not query or not query.strip():
        logger.warning("retrieve_candidates called with empty query")
        return []

    # 1. Embed the query string.
    try:
        [query_vector] = embedder.embed([query.strip()])
    except Exception as exc:
        logger.error("Failed to embed query: %s", exc)
        return []

    # 2. Build pgvector cosine search.
    #    <=> is the cosine distance operator; lower = more similar.
    #    similarity = 1 - cosine_distance
    today = date.today()

    try:
        distance_expr = Embedding.vector.cosine_distance(query_vector)
    except AttributeError:
        # pgvector extension not available (e.g., unit test environment).
        logger.error(
            "pgvector cosine_distance not available — is the pgvector "
            "Python package installed and the DB extension enabled?"
        )
        return []

    similarity_expr = (1 - distance_expr).label("similarity")

    stmt = (
        select(Ruling, similarity_expr)
        .join(
            Embedding,
            (Embedding.object_id == Ruling.id)
            & (Embedding.object_type == ObjectType.RULING),
        )
        .where(
            Embedding.jurisdiction == jurisdiction,
            Embedding.model == embedder.model_name,
            or_(Ruling.valid_to.is_(None), Ruling.valid_to >= today),
        )
        .order_by(distance_expr)
        .limit(top_k)
    )

    if exclude_eval:
        stmt = stmt.where(Ruling.is_eval.is_(False))

    rows = session.execute(stmt).all()

    return [
        CandidateResult(
            ruling_id=ruling.id,
            ruling_ref=ruling.reference,
            code=ruling.assigned_code,
            description=ruling.product_description[:300],
            similarity=round(float(similarity), 4),
            jurisdiction=str(ruling.jurisdiction),
        )
        for ruling, similarity in rows
    ]


# ---------------------------------------------------------------------------
# Convenience: embed a list of texts and return vectors (used by eval harness)
# ---------------------------------------------------------------------------


def embed_texts(texts: Sequence[str], embedder: Embedder | None = None) -> list[list[float]]:
    """Embed a batch of texts using the default or provided embedder."""
    if embedder is None:
        embedder = OpenAIEmbedder()
    return embedder.embed(texts)
