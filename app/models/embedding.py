from sqlalchemy import DDL, Index, String, event
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.enums import JurisdictionType, ObjectTypeType

# Import at runtime to avoid hard dependency during migrations on machines
# without pgvector installed.
try:
    from pgvector.sqlalchemy import Vector as PGVector

    _VECTOR_TYPE = PGVector(1536)
except ImportError:  # pragma: no cover
    from sqlalchemy import LargeBinary

    _VECTOR_TYPE = LargeBinary()  # type: ignore[assignment]


class Embedding(Base):
    """
    Polymorphic vector store backed by pgvector HNSW index.

    object_type + object_id reference either ruling.id or nomenclature_node.id.
    jurisdiction is denormalized so filtered ANN searches need no join.
    model tracks which embedding model produced each vector — re-embed all
    rows whenever you change models.
    """

    __tablename__ = "embedding"
    __table_args__ = (
        Index("ix_embedding_object", "object_type", "object_id"),
        Index("ix_embedding_jurisdiction_type", "jurisdiction", "object_type"),
        # HNSW index created via DDL event below (requires pgvector extension)
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    object_type: Mapped[str] = mapped_column(ObjectTypeType, nullable=False)
    object_id: Mapped[int] = mapped_column(nullable=False)
    # denormalized for filtered ANN without a join
    jurisdiction: Mapped[str] = mapped_column(JurisdictionType, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    vector: Mapped[list] = mapped_column(_VECTOR_TYPE, nullable=False)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# HNSW index — created after the table exists so the pgvector extension can
# resolve the operator class.  m=16, ef_construction=64 per the spec.
# ---------------------------------------------------------------------------

@event.listens_for(Embedding.__table__, "after_create")
def _create_hnsw_index(target, connection, **kw):  # type: ignore[no-untyped-def]
    connection.execute(
        DDL(
            "CREATE INDEX IF NOT EXISTS ix_embedding_vector_hnsw "
            "ON embedding USING hnsw (vector vector_cosine_ops) "
            "WITH (m=16, ef_construction=64)"
        )
    )
