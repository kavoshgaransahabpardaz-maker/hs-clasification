"""
Database engine, session factory, init_db(), and validate_code().
"""

from contextlib import contextmanager
from datetime import date
from typing import Generator

from sqlalchemy import create_engine, or_, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import Base, NomenclatureNode

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """
    Bootstrap the database for local dev / first spike.

    Steps:
      1. Enable the pgvector extension (idempotent).
      2. Create all tables + regular indexes via SQLAlchemy DDL.
      3. The HNSW index on embedding.vector is created by an after_create
         event listener in app/models/embedding.py.

    Production: use Alembic migrations instead.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional session, rolling back on exception."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def validate_code(code: str, jurisdiction: str, session: Session) -> bool:
    """
    Return True iff *code* is present and currently valid in nomenclature_node
    for the given *jurisdiction*.

    A code is valid when today falls within [valid_from, valid_to] (NULL on
    either end means unbounded).
    """
    today = date.today()
    node = session.scalar(
        select(NomenclatureNode)
        .where(
            NomenclatureNode.code == code,
            NomenclatureNode.jurisdiction == jurisdiction,
            or_(
                NomenclatureNode.valid_from.is_(None),
                NomenclatureNode.valid_from <= today,
            ),
            or_(
                NomenclatureNode.valid_to.is_(None),
                NomenclatureNode.valid_to >= today,
            ),
        )
        .limit(1)
    )
    return node is not None


def seed_db(session: Session) -> None:
    """
    Insert a minimal set of nomenclature_node rows for M1 smoke tests.

    These are real HS/CN codes so validate_code() returns True for them.
    """
    from app.models.enums import Jurisdiction, NomenclatureLevel

    seeds = [
        # UK commodity: fresh tomatoes
        NomenclatureNode(
            code="0702000007",
            level=NomenclatureLevel.COMMODITY,
            jurisdiction=Jurisdiction.UK,
            description="Tomatoes, fresh or chilled",
            path="07.0702.070200.0702000007",
        ),
        # UK heading parent
        NomenclatureNode(
            code="0702",
            level=NomenclatureLevel.HEADING,
            jurisdiction=Jurisdiction.UK,
            description="Tomatoes, fresh or chilled",
            path="07.0702",
        ),
        # EU CN8: fresh tomatoes
        NomenclatureNode(
            code="07020000",
            level=NomenclatureLevel.CN8,
            jurisdiction=Jurisdiction.EU,
            description="Tomatoes, fresh or chilled",
            path="07.0702.07020000",
        ),
    ]

    for node in seeds:
        existing = session.scalar(
            select(NomenclatureNode).where(
                NomenclatureNode.code == node.code,
                NomenclatureNode.jurisdiction == node.jurisdiction,
            )
        )
        if existing is None:
            session.add(node)
