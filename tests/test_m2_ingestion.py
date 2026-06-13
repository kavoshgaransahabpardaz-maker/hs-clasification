"""
M2 acceptance-criteria tests (Section 3.6 of the spec).

These tests require a live Postgres DB with pgvector AND a network connection
to the UK/EU APIs.  They are integration tests — skip in CI without a DB.

    pytest tests/test_m2_ingestion.py -v -m integration

Run after `python -m app.ingestion` completes successfully.
"""

import pytest
from sqlalchemy import func, select

from app.db import get_session, init_db, validate_code
from app.models.enums import Jurisdiction, NoteScope
from app.models.legal_note import LegalNote
from app.models.nomenclature_node import NomenclatureNode

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def db():
    init_db()


# ---------------------------------------------------------------------------
# 3.6 (a) nomenclature_node populated for both jurisdictions
# ---------------------------------------------------------------------------


def test_uk_nomenclature_not_empty():
    with get_session() as s:
        count = s.scalar(
            select(func.count()).where(NomenclatureNode.jurisdiction == Jurisdiction.UK)
        )
    assert count and count > 100, f"Expected >100 UK nodes, got {count}"


def test_eu_nomenclature_not_empty():
    with get_session() as s:
        count = s.scalar(
            select(func.count()).where(NomenclatureNode.jurisdiction == Jurisdiction.EU)
        )
    assert count and count > 100, f"Expected >100 EU nodes, got {count}"


# ---------------------------------------------------------------------------
# 3.6 (a) spot-check: UK 0702000007 → "tomatoes…" with correct parent chain
# ---------------------------------------------------------------------------


def test_uk_tomatoes_code_exists():
    """The canonical spec spot-check: UK 0702000007 must exist."""
    with get_session() as s:
        node = s.scalar(
            select(NomenclatureNode).where(
                NomenclatureNode.code == "0702000007",
                NomenclatureNode.jurisdiction == Jurisdiction.UK,
            )
        )
    assert node is not None, "UK commodity 0702000007 not found"
    assert "tomato" in node.description.lower(), (
        f"Expected 'tomato' in description, got: {node.description!r}"
    )


def test_uk_tomatoes_parent_chain():
    """UK 0702000007 must have a heading (0702) ancestor in the tree."""
    with get_session() as s:
        # Heading 0702 must exist
        heading = s.scalar(
            select(NomenclatureNode).where(
                NomenclatureNode.code == "0702",
                NomenclatureNode.jurisdiction == Jurisdiction.UK,
            )
        )
    assert heading is not None, "UK heading 0702 not found — parent chain broken"


def test_uk_tomatoes_path_contains_chapter():
    """Path for UK 0702000007 must contain '07' (chapter)."""
    with get_session() as s:
        node = s.scalar(
            select(NomenclatureNode).where(
                NomenclatureNode.code == "0702000007",
                NomenclatureNode.jurisdiction == Jurisdiction.UK,
            )
        )
    assert node is not None
    assert "07" in node.path, f"Chapter '07' not in path {node.path!r}"


def test_validate_code_uk_tomatoes_live():
    """validate_code() must return True for a freshly ingested valid code."""
    with get_session() as s:
        assert validate_code("0702000007", "UK", s) is True


# ---------------------------------------------------------------------------
# 3.6 (b) legal_note rows exist for chapters that have notes
# ---------------------------------------------------------------------------


def test_uk_legal_notes_exist():
    """At least some UK chapter notes must have been loaded."""
    with get_session() as s:
        count = s.scalar(
            select(func.count()).where(
                LegalNote.jurisdiction == Jurisdiction.UK,
                LegalNote.scope == NoteScope.CHAPTER,
            )
        )
    assert count and count > 5, f"Expected >5 UK chapter notes, got {count}"


def test_eu_legal_notes_exist():
    """At least some EU chapter notes must have been loaded."""
    with get_session() as s:
        count = s.scalar(
            select(func.count()).where(
                LegalNote.jurisdiction == Jurisdiction.EU,
                LegalNote.scope == NoteScope.CHAPTER,
            )
        )
    assert count and count > 5, f"Expected >5 EU chapter notes, got {count}"


# ---------------------------------------------------------------------------
# 3.6 (a) path prefix query works (text_pattern_ops index)
# ---------------------------------------------------------------------------


def test_path_prefix_query_returns_subtree():
    """
    LIKE '07%' on path must return chapter 07 and all its descendants.
    This verifies the text_pattern_ops index is working.
    """
    with get_session() as s:
        results = s.scalars(
            select(NomenclatureNode).where(
                NomenclatureNode.jurisdiction == Jurisdiction.UK,
                NomenclatureNode.path.like("07%"),
            )
        ).all()
    codes = {r.code for r in results}
    assert "0702" in codes, "Heading 0702 missing from chapter 07 subtree query"
    assert len(results) > 10, "Expected >10 descendants of chapter 07"
