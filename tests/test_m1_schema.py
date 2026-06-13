"""
M1 smoke tests — run against a live Postgres DB with pgvector.

Set DATABASE_URL in your environment or .env before running:

    pytest tests/test_m1_schema.py -v
"""

import pytest

from app.db import get_session, init_db, seed_db, validate_code


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    with get_session() as session:
        seed_db(session)


def test_validate_code_uk_tomatoes():
    with get_session() as session:
        assert validate_code("0702000007", "UK", session) is True


def test_validate_code_eu_tomatoes():
    with get_session() as session:
        assert validate_code("07020000", "EU", session) is True


def test_validate_code_unknown_returns_false():
    with get_session() as session:
        assert validate_code("9999999999", "UK", session) is False


def test_validate_code_wrong_jurisdiction_returns_false():
    with get_session() as session:
        # UK code asked against EU jurisdiction
        assert validate_code("0702000007", "EU", session) is False
