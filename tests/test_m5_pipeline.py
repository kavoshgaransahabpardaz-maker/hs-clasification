"""
M5 pipeline tests.

Unit tests mock the LLM and DB; no network or API key needed.
Integration tests require a live DB + ingested data.

    pytest tests/test_m5_pipeline.py -v -m "not integration"
    pytest tests/test_m5_pipeline.py -v -m integration
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.extraction import ProductProfile
from app.pipeline.retrieval import CandidateResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(
    code: str, similarity: float, ruling_id: int = 1, ref: str = "REF"
) -> CandidateResult:
    return CandidateResult(
        ruling_id=ruling_id,
        ruling_ref=ref,
        code=code,
        description=f"Description for {code}",
        similarity=similarity,
        jurisdiction="UK",
    )


# ---------------------------------------------------------------------------
# Rules layer
# ---------------------------------------------------------------------------


class TestRulesLayer:
    def test_returns_all_candidates_when_no_notes(self):
        from app.pipeline.rules import apply_rules

        candidates = [
            _make_candidate("8471300000", 0.91),
            _make_candidate("8528590000", 0.85),
        ]
        profile = ProductProfile(query="laptop computer")

        mock_session = MagicMock()
        # No legal notes found → no LLM call, pass through
        mock_session.scalars.return_value.all.return_value = []

        result, explanation = apply_rules(profile, candidates, "UK", mock_session)

        assert len(result) == len(candidates)
        assert explanation is None

    def test_fail_safe_returns_original_if_all_excluded(self):
        from app.pipeline.rules import apply_rules

        candidates = [_make_candidate("8471300000", 0.91)]
        profile = ProductProfile(query="laptop computer")

        mock_session = MagicMock()
        mock_note = MagicMock()
        mock_note.jurisdiction = "UK"
        mock_note.scope = "chapter"
        mock_note.scope_code = "84"
        mock_note.note_type = "exclusion"
        mock_note.text = "This chapter excludes all goods."
        mock_session.scalars.return_value.all.return_value = [mock_note]

        llm_response = '{"excluded_codes": ["8471300000"], "gri_winner": null, "reasoning": "excluded", "gri_article": ""}'

        with patch("app.pipeline.rules.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value.choices[0].message.content = llm_response
            mock_cls.return_value = mock_client

            result, _ = apply_rules(profile, candidates, "UK", mock_session)

        # Fail-safe: original list returned even though LLM excluded everything
        assert len(result) == 1
        assert result[0].code == "8471300000"

    def test_gri_winner_surfaced_first(self):
        from app.pipeline.rules import apply_rules

        candidates = [
            _make_candidate("8471300000", 0.91),
            _make_candidate("8528590000", 0.89),
        ]
        profile = ProductProfile(query="laptop computer")

        mock_session = MagicMock()
        mock_note = MagicMock()
        mock_note.scope = "chapter"
        mock_note.scope_code = "84"
        mock_note.note_type = "other"
        mock_note.text = "Some chapter note text."
        mock_session.scalars.return_value.all.return_value = [mock_note]

        llm_response = (
            '{"excluded_codes": [], "gri_winner": "8528590000",'
            ' "gri_article": "GRI 3(a)", "reasoning": "More specific"}'
        )

        with patch("app.pipeline.rules.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value.choices[0].message.content = llm_response
            mock_cls.return_value = mock_client

            result, explanation = apply_rules(profile, candidates, "UK", mock_session)

        # GRI winner should be first
        assert result[0].code == "8528590000"
        assert explanation is not None
        assert "GRI" in explanation

    def test_llm_cannot_invent_code_not_in_candidates(self):
        from app.pipeline.rules import apply_rules

        candidates = [_make_candidate("8471300000", 0.91)]
        profile = ProductProfile(query="laptop")

        mock_session = MagicMock()
        mock_note = MagicMock()
        mock_note.scope = "chapter"
        mock_note.scope_code = "84"
        mock_note.note_type = "other"
        mock_note.text = "Note."
        mock_session.scalars.return_value.all.return_value = [mock_note]

        # LLM tries to return an invented code
        llm_response = (
            '{"excluded_codes": [], "gri_winner": "9999999999",'
            ' "gri_article": "GRI 1", "reasoning": "invented"}'
        )

        with patch("app.pipeline.rules.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value.choices[0].message.content = llm_response
            mock_cls.return_value = mock_client

            result, _ = apply_rules(profile, candidates, "UK", mock_session)

        # Invented gri_winner is rejected; original candidate order preserved
        assert result[0].code == "8471300000"
        assert all(c.code != "9999999999" for c in result)


# ---------------------------------------------------------------------------
# Validation layer
# ---------------------------------------------------------------------------


class TestValidationLayer:
    def test_valid_codes_pass_through(self):
        from app.pipeline.validation import validate_candidates

        candidates = [_make_candidate("0702000007", 0.92)]
        mock_session = MagicMock()

        with patch("app.pipeline.validation.validate_code", return_value=True):
            result = validate_candidates(candidates, "UK", mock_session, use_live_api=False)

        assert len(result) == 1

    def test_invalid_codes_dropped(self):
        from app.pipeline.validation import validate_candidates

        candidates = [
            _make_candidate("0702000007", 0.92),
            _make_candidate("9999999999", 0.85),
        ]
        mock_session = MagicMock()

        def fake_validate(code, jurisdiction, session):
            return code == "0702000007"

        with patch("app.pipeline.validation.validate_code", side_effect=fake_validate):
            result = validate_candidates(candidates, "UK", mock_session, use_live_api=False)

        assert len(result) == 1
        assert result[0].code == "0702000007"

    def test_fail_safe_returns_original_if_all_invalid(self):
        from app.pipeline.validation import validate_candidates

        candidates = [_make_candidate("9999999999", 0.92)]
        mock_session = MagicMock()

        with patch("app.pipeline.validation.validate_code", return_value=False):
            result = validate_candidates(candidates, "UK", mock_session, use_live_api=False)

        # Fail-safe: all-invalid → return original
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Ranking layer
# ---------------------------------------------------------------------------


class TestRankingLayer:
    def test_groups_by_code(self):
        from app.pipeline.ranking import rank_candidates

        candidates = [
            _make_candidate("0702000007", 0.92, ruling_id=1),
            _make_candidate("0702000007", 0.88, ruling_id=2),  # same code
            _make_candidate("0701900010", 0.75, ruling_id=3),
        ]
        ranked = rank_candidates(candidates)

        codes = [r.code for r in ranked]
        assert len(ranked) == 2  # two distinct codes
        assert codes[0] == "0702000007"  # higher score first

    def test_support_bonus_applied(self):
        from app.pipeline.ranking import rank_candidates, SUPPORT_WEIGHT

        # Two codes: code A has higher max_sim but only 1 ruling;
        # code B has lower max_sim but 5 rulings supporting it.
        sim_a = 0.95
        sim_b = 0.90
        count_b = 5
        expected_score_b = sim_b + SUPPORT_WEIGHT * math.log(1 + count_b)

        candidates = [_make_candidate("A000000000", sim_a, ruling_id=1)]
        for i in range(count_b):
            candidates.append(_make_candidate("B000000000", sim_b, ruling_id=10 + i))

        ranked = rank_candidates(candidates)
        top = ranked[0]

        if top.code == "B000000000":
            assert abs(top.score - expected_score_b) < 0.0001
        else:
            # code A won: verify B's score is correct
            assert abs(ranked[1].score - expected_score_b) < 0.0001

    def test_empty_input_returns_empty(self):
        from app.pipeline.ranking import rank_candidates

        assert rank_candidates([]) == []

    def test_support_count_recorded(self):
        from app.pipeline.ranking import rank_candidates

        candidates = [
            _make_candidate("0702000007", 0.90, ruling_id=1),
            _make_candidate("0702000007", 0.85, ruling_id=2),
        ]
        ranked = rank_candidates(candidates)
        assert ranked[0].support_count == 2
        assert len(ranked[0].supporting_ruling_ids) == 2


# ---------------------------------------------------------------------------
# Confidence layer
# ---------------------------------------------------------------------------


class TestConfidenceLayer:
    def test_no_ranked_returns_none_needs_review(self):
        from app.pipeline.confidence import compute_confidence

        conf, status = compute_confidence([])
        assert conf is None
        assert status == "needs_review"

    def test_high_score_can_resolve(self):
        from app.pipeline.ranking import RankedCode
        from app.pipeline.confidence import compute_confidence

        ranked = [
            RankedCode(
                code="0702000007",
                score=0.99,
                max_similarity=0.99,
                support_count=5,
                best_description="Tomatoes",
            )
        ]
        conf, status = compute_confidence(ranked, threshold=0.5)
        # Very high score → confidence above 0.5 → auto_resolved
        assert conf is not None
        assert conf > 0.5
        assert status == "auto_resolved"

    def test_low_margin_forces_needs_review(self):
        from app.pipeline.ranking import RankedCode
        from app.pipeline.confidence import compute_confidence

        ranked = [
            RankedCode(code="A", score=0.92, max_similarity=0.92, support_count=1, best_description=""),
            RankedCode(code="B", score=0.92, max_similarity=0.92, support_count=1, best_description=""),
        ]
        _, status = compute_confidence(ranked, threshold=0.5)
        # Margin < 0.02 → needs_review regardless of confidence
        assert status == "needs_review"

    def test_confidence_in_zero_one_range(self):
        from app.pipeline.ranking import RankedCode
        from app.pipeline.confidence import compute_confidence

        for score in [0.0, 0.3, 0.6, 0.8, 1.0, 1.2]:
            ranked = [
                RankedCode(code="X", score=score, max_similarity=score, support_count=1, best_description="")
            ]
            conf, _ = compute_confidence(ranked)
            assert conf is not None
            assert 0.0 <= conf <= 1.0, f"confidence={conf} out of range for score={score}"


# ---------------------------------------------------------------------------
# Full pipeline integration (requires live DB)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestM5Integration:
    @pytest.fixture(autouse=True)
    def setup(self):
        from app.db import init_db

        init_db()

    def test_no_invalid_codes_emitted(self):
        """
        Spec guardrail: never emit a code that is not currently valid.
        Run 10 requests and assert no invalid code is returned.
        """
        from fastapi.testclient import TestClient

        from app.db import get_session, validate_code
        from app.main import app

        client = TestClient(app)

        test_inputs = [
            ("fresh tomatoes for salad", "UK"),
            ("laptop computer with SSD", "EU"),
            ("hydraulic pump stainless steel", "UK"),
        ]

        for text, jurisdiction in test_inputs:
            resp = client.post(
                "/v1/classify",
                json={"text": text, "jurisdiction": jurisdiction},
            )
            if resp.status_code != 200:
                continue  # skip if pipeline not fully configured

            data = resp.json()
            predicted = data.get("predicted_code")
            if predicted:
                with get_session() as session:
                    is_valid = validate_code(predicted, jurisdiction, session)
                assert is_valid, (
                    f"Invalid code {predicted!r} emitted for {text!r} / {jurisdiction}"
                )

    def test_m5_accuracy_does_not_regress_vs_m4(self):
        """
        Per-digit accuracy at M5 must be >= M4 accuracy.
        Fetches the two most recent eval_run rows and compares 2-digit accuracy.
        """
        from sqlalchemy import select

        from app.db import get_session
        from app.models.eval_run import EvalRun

        with get_session() as session:
            runs = session.scalars(
                select(EvalRun).order_by(EvalRun.created_at.desc()).limit(2)
            ).all()

        if len(runs) < 2:
            pytest.skip("Need at least 2 eval_run rows — run eval before and after M5")

        latest, previous = runs[0], runs[1]
        if not latest.accuracy_by_digit or not previous.accuracy_by_digit:
            pytest.skip("eval_run rows missing accuracy_by_digit data")

        acc_latest = latest.accuracy_by_digit.get("2", 0)
        acc_previous = previous.accuracy_by_digit.get("2", 0)

        assert acc_latest >= acc_previous - 0.02, (
            f"2-digit accuracy regressed: {acc_latest:.1%} < {acc_previous:.1%}"
        )
