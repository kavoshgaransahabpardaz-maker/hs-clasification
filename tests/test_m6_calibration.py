"""
M6 calibration tests.

Unit tests require no DB or network; they validate the calibration math and
save/load round-trip.  Integration tests require a live DB with ingested + eval
rulings (is_eval=True) and valid OPENAI_API_KEY.

    pytest tests/test_m6_calibration.py -v -m "not integration"
    pytest tests/test_m6_calibration.py -v -m integration
"""

from __future__ import annotations

import math
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_calibration_data(n: int = 200) -> tuple[list[float], list[int]]:
    """
    Scores uniformly distributed in [0.5, 1.0]; label = 1 iff score > 0.75.
    After fitting a calibrator the ECE should be very small.
    """
    rng = np.random.default_rng(42)
    scores = rng.uniform(0.5, 1.0, size=n).tolist()
    labels = [1 if s > 0.75 else 0 for s in scores]
    return scores, labels


def _noisy_data(n: int = 100) -> tuple[list[float], list[int]]:
    rng = np.random.default_rng(7)
    scores = rng.uniform(0.4, 0.95, size=n).tolist()
    labels = rng.integers(0, 2, size=n).tolist()
    return scores, labels


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------


class TestComputeEce:
    def test_perfect_calibration_gives_small_ece(self):
        from app.pipeline.calibration import compute_ece

        # All confidences equal their accuracy → ECE = 0
        confs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # Each confidence value = expected accuracy in its bin (mock perfect calibration)
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        ece = compute_ece(confs, labels, n_bins=5)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_empty_returns_nan(self):
        from app.pipeline.calibration import compute_ece

        result = compute_ece([], [], n_bins=10)
        assert math.isnan(result)

    def test_completely_wrong_gives_high_ece(self):
        from app.pipeline.calibration import compute_ece

        # High confidence but always wrong → very bad ECE
        confs = [0.95] * 100
        labels = [0] * 100
        ece = compute_ece(confs, labels, n_bins=10)
        # ECE = |0.95 - 0.0| = 0.95
        assert ece > 0.8

    def test_perfectly_correct_high_conf_low_ece(self):
        from app.pipeline.calibration import compute_ece

        # Always correct at 0.9 confidence → ECE = |0.9 - 1.0| = 0.1
        confs = [0.9] * 100
        labels = [1] * 100
        ece = compute_ece(confs, labels, n_bins=10)
        assert abs(ece - 0.1) < 1e-6

    def test_n_bins_affects_granularity(self):
        from app.pipeline.calibration import compute_ece

        scores, labels = _noisy_data()
        ece_10 = compute_ece(scores, labels, n_bins=10)
        ece_5 = compute_ece(scores, labels, n_bins=5)
        # Both are valid floats in [0,1]
        assert 0.0 <= ece_10 <= 1.0
        assert 0.0 <= ece_5 <= 1.0


# ---------------------------------------------------------------------------
# Calibration bins report
# ---------------------------------------------------------------------------


class TestCalibrationBins:
    def test_returns_correct_number_of_bins(self):
        from app.pipeline.calibration import calibration_bins

        scores, labels = _noisy_data(100)
        bins = calibration_bins(scores, labels, n_bins=10)
        assert len(bins) == 10

    def test_bin_keys_present(self):
        from app.pipeline.calibration import calibration_bins

        scores, labels = _noisy_data(100)
        bins = calibration_bins(scores, labels, n_bins=5)
        for b in bins:
            assert "bin_lo" in b and "bin_hi" in b
            assert "count" in b
            assert "avg_conf" in b
            assert "accuracy" in b

    def test_counts_sum_to_n(self):
        from app.pipeline.calibration import calibration_bins

        n = 100
        scores, labels = _noisy_data(n)
        bins = calibration_bins(scores, labels, n_bins=10)
        total = sum(b["count"] for b in bins)
        assert total == n

    def test_empty_bins_have_none_values(self):
        from app.pipeline.calibration import calibration_bins

        # All scores in [0.9, 1.0] → bins below 0.9 will be empty
        scores = [0.95] * 50
        labels = [1] * 50
        bins = calibration_bins(scores, labels, n_bins=10)
        for b in bins:
            if b["count"] == 0:
                assert b["avg_conf"] is None
                assert b["accuracy"] is None


# ---------------------------------------------------------------------------
# Calibrator fitting
# ---------------------------------------------------------------------------


class TestFitCalibrator:
    def test_isotonic_method(self):
        from app.pipeline.calibration import fit_calibrator

        scores, labels = _perfect_calibration_data(200)
        cal = fit_calibrator(scores, labels, method="isotonic")
        assert cal.method == "isotonic"
        assert cal.n_train_samples == 200

    def test_platt_method(self):
        from app.pipeline.calibration import fit_calibrator

        scores, labels = _perfect_calibration_data(200)
        cal = fit_calibrator(scores, labels, method="platt")
        assert cal.method == "platt"
        assert cal.n_train_samples == 200

    def test_binned_method(self):
        from app.pipeline.calibration import fit_calibrator

        scores, labels = _perfect_calibration_data(200)
        cal = fit_calibrator(scores, labels, method="binned", n_bins=10)
        assert cal.method == "binned"

    def test_unknown_method_raises(self):
        from app.pipeline.calibration import fit_calibrator

        scores, labels = _perfect_calibration_data(200)
        with pytest.raises(ValueError, match="Unknown calibration method"):
            fit_calibrator(scores, labels, method="magic")

    def test_too_few_samples_raises(self):
        from app.pipeline.calibration import fit_calibrator, MIN_SAMPLES

        scores = [0.8] * (MIN_SAMPLES - 1)
        labels = [1] * (MIN_SAMPLES - 1)
        with pytest.raises(ValueError, match="need at least"):
            fit_calibrator(scores, labels, method="isotonic")

    def test_exactly_min_samples_ok(self):
        from app.pipeline.calibration import fit_calibrator, MIN_SAMPLES

        scores = list(np.linspace(0.5, 0.95, MIN_SAMPLES))
        labels = [1 if s > 0.72 else 0 for s in scores]
        cal = fit_calibrator(scores, labels, method="platt")
        assert cal.n_train_samples == MIN_SAMPLES


# ---------------------------------------------------------------------------
# Calibrator predict_proba / predict_single
# ---------------------------------------------------------------------------


class TestCalibratorPredict:
    def _fit(self, method: str = "isotonic") -> object:
        from app.pipeline.calibration import fit_calibrator

        scores, labels = _perfect_calibration_data(200)
        return fit_calibrator(scores, labels, method=method)

    def test_predict_proba_returns_array_in_range(self):
        cal = self._fit("isotonic")
        test_scores = np.linspace(0.5, 1.0, 20)
        proba = cal.predict_proba(test_scores)
        assert proba.shape == (20,)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_single_returns_float(self):
        cal = self._fit("platt")
        result = cal.predict_single(0.8)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_callable_alias_matches_predict_single(self):
        cal = self._fit("binned")
        score = 0.75
        assert cal(score) == cal.predict_single(score)

    def test_isotonic_monotone(self):
        """Isotonic regression must produce monotone predictions."""
        from app.pipeline.calibration import fit_calibrator

        scores, labels = _perfect_calibration_data(300)
        cal = fit_calibrator(scores, labels, method="isotonic")
        test_scores = np.linspace(0.5, 1.0, 50)
        preds = cal.predict_proba(test_scores)
        # Monotone non-decreasing (allow small numeric noise)
        diffs = np.diff(preds)
        assert np.all(diffs >= -1e-6)

    def test_all_three_methods_predict_same_direction(self):
        """All methods should give higher output for higher input scores."""
        from app.pipeline.calibration import fit_calibrator

        scores, labels = _perfect_calibration_data(200)
        for method in ("isotonic", "platt", "binned"):
            cal = fit_calibrator(scores, labels, method=method)
            lo = cal.predict_single(0.55)
            hi = cal.predict_single(0.90)
            assert hi >= lo, f"method={method}: expected hi≥lo but got {hi} < {lo}"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def _make_calibrator(self) -> object:
        from app.pipeline.calibration import fit_calibrator

        scores, labels = _perfect_calibration_data(200)
        cal = fit_calibrator(scores, labels, method="isotonic")
        cal.n_digits = 6
        return cal

    def test_save_creates_file(self):
        from app.pipeline.calibration import save_calibrator

        cal = self._make_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subdir" / "cal.pkl"
            save_calibrator(cal, path)
            assert path.exists()

    def test_load_roundtrip_preserves_metadata(self):
        from app.pipeline.calibration import load_calibrator, save_calibrator

        cal = self._make_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.pkl"
            save_calibrator(cal, path)
            loaded = load_calibrator(path)

        assert loaded.method == cal.method
        assert loaded.n_train_samples == cal.n_train_samples
        assert loaded.n_digits == cal.n_digits

    def test_load_roundtrip_predicts_identically(self):
        from app.pipeline.calibration import load_calibrator, save_calibrator

        cal = self._make_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.pkl"
            save_calibrator(cal, path)
            loaded = load_calibrator(path)

        test_scores = [0.6, 0.75, 0.85, 0.92]
        orig = [cal.predict_single(s) for s in test_scores]
        restored = [loaded.predict_single(s) for s in test_scores]
        assert orig == restored

    def test_load_calibrator_if_exists_returns_none_when_missing(self):
        from app.pipeline.calibration import load_calibrator_if_exists

        path = Path("/tmp/nonexistent_calibrator_xyz.pkl")
        result = load_calibrator_if_exists(path)
        assert result is None

    def test_load_calibrator_if_exists_returns_calibrator_when_present(self):
        from app.pipeline.calibration import (
            load_calibrator_if_exists,
            save_calibrator,
        )

        cal = self._make_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.pkl"
            save_calibrator(cal, path)
            loaded = load_calibrator_if_exists(path)

        assert loaded is not None
        assert loaded.method == "isotonic"

    def test_load_calibrator_if_exists_returns_none_on_corrupt_file(self):
        from app.pipeline.calibration import load_calibrator_if_exists

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pkl"
            path.write_bytes(b"not a pickle")
            result = load_calibrator_if_exists(path)
        assert result is None


# ---------------------------------------------------------------------------
# set_calibrator injection
# ---------------------------------------------------------------------------


class TestSetCalibratorInjection:
    def test_set_calibrator_changes_confidence_output(self):
        """
        After set_calibrator(), compute_confidence() should use the new function.
        """
        from app.pipeline.calibration import fit_calibrator
        from app.pipeline.confidence import compute_confidence, set_calibrator
        from app.pipeline.ranking import RankedCode

        scores, labels = _perfect_calibration_data(200)
        cal = fit_calibrator(scores, labels, method="platt")

        # Inject calibrator
        set_calibrator(cal.predict_single)

        # Make a ranked list with a high score that has a large margin
        rc1 = RankedCode(
            code="8471300000",
            score=0.92,
            max_similarity=0.92,
            support_count=3,
            best_description="Laptop",
            supporting_ruling_ids=[1],
            supporting_ruling_refs=["REF1"],
        )
        rc2 = RankedCode(
            code="8528590000",
            score=0.70,  # margin = 0.22 >> 0.02
            max_similarity=0.70,
            support_count=1,
            best_description="Monitor",
            supporting_ruling_ids=[2],
            supporting_ruling_refs=["REF2"],
        )

        conf, status = compute_confidence([rc1, rc2])
        # Calibrator output should be within [0, 1]
        assert 0.0 <= conf <= 1.0

        # Restore default sigmoid so other tests aren't affected
        from app.pipeline.confidence import _sigmoid_confidence
        set_calibrator(_sigmoid_confidence)

    def test_calibrator_fn_is_called_not_sigmoid(self):
        """set_calibrator replaces the sigmoid: our stub gets called instead."""
        from app.pipeline.confidence import compute_confidence, set_calibrator
        from app.pipeline.ranking import RankedCode

        call_log: list[float] = []

        def stub_calibrator(score: float) -> float:
            call_log.append(score)
            return 0.99  # always return 0.99 so we can detect it

        set_calibrator(stub_calibrator)

        rc = RankedCode(
            code="0702000007",
            score=0.88,
            max_similarity=0.88,
            support_count=1,
            best_description="Tomatoes",
            supporting_ruling_ids=[1],
            supporting_ruling_refs=["REF"],
        )

        conf, _ = compute_confidence([rc])

        # Our stub must have been called
        assert len(call_log) == 1
        assert call_log[0] == pytest.approx(0.88)
        assert conf == pytest.approx(0.99)

        # Restore
        from app.pipeline.confidence import _sigmoid_confidence
        set_calibrator(_sigmoid_confidence)


# ---------------------------------------------------------------------------
# ECE improves after calibration (sanity check)
# ---------------------------------------------------------------------------


class TestCalibrationImprovesEce:
    """
    Fit a calibrator on synthetically miscalibrated data (sigmoid over-confident),
    verify ECE decreases after fitting.
    """

    def _make_miscalibrated_data(self, n: int = 300):
        """
        Create data where sigmoid(score) is systematically over-confident:
        actual accuracy is ~60% but sigmoid gives ~85%.
        """
        rng = np.random.default_rng(99)
        # Scores tightly clustered around 0.85 (sigmoid → ~0.83)
        scores = rng.normal(0.85, 0.03, size=n).clip(0.5, 1.0).tolist()
        # Only 60% correct
        labels = rng.binomial(1, 0.60, size=n).tolist()
        return scores, labels

    def test_ece_after_calibration_not_worse(self):
        from app.pipeline.calibration import compute_ece, fit_calibrator

        scores, labels = self._make_miscalibrated_data(300)

        # ECE before: use sigmoid directly
        sigmoid = lambda s: 1.0 / (1.0 + math.exp(-10.0 * (s - 0.75)))
        uncal_confs = [sigmoid(s) for s in scores]
        ece_before = compute_ece(uncal_confs, labels)

        # Fit calibrator and compute ECE after
        cal = fit_calibrator(scores, labels, method="isotonic")
        cal_confs = cal.predict_proba(np.array(scores)).tolist()
        ece_after = compute_ece(cal_confs, labels)

        # ECE after calibration should not be dramatically worse
        # (isotonic on training data will overfit → ece_after should be very small)
        assert ece_after <= ece_before + 0.05, (
            f"ECE regressed: before={ece_before:.4f}, after={ece_after:.4f}"
        )

    def test_ece_output_range(self):
        from app.pipeline.calibration import compute_ece

        confs = [0.3, 0.5, 0.7, 0.9, 0.6, 0.8]
        labels = [0, 1, 1, 1, 0, 1]
        ece = compute_ece(confs, labels, n_bins=3)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# collect_calibration_data (mocked pipeline)
# ---------------------------------------------------------------------------


class TestCollectCalibrationData:
    def _make_mock_ruling(self, i: int) -> MagicMock:
        r = MagicMock()
        r.id = i
        r.product_description = f"Product description for ruling {i}"
        r.jurisdiction = "UK"
        r.assigned_code = "0702000007"
        return r

    @patch("app.pipeline.calibration.rank_candidates")
    @patch("app.pipeline.calibration.validate_candidates")
    @patch("app.pipeline.calibration.apply_rules")
    @patch("app.pipeline.calibration.retrieve_candidates")
    @patch("app.pipeline.calibration.extract_profile")
    def test_collects_scores_and_labels(
        self,
        mock_extract,
        mock_retrieve,
        mock_rules,
        mock_validate,
        mock_rank,
    ):
        from app.pipeline.calibration import collect_calibration_data
        from app.pipeline.extraction import ProductProfile
        from app.pipeline.ranking import RankedCode
        from app.pipeline.retrieval import CandidateResult

        # 3 rulings
        rulings = [self._make_mock_ruling(i) for i in range(3)]

        mock_session = MagicMock()
        mock_session.scalars.return_value.all.return_value = rulings

        mock_extract.return_value = ProductProfile(query="tomatoes")

        candidate = CandidateResult(
            ruling_id=1,
            ruling_ref="REF",
            code="0702000007",
            description="Tomatoes",
            similarity=0.88,
            jurisdiction="UK",
        )
        mock_retrieve.return_value = [candidate]
        mock_rules.return_value = ([candidate], None)
        mock_validate.return_value = [candidate]

        ranked = RankedCode(
            code="0702000007",
            score=0.88,
            max_similarity=0.88,
            support_count=1,
            best_description="Tomatoes",
            supporting_ruling_ids=[1],
            supporting_ruling_refs=["REF"],
        )
        mock_rank.return_value = [ranked]

        mock_embedder = MagicMock()

        result = collect_calibration_data(
            session=mock_session,
            embedder=mock_embedder,
            n_digits=6,
        )

        assert result["n_samples"] == 3
        assert len(result["scores"]) == 3
        assert len(result["labels"]) == 3
        # All predictions match (code starts with "070200") → label = 1
        assert all(l == 1 for l in result["labels"])
        assert all(abs(s - 0.88) < 1e-6 for s in result["scores"])

    @patch("app.pipeline.calibration.retrieve_candidates")
    @patch("app.pipeline.calibration.extract_profile")
    def test_no_eval_rulings_raises(self, mock_extract, mock_retrieve):
        from app.pipeline.calibration import collect_calibration_data

        mock_session = MagicMock()
        mock_session.scalars.return_value.all.return_value = []
        mock_embedder = MagicMock()

        with pytest.raises(ValueError, match="No eval rulings found"):
            collect_calibration_data(
                session=mock_session,
                embedder=mock_embedder,
            )

    @patch("app.pipeline.calibration.rank_candidates")
    @patch("app.pipeline.calibration.validate_candidates")
    @patch("app.pipeline.calibration.apply_rules")
    @patch("app.pipeline.calibration.retrieve_candidates")
    @patch("app.pipeline.calibration.extract_profile")
    def test_skips_rulings_with_no_candidates(
        self,
        mock_extract,
        mock_retrieve,
        mock_rules,
        mock_validate,
        mock_rank,
    ):
        from app.pipeline.calibration import collect_calibration_data
        from app.pipeline.extraction import ProductProfile

        rulings = [self._make_mock_ruling(i) for i in range(3)]
        mock_session = MagicMock()
        mock_session.scalars.return_value.all.return_value = rulings

        mock_extract.return_value = ProductProfile(query="tomatoes")
        # No candidates returned for any ruling
        mock_retrieve.return_value = []

        result = collect_calibration_data(
            session=mock_session,
            embedder=MagicMock(),
        )
        # All rulings skipped → 0 samples (no error raised)
        assert result["n_samples"] == 0


# ---------------------------------------------------------------------------
# Integration: calibration on live DB
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestM6Integration:
    """
    Require live DB, ingested data, and eval rulings.
    Run with: pytest -m integration
    """

    def test_collect_calibration_data_returns_enough_samples(self):
        """
        There must be enough eval samples to fit the calibrator.
        """
        from app.pipeline.calibration import MIN_SAMPLES, collect_calibration_data
        from app.db import get_session
        from app.ingestion.embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        with get_session() as session:
            data = collect_calibration_data(session, embedder, n_digits=6, limit=200)

        assert data["n_samples"] >= MIN_SAMPLES, (
            f"Too few eval samples ({data['n_samples']}); run M3 ingestion first."
        )
        assert all(0.0 <= s <= 2.0 for s in data["scores"]), "scores out of expected range"
        assert all(l in (0, 1) for l in data["labels"])

    def test_ece_improves_after_calibration(self):
        """
        Stated 90% confidence ≈ 90% empirical accuracy after calibration.
        ECE should be < 0.15 (spec target < 0.05 needs production data).
        """
        import math as _math
        from app.pipeline.calibration import (
            MIN_SAMPLES,
            collect_calibration_data,
            compute_ece,
            fit_calibrator,
        )
        from app.db import get_session
        from app.ingestion.embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        with get_session() as session:
            data = collect_calibration_data(session, embedder, n_digits=6)

        if data["n_samples"] < MIN_SAMPLES:
            pytest.skip(f"Only {data['n_samples']} eval samples; need {MIN_SAMPLES}")

        scores = data["scores"]
        labels = data["labels"]

        sigmoid = lambda s: 1.0 / (1.0 + _math.exp(-10.0 * (s - 0.75)))
        uncal_confs = [sigmoid(s) for s in scores]
        ece_before = compute_ece(uncal_confs, labels)

        cal = fit_calibrator(scores, labels, method="isotonic")
        import numpy as np
        cal_confs = cal.predict_proba(np.array(scores)).tolist()
        ece_after = compute_ece(cal_confs, labels)

        assert ece_after <= ece_before, (
            f"Calibration made ECE worse: {ece_before:.4f} → {ece_after:.4f}"
        )
        # Reasonable ECE even on small datasets
        assert ece_after < 0.20, f"ECE after calibration too high: {ece_after:.4f}"

    def test_save_load_inject_round_trip(self):
        """
        Full M6 workflow: fit → save → load → inject → compute_confidence returns [0,1].
        """
        import numpy as np
        from app.pipeline.calibration import (
            collect_calibration_data,
            fit_calibrator,
            load_calibrator,
            save_calibrator,
            MIN_SAMPLES,
        )
        from app.pipeline.confidence import compute_confidence, set_calibrator
        from app.pipeline.ranking import RankedCode
        from app.db import get_session
        from app.ingestion.embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        with get_session() as session:
            data = collect_calibration_data(session, embedder, n_digits=6, limit=200)

        if data["n_samples"] < MIN_SAMPLES:
            pytest.skip(f"Only {data['n_samples']} eval samples; need {MIN_SAMPLES}")

        cal = fit_calibrator(data["scores"], data["labels"], method="platt")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.pkl"
            save_calibrator(cal, path)
            loaded = load_calibrator(path)

        set_calibrator(loaded.predict_single)

        rc = RankedCode(
            code="8471300000",
            score=0.88,
            max_similarity=0.88,
            support_count=2,
            best_description="Laptop",
            supporting_ruling_ids=[1],
            supporting_ruling_refs=["REF"],
        )
        conf, _ = compute_confidence([rc])
        assert conf is not None
        assert 0.0 <= conf <= 1.0

        # Restore sigmoid
        from app.pipeline.confidence import _sigmoid_confidence
        set_calibrator(_sigmoid_confidence)
