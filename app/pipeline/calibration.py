"""
M6 Calibration — fit a score calibrator on the gold eval set.

Problem: the sigmoid in confidence.py gives a reasonable ordering but the
absolute values are not trustworthy (a stated 0.80 does not imply 80%
empirical accuracy).  Calibration fixes this.

Approach:
  1. Run the pipeline (extraction → retrieval → rules → validation → ranking)
     over the is_eval=True rulings, collecting pairs:
       (raw_ranking_score, is_correct_at_n_digits)
  2. Fit a calibrator that maps raw score → calibrated probability.
  3. Save the calibrator to disk.
  4. Inject it via confidence.set_calibrator() so the hot path uses it.

Three calibration methods (all from scikit-learn):
  isotonic  — IsotonicRegression, non-parametric; recommended for ≥200 samples
  platt     — Logistic regression (Platt scaling); smooth, fewer samples ok
  binned    — Simple bin-mean lookup; interpretable but coarse

ECE (Expected Calibration Error):
  ECE = Σ_b (|b|/N) × |avg_conf_in_b − accuracy_in_b|
  Lower is better.  Target: ECE < 0.05 (well-calibrated).

Usage:
    # Fit and save (run after M5 eval to have scores available)
    python -m app.pipeline.calibration

    # Specify method and accuracy definition
    python -m app.pipeline.calibration --method platt --accuracy-digits 6

    # Dry-run: fit but don't save or inject
    python -m app.pipeline.calibration --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings

# Honour CALIBRATOR_PATH env var first, then settings.
DEFAULT_CALIBRATOR_PATH = settings.calibrator_path  # type: ignore[assignment]
from app.ingestion.embedder import Embedder, OpenAIEmbedder
from app.models.ruling import Ruling
from app.pipeline.extraction import extract_profile
from app.pipeline.ranking import RankedCode, rank_candidates
from app.pipeline.retrieval import retrieve_candidates
from app.pipeline.rules import apply_rules
from app.pipeline.validation import validate_candidates

logger = logging.getLogger(__name__)

# Default path to save the fitted calibrator.
DEFAULT_CALIBRATOR_PATH = Path(os.getenv("CALIBRATOR_PATH", "data/calibrator.pkl"))

# Minimum number of eval samples required to fit a trustworthy calibrator.
MIN_SAMPLES = 50


# ---------------------------------------------------------------------------
# Calibrator wrapper
# ---------------------------------------------------------------------------


@dataclass
class Calibrator:
    """
    Wraps a fitted sklearn estimator and exposes a float→float interface
    compatible with confidence.set_calibrator().
    """

    method: str
    n_digits: int  # accuracy definition used during fitting
    n_train_samples: int
    _estimator: object = field(repr=False)

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for an array of raw ranking scores."""
        est = self._estimator
        scores = np.asarray(scores, dtype=float).reshape(-1)

        if self.method == "isotonic":
            return np.clip(est.predict(scores), 0.0, 1.0)  # type: ignore[union-attr]
        elif self.method == "platt":
            return est.predict_proba(scores.reshape(-1, 1))[:, 1]  # type: ignore[union-attr]
        elif self.method == "binned":
            # _estimator is a dict: {bin_index: mean_accuracy}
            bins: np.ndarray = est["bins"]  # type: ignore[index]
            values: np.ndarray = est["values"]
            idx = np.digitize(scores, bins) - 1
            idx = np.clip(idx, 0, len(values) - 1)
            return values[idx]
        else:
            raise ValueError(f"Unknown calibration method: {self.method!r}")

    def predict_single(self, score: float) -> float:
        """Single-score interface; used by confidence.set_calibrator()."""
        return float(self.predict_proba(np.array([score]))[0])

    def __call__(self, score: float) -> float:  # convenience alias
        return self.predict_single(score)


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------


def compute_ece(
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.

    Args:
        confidences: predicted probability values ∈ [0, 1].
        labels:      binary ground truth (1 = correct, 0 = incorrect).
        n_bins:      number of equal-width bins.

    Returns:
        ECE ∈ [0, 1].  Lower is better.  Target < 0.05.
    """
    confs = np.asarray(confidences, dtype=float)
    labs = np.asarray(labels, dtype=float)
    n = len(confs)
    if n == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi) if i < n_bins - 1 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = labs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


def calibration_bins(
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> list[dict]:
    """
    Return per-bin stats for a calibration report / plot.

    Each entry: {"bin_lo": float, "bin_hi": float, "count": int,
                 "avg_conf": float, "accuracy": float}
    """
    confs = np.asarray(confidences, dtype=float)
    labs = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi) if i < n_bins - 1 else (confs >= lo) & (confs <= hi)
        count = int(mask.sum())
        rows.append({
            "bin_lo": round(float(lo), 2),
            "bin_hi": round(float(hi), 2),
            "count": count,
            "avg_conf": round(float(confs[mask].mean()), 4) if count > 0 else None,
            "accuracy": round(float(labs[mask].mean()), 4) if count > 0 else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Calibration data collection
# ---------------------------------------------------------------------------


def collect_calibration_data(
    session: Session,
    embedder: Embedder,
    n_digits: int = 6,
    limit: int | None = None,
) -> dict:
    """
    Run the pipeline (steps 1–5) over is_eval=True rulings and collect
    (raw_ranking_score, is_correct) pairs.

    Args:
        session:   DB session.
        embedder:  Embedder instance.
        n_digits:  Number of leading digits that must match for "correct".
        limit:     Max eval rulings to process (None = all).

    Returns:
        dict with keys: "scores" (list[float]), "labels" (list[int]),
        "n_samples" (int), "n_digits" (int).
    """
    query = select(Ruling).where(Ruling.is_eval.is_(True))
    if limit:
        query = query.limit(limit)
    eval_rulings = session.scalars(query).all()

    if not eval_rulings:
        raise ValueError("No eval rulings found. Run ingestion first.")

    logger.info(
        "Collecting calibration data from %d eval rulings (n_digits=%d)",
        len(eval_rulings), n_digits,
    )

    scores: list[float] = []
    labels: list[int] = []

    for i, ruling in enumerate(eval_rulings):
        try:
            profile = extract_profile(ruling.product_description)
            query_text = profile.query or ruling.product_description[:300]

            raw_candidates = retrieve_candidates(
                query=query_text,
                jurisdiction=str(ruling.jurisdiction),
                session=session,
                embedder=embedder,
                top_k=settings.retrieval_top_k,
                exclude_eval=True,
            )
            if not raw_candidates:
                continue

            pruned, _ = apply_rules(profile, raw_candidates, str(ruling.jurisdiction), session)
            valid = validate_candidates(pruned, str(ruling.jurisdiction), session)
            ranked = rank_candidates(valid)

            if not ranked:
                continue

            raw_score = ranked[0].score
            predicted_code = ranked[0].code
            is_correct = int(
                ruling.assigned_code[:n_digits] == predicted_code[:n_digits]
            )

            scores.append(raw_score)
            labels.append(is_correct)

        except Exception as exc:
            logger.warning("Calibration data: error on ruling %d: %s", ruling.id, exc)

        if (i + 1) % 20 == 0:
            n = len(scores)
            acc = sum(labels) / n if n > 0 else 0
            logger.info(
                "Calibration data: %d/%d processed — accuracy@%dd so far %.1f%%",
                i + 1, len(eval_rulings), n_digits, acc * 100,
            )

    logger.info(
        "Calibration data collected: %d samples, accuracy@%dd=%.1f%%",
        len(scores), n_digits,
        100 * sum(labels) / max(len(labels), 1),
    )
    return {"scores": scores, "labels": labels, "n_samples": len(scores), "n_digits": n_digits}


# ---------------------------------------------------------------------------
# Calibrator fitting
# ---------------------------------------------------------------------------


def fit_calibrator(
    scores: list[float],
    labels: list[int],
    method: str = "isotonic",
    n_bins: int = 10,
) -> Calibrator:
    """
    Fit a calibrator on (score, label) pairs.

    Args:
        scores:  Raw ranking scores from rank_candidates()[0].score.
        labels:  Binary correctness labels (1 = correct, 0 = wrong).
        method:  "isotonic" | "platt" | "binned".
        n_bins:  Number of bins for method="binned".

    Returns:
        Fitted Calibrator instance.
    """
    X = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)

    if len(X) < MIN_SAMPLES:
        raise ValueError(
            f"Only {len(X)} samples — need at least {MIN_SAMPLES} to fit a "
            "reliable calibrator.  Load more rulings or reduce --min-samples."
        )

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        est = IsotonicRegression(out_of_bounds="clip")
        est.fit(X, y)

    elif method == "platt":
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(C=1.0, max_iter=1000)
        est.fit(X.reshape(-1, 1), y)

    elif method == "binned":
        bins_edges = np.linspace(X.min(), X.max(), n_bins + 1)
        values = np.zeros(n_bins)
        for i in range(n_bins):
            lo, hi = bins_edges[i], bins_edges[i + 1]
            mask = (X >= lo) & (X < hi) if i < n_bins - 1 else (X >= lo) & (X <= hi)
            values[i] = y[mask].mean() if mask.sum() > 0 else 0.5
        est = {"bins": bins_edges[1:], "values": values}

    else:
        raise ValueError(f"Unknown calibration method: {method!r}. Choose isotonic|platt|binned.")

    return Calibrator(
        method=method,
        n_digits=0,  # set by caller
        n_train_samples=len(X),
        _estimator=est,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_calibrator(cal: Calibrator, path: Path = DEFAULT_CALIBRATOR_PATH) -> None:
    """Pickle the calibrator to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cal, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Calibrator saved to %s", path)


def load_calibrator(path: Path = DEFAULT_CALIBRATOR_PATH) -> Calibrator:
    """Load a previously fitted calibrator from disk."""
    with open(path, "rb") as f:
        cal: Calibrator = pickle.load(f)
    logger.info(
        "Calibrator loaded from %s (method=%s, n_train=%d)",
        path, cal.method, cal.n_train_samples,
    )
    return cal


def load_calibrator_if_exists(
    path: Path = DEFAULT_CALIBRATOR_PATH,
) -> Calibrator | None:
    """Return a loaded Calibrator, or None if the file does not exist."""
    if not path.exists():
        logger.debug("No calibrator file found at %s — using uncalibrated sigmoid.", path)
        return None
    try:
        return load_calibrator(path)
    except Exception as exc:
        logger.warning("Could not load calibrator from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Full calibration run
# ---------------------------------------------------------------------------


def run_calibration(
    session: Session,
    embedder: Embedder | None = None,
    method: str = "isotonic",
    n_digits: int = 6,
    limit: int | None = None,
    dry_run: bool = False,
    save_path: Path = DEFAULT_CALIBRATOR_PATH,
) -> dict:
    """
    Collect calibration data, fit calibrator, compute ECE, optionally save.

    Returns a dict with keys:
      n_samples, ece_before, ece_after, calibrator (Calibrator instance)
    """
    from app.pipeline.confidence import set_calibrator
    import math

    if embedder is None:
        embedder = OpenAIEmbedder()

    # 1. Collect (score, label) pairs
    data = collect_calibration_data(session, embedder, n_digits=n_digits, limit=limit)
    scores = data["scores"]
    labels = data["labels"]

    if len(scores) < MIN_SAMPLES:
        raise ValueError(
            f"Only {len(scores)} eval samples — need ≥{MIN_SAMPLES}. "
            "Load more rulings (M3) and re-run."
        )

    # 2. Compute ECE before calibration (using current sigmoid)
    import math as _math
    sigmoid = lambda s: 1.0 / (1.0 + _math.exp(-10.0 * (s - 0.75)))
    uncal_confs = [sigmoid(s) for s in scores]
    ece_before = compute_ece(uncal_confs, labels)

    # 3. Fit calibrator
    cal = fit_calibrator(scores, labels, method=method)
    cal.n_digits = n_digits

    # 4. Compute ECE after calibration
    cal_confs = cal.predict_proba(np.array(scores)).tolist()
    ece_after = compute_ece(cal_confs, labels)

    logger.info(
        "Calibration (%s): ECE before=%.4f → after=%.4f (%+.4f)",
        method, ece_before, ece_after, ece_after - ece_before,
    )

    # 5. Save and inject (unless dry-run)
    if not dry_run:
        save_calibrator(cal, save_path)
        set_calibrator(cal.predict_single)
        logger.info("Calibrator injected into confidence pipeline.")
    else:
        logger.info("Dry-run: calibrator NOT saved or injected.")

    return {
        "n_samples": len(scores),
        "n_digits": n_digits,
        "method": method,
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "calibrator": cal,
        "cal_bins": calibration_bins(cal_confs, labels),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_calibration_report(result: dict) -> None:
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  M6 Calibration Report — {result['method']}")
    print(sep)
    print(f"  Samples : {result['n_samples']}")
    print(f"  Accuracy definition : top-1 code @ {result['n_digits']} digits")
    print(f"  ECE before : {result['ece_before']:.4f}")
    print(f"  ECE after  : {result['ece_after']:.4f}  {'✓' if result['ece_after'] < 0.05 else '⚠ target < 0.05'}")
    print(f"\n  Calibration bins (conf_mid | accuracy | count):")
    for b in result["cal_bins"]:
        if b["count"] == 0:
            continue
        mid = (b["bin_lo"] + b["bin_hi"]) / 2
        bar_conf = "█" * int(mid * 20)
        bar_acc = "░" * int((b["accuracy"] or 0) * 20)
        diff = (b["accuracy"] or 0) - mid
        sign = "+" if diff >= 0 else ""
        print(
            f"  [{b['bin_lo']:.1f}–{b['bin_hi']:.1f}]  conf={mid:.2f}  "
            f"acc={b['accuracy']:.2f}  ({sign}{diff:.2f})  n={b['count']}"
        )
    print(f"{sep}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="M6 confidence calibration")
    parser.add_argument(
        "--method", choices=["isotonic", "platt", "binned"], default="isotonic",
        help="Calibration method (default: isotonic)",
    )
    parser.add_argument(
        "--accuracy-digits", type=int, default=6, dest="n_digits",
        help="Number of leading digits for 'correct' label (default: 6)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max eval samples to use",
    )
    parser.add_argument(
        "--save-path", type=Path, default=DEFAULT_CALIBRATOR_PATH,
        help=f"Where to save the calibrator (default: {DEFAULT_CALIBRATOR_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fit but do not save or inject the calibrator",
    )
    args = parser.parse_args()

    from app.db import get_session, init_db

    init_db()
    embedder = OpenAIEmbedder()

    with get_session() as session:
        result = run_calibration(
            session,
            embedder=embedder,
            method=args.method,
            n_digits=args.n_digits,
            limit=args.limit,
            dry_run=args.dry_run,
            save_path=args.save_path,
        )

    _print_calibration_report(result)


if __name__ == "__main__":
    main()
    sys.exit(0)
