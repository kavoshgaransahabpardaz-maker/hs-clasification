"""
Evaluation harness.

    python -m app.eval
    python -m app.eval --jurisdiction eu --limit 200
    python -m app.eval --full-pipeline   # include rules+validation (slower)

Runs the pipeline over ruling rows with is_eval=True, reports:
  • recall@k   (k = 1, 5, 10) — primary metric; optimise retrieval first
  • accuracy@n (n = 2, 4, 6, 8, 10 digits) — per-digit accuracy of top-1 code
  • ECE        — Expected Calibration Error (reported after M6 calibration)

Writes one eval_run row to the DB so benchmark history is preserved.
A change is only "done" if eval doesn't regress.

Note on eval filtering:
  Retrieval runs with exclude_eval=True so eval rulings are never their
  own nearest neighbours (prevents data leakage into recall metrics).
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_session, init_db
from app.ingestion.embedder import Embedder, OpenAIEmbedder
from app.models.eval_run import EvalRun
from app.models.ruling import Ruling
from app.pipeline.extraction import extract_profile
from app.pipeline.retrieval import retrieve_candidates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _recall_at_k(target_code: str, candidate_codes: list[str], k: int) -> bool:
    return target_code in candidate_codes[:k]


def _accuracy_at_n_digits(target_code: str, predicted_code: str | None, n: int) -> bool:
    if not predicted_code:
        return False
    return target_code[:n] == predicted_code[:n]


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------


def run_eval(
    session: Session,
    embedder: Embedder | None = None,
    jurisdiction: str | None = None,
    limit: int | None = None,
    top_k: int = 10,
    full_pipeline: bool = False,
) -> dict:
    """
    Run the evaluation loop over is_eval=True rulings.

    Args:
        full_pipeline: If True, run rules+validation+ranking steps
                       (slower but tests the complete M5 pipeline and collects
                       raw ranking scores for ECE computation).
                       If False, recall@k and accuracy@n are computed from
                       retrieval alone (fast, M4-style).

    Returns dict with: recall_at_k, accuracy_by_digit, n_samples, ece (or None).
    """
    if embedder is None:
        embedder = OpenAIEmbedder()

    query = select(Ruling).where(Ruling.is_eval.is_(True))
    if jurisdiction:
        query = query.where(Ruling.jurisdiction == jurisdiction.upper())
    if limit:
        query = query.limit(limit)

    eval_rulings = session.scalars(query).all()
    n = len(eval_rulings)

    if n == 0:
        raise ValueError(
            "No eval rulings found.  Run `python -m app.ingestion` first."
        )

    logger.info(
        "Evaluating %d rulings (jurisdiction=%s, full_pipeline=%s)",
        n, jurisdiction or "all", full_pipeline,
    )

    recalls: dict[int, int] = {1: 0, 5: 0, 10: 0}
    accuracy: dict[int, int] = {2: 0, 4: 0, 6: 0, 8: 0, 10: 0}

    # For ECE: collect (confidence, is_correct_at_6_digits) pairs
    confidences: list[float] = []
    correct_labels: list[int] = []

    for i, ruling in enumerate(eval_rulings):
        try:
            profile = extract_profile(ruling.product_description)
            query_text = profile.query or ruling.product_description[:300]
        except Exception as exc:
            logger.warning("Extraction failed for ruling %d: %s", ruling.id, exc)
            query_text = ruling.product_description[:300]

        candidates = retrieve_candidates(
            query=query_text,
            jurisdiction=str(ruling.jurisdiction),
            session=session,
            embedder=embedder,
            top_k=top_k,
            exclude_eval=True,
        )

        if full_pipeline and candidates:
            # Run M5 steps to get ranked codes and calibrated confidence.
            from app.pipeline.rules import apply_rules
            from app.pipeline.validation import validate_candidates
            from app.pipeline.ranking import rank_candidates
            from app.pipeline.confidence import compute_confidence

            pruned, _ = apply_rules(profile, candidates, str(ruling.jurisdiction), session)
            valid = validate_candidates(pruned, str(ruling.jurisdiction), session)
            ranked = rank_candidates(valid)
            confidence, _ = compute_confidence(ranked)

            candidate_codes = [r.code for r in ranked]
            predicted_code = candidate_codes[0] if candidate_codes else None

            if confidence is not None:
                is_correct_6d = int(
                    (ruling.assigned_code[:6] == (predicted_code or "")[:6])
                )
                confidences.append(confidence)
                correct_labels.append(is_correct_6d)
        else:
            candidate_codes = [c.code for c in candidates]
            predicted_code = candidate_codes[0] if candidate_codes else None

        for k in (1, 5, 10):
            if _recall_at_k(ruling.assigned_code, candidate_codes, k):
                recalls[k] += 1

        for n_digits in (2, 4, 6, 8, 10):
            if _accuracy_at_n_digits(ruling.assigned_code, predicted_code, n_digits):
                accuracy[n_digits] += 1

        if (i + 1) % 20 == 0 or (i + 1) == n:
            logger.info(
                "Eval: %d/%d — recall@1=%.2f recall@10=%.2f",
                i + 1, n,
                recalls[1] / (i + 1),
                recalls[10] / (i + 1),
            )

    # Compute ECE if we have confidence data (full_pipeline mode).
    ece: float | None = None
    if confidences:
        from app.pipeline.calibration import compute_ece

        ece = round(compute_ece(confidences, correct_labels), 6)
        logger.info(
            "ECE (n=%d, 6-digit accuracy): %.4f  %s",
            len(confidences),
            ece,
            "✓ well-calibrated" if ece < 0.05 else "⚠ target < 0.05",
        )

    return {
        "n_samples": n,
        "recall_at_k": {f"k{k}": round(recalls[k] / n, 4) for k in recalls},
        "accuracy_by_digit": {
            str(nd): round(accuracy[nd] / n, 4) for nd in sorted(accuracy)
        },
        "ece": ece,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_eval_run(session: Session, metrics: dict) -> EvalRun:
    """Write an eval_run row so benchmark history is preserved."""
    row = EvalRun(
        pipeline_version=settings.pipeline_version,
        n_samples=metrics["n_samples"],
        recall_at_k=metrics["recall_at_k"],
        accuracy_by_digit=metrics["accuracy_by_digit"],
        ece=metrics.get("ece"),
        notes=metrics.get("notes"),
    )
    session.add(row)
    session.flush()
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(metrics: dict) -> None:
    sep = "\u2500" * 52
    print(f"\n{sep}")
    print(f"  Pipeline v{settings.pipeline_version}  |  {metrics['n_samples']} eval samples")
    print(sep)

    print("\nRecall@K (ceiling on the pipeline):")
    for k_label, v in metrics["recall_at_k"].items():
        bar = "\u2588" * int(v * 20) + "\u2591" * (20 - int(v * 20))
        print(f"  {k_label:>4}  {bar}  {v:.1%}")

    print("\nAccuracy by digit match (top-1 code):")
    for n_label, v in metrics["accuracy_by_digit"].items():
        bar = "\u2588" * int(v * 20) + "\u2591" * (20 - int(v * 20))
        print(f"  {n_label:>2}d   {bar}  {v:.1%}")

    if metrics.get("ece") is not None:
        ece = metrics["ece"]
        status = "\u2713 well-calibrated" if ece < 0.05 else "\u26a0  target ECE < 0.05"
        print(f"\nECE (6-digit accuracy): {ece:.4f}  {status}")

    print(f"\n{sep}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="HS Classification evaluation harness")
    parser.add_argument(
        "--jurisdiction", choices=["eu", "uk"], default=None,
        help="Restrict to one jurisdiction (default: both)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max eval rulings to process",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Retrieval top-K (default: 10)",
    )
    parser.add_argument(
        "--full-pipeline", action="store_true",
        help="Run rules+validation+ranking+confidence (enables ECE computation)",
    )
    parser.add_argument(
        "--no-persist", action="store_true",
        help="Skip writing an eval_run row to the DB",
    )
    args = parser.parse_args()

    init_db()

    # Load calibrator if one has been fitted (M6).
    from app.pipeline.calibration import load_calibrator_if_exists
    from app.pipeline.confidence import set_calibrator

    cal = load_calibrator_if_exists()
    if cal:
        set_calibrator(cal.predict_single)
        logger.info("Loaded calibrator (method=%s, n_train=%d)", cal.method, cal.n_train_samples)

    embedder = OpenAIEmbedder()

    with get_session() as session:
        metrics = run_eval(
            session,
            embedder=embedder,
            jurisdiction=args.jurisdiction,
            limit=args.limit,
            top_k=args.top_k,
            full_pipeline=args.full_pipeline,
        )

        if not args.no_persist:
            row = persist_eval_run(session, metrics)
            logger.info("eval_run row written (id=%d)", row.id)

    _print_report(metrics)


if __name__ == "__main__":
    main()
    sys.exit(0)
