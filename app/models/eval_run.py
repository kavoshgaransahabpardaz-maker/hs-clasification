from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class EvalRun(Base):
    """
    Benchmark history written by `python -m app.eval`.

    Keeps a permanent record so regressions are visible across pipeline
    versions.  A change is only "done" when eval doesn't regress.
    """

    __tablename__ = "eval_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)

    # {"k1": 0.72, "k5": 0.91, "k10": 0.95}
    recall_at_k: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    # {"2": 0.91, "4": 0.85, "6": 0.78, "8": 0.70, "10": 0.65}
    accuracy_by_digit: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    # Expected Calibration Error — populated after M6 calibration
    ece: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
