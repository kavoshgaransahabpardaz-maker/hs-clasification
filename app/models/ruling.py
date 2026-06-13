from datetime import date, datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, Date, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.enums import JurisdictionType, RulingSourceType


class Ruling(Base):
    """
    Retrieval corpus AND gold evaluation labels.

    Rows where is_eval=True are held out for benchmarking (recall@k, accuracy).
    After human review, INTERNAL rulings are inserted here and embedded so
    future retrieval improves (online learning loop).
    """

    __tablename__ = "ruling"
    __table_args__ = (
        Index("ix_ruling_jurisdiction_code", "jurisdiction", "assigned_code"),
        Index("ix_ruling_is_eval", "is_eval"),
        Index("ix_ruling_reference", "reference"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(RulingSourceType, nullable=False)
    reference: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    jurisdiction: Mapped[str] = mapped_column(JurisdictionType, nullable=False)
    product_description: Mapped[str] = mapped_column(Text, nullable=False)
    assigned_code: Mapped[str] = mapped_column(String(12), nullable=False)
    justification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    keywords: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    valid_from: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    valid_to: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    # True for rows reserved for evaluation; False for retrieval corpus
    is_eval: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
