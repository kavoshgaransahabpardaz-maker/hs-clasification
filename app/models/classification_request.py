from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.enums import JurisdictionType, RequestStatusType


class ClassificationRequest(Base):
    """
    Audit trail for every /v1/classify call.

    Also serves as training data: reviewed_code + reviewer_note are written by
    /v1/classify/{id}/review, and an INTERNAL ruling is then inserted so the
    retrieval corpus improves over time.
    """

    __tablename__ = "classification_request"

    id: Mapped[int] = mapped_column(primary_key=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_jurisdiction: Mapped[str] = mapped_column(JurisdictionType, nullable=False)

    # Structured profile extracted by the LLM (material, function, etc.)
    profile: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    predicted_code: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        RequestStatusType, nullable=False, default="needs_review"
    )

    # Full ranked candidate list: [{code, description, score, supporting_ruling_id, ...}]
    candidates: Mapped[Optional[List[Any]]] = mapped_column(JSONB, nullable=True)
    # IDs of ruling rows cited in the decision
    cited_ruling_ids: Mapped[Optional[List[Any]]] = mapped_column(JSONB, nullable=True)

    # Written by /review
    reviewed_code: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)
    reviewer_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
