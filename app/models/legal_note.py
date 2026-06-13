from sqlalchemy import Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.enums import JurisdictionType, NoteScopeType, NoteTypeType


class LegalNote(Base):
    """
    Section and chapter notes from the HS/CN nomenclature regulations.

    These drive the rules layer: exclusion notes eliminate impossible candidates
    before GRI tie-breaking.
    """

    __tablename__ = "legal_note"
    __table_args__ = (
        Index("ix_legal_note_jurisdiction_scope_code", "jurisdiction", "scope", "scope_code"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    jurisdiction: Mapped[str] = mapped_column(JurisdictionType, nullable=False)
    # scope_code: 2-digit chapter (e.g. "84") or 1-2 digit section number (e.g. "16")
    scope: Mapped[str] = mapped_column(NoteScopeType, nullable=False)
    scope_code: Mapped[str] = mapped_column(String(4), nullable=False)
    note_type: Mapped[str] = mapped_column(NoteTypeType, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
