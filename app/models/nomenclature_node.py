from __future__ import annotations

from datetime import date
from typing import List, Optional

from sqlalchemy import Date, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.enums import JurisdictionType, NomenclatureLevelType


class NomenclatureNode(Base):
    """
    The HS/CN/TARIC code tree for both UK and EU.

    path  — materialized ancestor path, e.g. "84.8471.847130", used for
            LIKE 'prefix%' subtree queries.  The text_pattern_ops index
            supports this efficiently.
    """

    __tablename__ = "nomenclature_node"
    __table_args__ = (
        Index("ix_nomenclature_jurisdiction_code", "jurisdiction", "code"),
        Index("ix_nomenclature_jurisdiction_level", "jurisdiction", "level"),
        # text_pattern_ops enables prefix LIKE queries on path
        Index(
            "ix_nomenclature_path_pattern",
            "path",
            postgresql_ops={"path": "text_pattern_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False)
    level: Mapped[str] = mapped_column(NomenclatureLevelType, nullable=False)
    jurisdiction: Mapped[str] = mapped_column(JurisdictionType, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("nomenclature_node.id", ondelete="SET NULL"), nullable=True
    )
    # materialized path, e.g. "84.8471.847130"
    path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    valid_from: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    valid_to: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    parent: Mapped[Optional[NomenclatureNode]] = relationship(
        "NomenclatureNode",
        remote_side="NomenclatureNode.id",
        back_populates="children",
    )
    children: Mapped[List[NomenclatureNode]] = relationship(
        "NomenclatureNode",
        back_populates="parent",
        cascade="all, delete-orphan",
    )
