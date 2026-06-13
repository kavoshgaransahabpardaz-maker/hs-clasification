"""
Shared Python enums and SQLAlchemy enum type instances.

Each SA type is defined exactly once here and imported into every model that
needs it.  This ensures PostgreSQL creates each TYPE only once, avoiding
"type already exists" errors on create_all().
"""

import enum

from sqlalchemy import Enum as SAEnum


class Jurisdiction(str, enum.Enum):
    WCO = "WCO"
    EU = "EU"
    UK = "UK"


class NomenclatureLevel(str, enum.Enum):
    SECTION = "section"
    CHAPTER = "chapter"
    HEADING = "heading"
    SUBHEADING = "subheading"
    CN8 = "cn8"
    COMMODITY = "commodity"


class NoteScope(str, enum.Enum):
    SECTION = "section"
    CHAPTER = "chapter"


class NoteType(str, enum.Enum):
    EXCLUSION = "exclusion"
    INCLUSION = "inclusion"
    DEFINITION = "definition"
    OTHER = "other"


class RulingSource(str, enum.Enum):
    EBTI = "EBTI"
    ATAR = "ATaR"
    INTERNAL = "INTERNAL"


class ObjectType(str, enum.Enum):
    RULING = "ruling"
    NOMENCLATURE = "nomenclature"


class RequestStatus(str, enum.Enum):
    AUTO_RESOLVED = "auto_resolved"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"


# ---------------------------------------------------------------------------
# SQLAlchemy column types — import these into models, never re-create them.
# ---------------------------------------------------------------------------

JurisdictionType = SAEnum(Jurisdiction, name="jurisdiction")
NomenclatureLevelType = SAEnum(NomenclatureLevel, name="nomenclature_level")
NoteScopeType = SAEnum(NoteScope, name="note_scope")
NoteTypeType = SAEnum(NoteType, name="note_type")
RulingSourceType = SAEnum(RulingSource, name="ruling_source")
ObjectTypeType = SAEnum(ObjectType, name="object_type")
RequestStatusType = SAEnum(RequestStatus, name="request_status")
