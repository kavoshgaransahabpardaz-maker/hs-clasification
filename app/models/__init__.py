"""
Import all models here so that:
  1. Alembic's env.py only needs `from app.models import Base`.
  2. create_all() sees every table in one call.
"""

from app.models.base import Base
from app.models.classification_request import ClassificationRequest
from app.models.embedding import Embedding
from app.models.eval_run import EvalRun
from app.models.legal_note import LegalNote
from app.models.nomenclature_node import NomenclatureNode
from app.models.ruling import Ruling

__all__ = [
    "Base",
    "ClassificationRequest",
    "Embedding",
    "EvalRun",
    "LegalNote",
    "NomenclatureNode",
    "Ruling",
]
