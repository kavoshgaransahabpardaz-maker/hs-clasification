"""
Extraction step — LLM turns raw product text into a structured profile.

The profile drives retrieval (via the clean `query` field) and the rules layer
(via material, function, form_state, etc.).  It is also stored on the
ClassificationRequest row for audit / human review.

Error policy: any LLM or JSON failure falls back to a minimal profile whose
`query` field is the raw text (truncated), so retrieval still runs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, field_validator

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profile schema
# ---------------------------------------------------------------------------

PROFILE_FIELDS = (
    "material",
    "function",
    "form_state",
    "intended_use",
    "components",
    "processing_level",
    "query",
)


class ProductProfile(BaseModel):
    """
    Structured representation of a product description.

    All fields default to empty; only `query` is required for retrieval.
    `query` is validated to always be non-empty.
    """

    material: str = ""
    function: str = ""
    form_state: str = ""
    intended_use: str = ""
    components: list[str] = []
    processing_level: str = ""
    # Clean, concise retrieval query (2–10 words)
    query: str = ""

    @field_validator("components", mode="before")
    @classmethod
    def coerce_components(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v] if v else []
        if isinstance(v, list):
            return [str(x) for x in v]
        return []

    @field_validator("query", mode="after")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        # Validator receives a value that may already be stripped.
        return v.strip()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a customs classification expert assisting with HS / CN tariff code assignment.
Given a product description, extract structured attributes that aid retrieval and classification.

Return ONLY valid JSON with exactly these keys (no extra keys, no markdown):
{
  "material":         "primary material, composition, or chemical family",
  "function":         "what the product does or its main technical function",
  "form_state":       "solid | liquid | gas | powder | mixture | article | preparation | other",
  "intended_use":     "end application, industry sector, or end user",
  "components":       ["list", "of", "main", "components or ingredients"],
  "processing_level": "raw | semi-finished | finished | manufactured",
  "query":            "2-8 word retrieval query optimised for customs ruling similarity search"
}
All fields are required. Use an empty string or empty list if the information is not present.
Do NOT include any text outside the JSON object.\
"""


# ---------------------------------------------------------------------------
# Extraction function
# ---------------------------------------------------------------------------


def extract_profile(text: str) -> ProductProfile:
    """
    Call GPT-4o to extract a structured profile from *text*.

    Falls back to `ProductProfile(query=text[:300])` on any error so the
    retrieval step can always proceed.
    """
    if not text or not text.strip():
        return ProductProfile(query="")

    # Truncate to avoid excess token spend on pathologically long inputs.
    truncated = text.strip()[:4000]

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": truncated},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=512,
        )
        raw_json = response.choices[0].message.content or "{}"
        data: dict[str, Any] = json.loads(raw_json)

        profile = ProductProfile.model_validate(data)

        # If the model omitted the query field, fall back to the raw text.
        if not profile.query:
            profile = profile.model_copy(update={"query": truncated[:300]})

        return profile

    except json.JSONDecodeError as exc:
        logger.warning("Extraction: JSON parse error — %s", exc)
    except Exception as exc:
        logger.warning("Extraction: LLM call failed — %s", exc)

    # Fallback profile
    return ProductProfile(query=truncated[:300])
