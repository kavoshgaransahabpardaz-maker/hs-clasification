"""
Pluggable embedder protocol + OpenAI concrete implementation.

Design:
  • Embedder  — structural Protocol; swap implementations without changing callers.
  • OpenAIEmbedder  — text-embedding-3-small, dim=1536, with batching + retry.

OpenAI rate limits (as of 2024):
  Tier 1: 1 M tokens/min, 3 000 req/min.
  A ruling description is ~50–200 tokens, so a batch of 256 texts ≈ 25 000 tokens.
  We stay well inside limits by default (EMBED_BATCH_SIZE=256, 0.1s sleep).
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

# Maximum texts per OpenAI embeddings call. 2048 is the hard ceiling;
# 256 keeps individual requests small and makes retry cheaper.
EMBED_BATCH_SIZE = 256


@runtime_checkable
class Embedder(Protocol):
    """
    Return one normalised float vector per input text.

    Implementations MUST:
      • Set model_name and dim as class attributes.
      • Accept empty sequences gracefully (return []).
      • Return vectors in the same order as inputs.
    """

    model_name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder:
    """
    Embedder backed by OpenAI text-embedding-3-small (1536 dimensions).

    Handles:
      • Batching — splits large inputs into EMBED_BATCH_SIZE chunks.
      • Retry    — exponential back-off on 429 / 5xx, up to max_retries attempts.
      • Ordering — sorts results by index so batches can be processed in any order.
    """

    model_name = "text-embedding-3-small"
    dim = 1536

    def __init__(self, max_retries: int = 4, inter_batch_sleep: float = 0.1) -> None:
        from openai import OpenAI

        from app.config import settings

        self._client = OpenAI(api_key=settings.openai_api_key)
        self.max_retries = max_retries
        self.inter_batch_sleep = inter_batch_sleep

    # ------------------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed *texts* and return vectors in the same order as inputs."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for batch_start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = list(texts[batch_start : batch_start + EMBED_BATCH_SIZE])
            vectors = self._embed_batch_with_retry(batch)
            all_vectors.extend(vectors)
            if batch_start + EMBED_BATCH_SIZE < len(texts):
                time.sleep(self.inter_batch_sleep)

        return all_vectors

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        backoff = 1.0
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                resp = self._client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                # API returns objects in input order but sort by index to be safe.
                ordered = sorted(resp.data, key=lambda e: e.index)
                return [e.embedding for e in ordered]
            except Exception as exc:
                last_exc = exc
                # Detect rate-limit by exception message; openai SDK raises
                # openai.RateLimitError (subclass of APIStatusError).
                msg = str(exc).lower()
                if "rate" in msg or "429" in msg or "quota" in msg:
                    sleep_for = backoff * 2
                    logger.warning(
                        "OpenAI rate limit (attempt %d/%d); sleeping %.1fs",
                        attempt + 1,
                        self.max_retries,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                else:
                    logger.warning(
                        "OpenAI embed error (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                    time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

        raise RuntimeError(
            f"OpenAI embed failed after {self.max_retries} attempts"
        ) from last_exc


def get_default_embedder() -> OpenAIEmbedder:
    """Return the configured embedder singleton (convenience factory)."""
    return OpenAIEmbedder()
