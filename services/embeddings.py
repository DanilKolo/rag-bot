"""Embedding generation via OpenAI (text-embedding-3-small, 1536 dims)."""

import asyncio
import logging
from collections.abc import Sequence

from openai import OpenAI, OpenAIError

from models import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embeddings cannot be produced."""


class EmbeddingService:
    """Wraps the synchronous OpenAI SDK; blocking calls run in a worker thread."""

    MAX_BATCH_SIZE = 100

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        timeout: float = 30.0,
        dimensions: int = EMBEDDING_DIM,
    ) -> None:
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=2)
        self._model = model
        self._dimensions = dimensions

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise EmbeddingError(
                    f"Unexpected embedding size {len(vector)}, expected {self._dimensions}"
                )
        return vectors

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        result: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.MAX_BATCH_SIZE):
                batch = list(texts[start : start + self.MAX_BATCH_SIZE])
                result.extend(await asyncio.to_thread(self._embed_sync, batch))
        except OpenAIError as exc:
            raise EmbeddingError("OpenAI embeddings request failed") from exc
        return result

    def close(self) -> None:
        self._client.close()
