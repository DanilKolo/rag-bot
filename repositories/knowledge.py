"""Knowledge base (vector store) persistence."""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import KnowledgeEmbedding


@dataclass(frozen=True, slots=True)
class SearchResult:
    content: str
    distance: float


class KnowledgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        threshold: float,
        limit: int = 3,
    ) -> list[SearchResult]:
        """Return up to ``limit`` nearest chunks by cosine distance (``<=>``).

        Only chunks with ``distance <= threshold`` are returned, so an empty
        list means nothing in the knowledge base is relevant enough.
        """
        distance = KnowledgeEmbedding.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(KnowledgeEmbedding.content, distance)
            .where(distance <= threshold)
            .order_by(distance)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [SearchResult(content=row.content, distance=float(row.distance)) for row in result]

    async def clear(self) -> None:
        """Delete all knowledge chunks (does not touch any other table)."""
        await self._session.execute(delete(KnowledgeEmbedding))

    async def add_many(self, items: Sequence[tuple[str, Sequence[float]]]) -> None:
        self._session.add_all(
            KnowledgeEmbedding(content=content, embedding=list(embedding)) for content, embedding in items
        )
        await self._session.flush()

    async def count(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(KnowledgeEmbedding))
        return int(result.scalar_one())

    async def commit(self) -> None:
        await self._session.commit()
