"""RAG orchestration: chunking, retrieval and prompt building."""

import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from clients.openai_client import OpenAIClient
from repositories.knowledge import KnowledgeRepository, SearchResult
from services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

NO_INFO_MESSAGE = "На жаль, у базі знань немає інформації для відповіді на це запитання."

SYSTEM_PROMPT = (
    "Ти — корисний асистент. Відповідай українською мовою, спираючись ВИКЛЮЧНО "
    "на наданий контекст із бази знань. Якщо в контексті немає відповіді, чесно "
    "скажи, що інформації немає. Не вигадуй фактів. Відповідай стисло та структуровано."
)

# Matches explicit section markers such as "Chunk 1:" at the start of a line.
_CHUNK_MARKER = re.compile(r"^\s*Chunk\s+\d+\s*:\s*", re.IGNORECASE | re.MULTILINE)


class RagService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedding_service: EmbeddingService,
        llm_client: OpenAIClient,
        threshold: float,
        top_k: int = 3,
    ) -> None:
        self._session_factory = session_factory
        self._embeddings = embedding_service
        self._llm = llm_client
        self._threshold = threshold
        self._top_k = top_k

    # ------------------------------------------------------------------ chunking
    @staticmethod
    def split_into_chunks(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
        """Split text into chunks.

        If the text contains explicit ``Chunk N:`` markers, they define the
        section boundaries. Oversized sections (or unmarked text) are further
        split by paragraphs into windows of ``max_chars`` with ``overlap``.
        """
        if overlap >= max_chars:
            raise ValueError("overlap must be smaller than max_chars")

        text = text.replace("\r\n", "\n").strip()
        if not text:
            return []

        if _CHUNK_MARKER.search(text):
            sections = [s.strip() for s in _CHUNK_MARKER.split(text) if s.strip()]
        else:
            sections = [text]

        chunks: list[str] = []
        for section in sections:
            if len(section) <= max_chars:
                chunks.append(section)
            else:
                chunks.extend(RagService._window_split(section, max_chars, overlap))
        return chunks

    @staticmethod
    def _window_split(text: str, max_chars: int, overlap: int) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            # A single paragraph longer than max_chars is hard-split.
            while len(para) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(para[:max_chars])
                para = para[max_chars - overlap :]
            candidate = f"{current}\n{para}" if current else para
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                tail = current[-overlap:] if overlap else ""
                current = f"{tail}\n{para}" if tail else para
                if len(current) > max_chars:
                    current = para
        if current:
            chunks.append(current)
        return chunks

    # ----------------------------------------------------------------- answering
    @staticmethod
    def build_prompt(question: str, contexts: list[SearchResult]) -> str:
        context_block = "\n\n".join(
            f"[Фрагмент {i}]\n{item.content}" for i, item in enumerate(contexts, start=1)
        )
        return (
            f"Контекст із бази знань:\n{context_block}\n\n"
            f"Запитання користувача: {question}\n\n"
            "Дай відповідь на запитання, використовуючи лише наведений контекст."
        )

    async def answer(self, question: str) -> str:
        question = question.strip()
        if not question:
            return "Будь ласка, надішліть текстове запитання."

        query_vector = await self._embeddings.embed_text(question)

        async with self._session_factory() as session:
            repo = KnowledgeRepository(session)
            results = await repo.similarity_search(
                query_vector, threshold=self._threshold, limit=self._top_k
            )

        if not results:
            logger.info("No chunks within threshold %.3f; LLM is not called.", self._threshold)
            return NO_INFO_MESSAGE

        logger.info(
            "Retrieved %d chunk(s), best distance=%.4f", len(results), results[0].distance
        )
        prompt = self.build_prompt(question, results)
        return await self._llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=prompt)
