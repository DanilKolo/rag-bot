"""Load knowledge_base.txt into the vector store.

Steps: read file -> chunk -> embed (1536-dim, text-embedding-3-small) -> save.

Only the ``knowledge_embeddings`` table is refreshed. The ``users`` table is
never dropped or modified. Embeddings are generated BEFORE the old knowledge
is removed, and replacement happens in a single transaction, so a failure
leaves the previous knowledge base intact.

Usage:
    python init_db.py [path/to/knowledge_base.txt]
"""

import asyncio
import logging
import sys
from pathlib import Path

from config import get_settings
from db import create_engine, create_session_factory, init_models
from models import EMBEDDING_DIM
from repositories.knowledge import KnowledgeRepository
from services.embeddings import EmbeddingService
from services.rag import RagService

logger = logging.getLogger("init_db")

DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_base.txt"


async def load_knowledge_base(kb_path: Path) -> int:
    settings = get_settings()

    if not kb_path.is_file():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    raw_text = kb_path.read_text(encoding="utf-8-sig")
    chunks = RagService.split_into_chunks(raw_text)
    if not chunks:
        raise ValueError(f"Knowledge base file is empty: {kb_path}")
    logger.info("Split %s into %d chunk(s).", kb_path.name, len(chunks))

    embedding_service = EmbeddingService(
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        model=settings.OPENAI_EMBEDDING_MODEL,
        timeout=settings.OPENAI_TIMEOUT,
        dimensions=EMBEDDING_DIM,
    )
    engine = create_engine()
    try:
        vectors = await embedding_service.embed_texts(chunks)
        logger.info("Generated %d embedding(s) of dim %d.", len(vectors), EMBEDDING_DIM)

        await init_models(engine)  # CREATE EXTENSION + create missing tables only
        session_factory = create_session_factory(engine)

        async with session_factory() as session:
            repo = KnowledgeRepository(session)
            try:
                await repo.clear()
                await repo.add_many(list(zip(chunks, vectors, strict=True)))
                await repo.commit()
            except Exception:
                await session.rollback()
                raise
            total = await repo.count()
        logger.info("Knowledge base saved: %d chunk(s) in database.", total)
        return total
    finally:
        embedding_service.close()
        await engine.dispose()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    kb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_KB_PATH
    try:
        asyncio.run(load_knowledge_base(kb_path))
    except Exception:
        logger.exception("Knowledge base initialisation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
