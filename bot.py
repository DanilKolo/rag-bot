"""Application entry point: wires dependencies and starts polling."""

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import ErrorEvent

from clients.openai_client import OpenAIClient
from config import get_settings
from db import create_engine, create_session_factory, init_models
from handlers import start, text
from services.embeddings import EmbeddingService
from services.rag import RagService

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    # Avoid leaking request details (incl. headers) at DEBUG level from HTTP libs.
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    # --- Infrastructure ---
    engine = create_engine()
    await init_models(engine)
    session_factory = create_session_factory(engine)

    # --- Singletons, created exactly once ---
    openai_key = settings.OPENAI_API_KEY.get_secret_value()
    embedding_service = EmbeddingService(
        api_key=openai_key,
        model=settings.OPENAI_EMBEDDING_MODEL,
        timeout=settings.OPENAI_TIMEOUT,
    )
    llm_client = OpenAIClient(
        api_key=openai_key,
        model=settings.OPENAI_CHAT_MODEL,
        timeout=settings.OPENAI_TIMEOUT,
    )
    rag_service = RagService(
        session_factory=session_factory,
        embedding_service=embedding_service,
        llm_client=llm_client,
        threshold=settings.RAG_THRESHOLD,
        top_k=settings.RAG_TOP_K,
    )

    bot = Bot(
        token=settings.BOT_TOKEN.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    # --- Dependency Injection: values in workflow_data are passed to handlers by name ---
    dp = Dispatcher(
        session_factory=session_factory,
        embedding_service=embedding_service,
        llm_client=llm_client,
        rag_service=rag_service,
    )
    dp.include_routers(start.router, text.router)

    @dp.errors()
    async def global_error_handler(event: ErrorEvent) -> bool:
        logger.exception("Unhandled error while processing update", exc_info=event.exception)
        message = event.update.message
        if message is not None:
            try:
                await message.answer("Вибачте, сталася непередбачена помилка. Спробуйте пізніше.")
            except Exception:
                logger.exception("Failed to send error notification to user")
        return True

    logger.info("Bot is starting (RAG threshold=%.3f)...", settings.RAG_THRESHOLD)
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        await llm_client.close()
        embedding_service.close()
        await bot.session.close()
        await engine.dispose()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
