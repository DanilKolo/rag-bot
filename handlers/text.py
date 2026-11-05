"""Free-text question handler (RAG)."""

import logging

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from services.rag import RagService

logger = logging.getLogger(__name__)
router = Router(name="text")

SAFE_ERROR_MESSAGE = "Вибачте, сталася помилка під час обробки запиту. Спробуйте, будь ласка, пізніше."
TELEGRAM_MAX_LEN = 4096
MAX_QUESTION_LEN = 2000


@router.message(F.text & ~F.text.startswith("/"))
async def handle_question(message: Message, rag_service: RagService) -> None:
    question = (message.text or "").strip()
    if len(question) > MAX_QUESTION_LEN:
        await message.answer(f"Запитання задовге. Максимум — {MAX_QUESTION_LEN} символів.")
        return

    try:
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        answer = await rag_service.answer(question)
    except Exception:
        logger.exception("Failed to answer question from user_id=%s", message.from_user and message.from_user.id)
        await message.answer(SAFE_ERROR_MESSAGE)
        return

    # LLM output is sent as plain text (parse_mode=None) so arbitrary "<", "&" etc.
    # can never break Telegram's HTML parser; long answers are split into parts.
    for start in range(0, len(answer), TELEGRAM_MAX_LEN):
        await message.answer(answer[start : start + TELEGRAM_MAX_LEN], parse_mode=None)
