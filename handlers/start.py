"""/start command handler."""

import logging

from aiogram import Router, html
from aiogram.filters import CommandStart
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from repositories.users import UserRepository

logger = logging.getLogger(__name__)
router = Router(name="start")


@router.message(CommandStart())
async def cmd_start(message: Message, session_factory: async_sessionmaker[AsyncSession]) -> None:
    tg_user = message.from_user
    if tg_user is None:
        return

    try:
        async with session_factory() as session:
            await UserRepository(session).upsert_user(
                telegram_id=tg_user.id,
                username=tg_user.username,
                first_name=tg_user.first_name,
            )
    except Exception:
        logger.exception("Failed to upsert user telegram_id=%s", tg_user.id)
        # Registration failure should not block the greeting.

    name = html.quote(tg_user.first_name or tg_user.username or "друже")
    await message.answer(
        f"Привіт, {html.bold(name)}! 👋\n\n"
        "Я бот-асистент з базою знань. Надішліть мені запитання текстом, "
        "і я спробую знайти відповідь."
    )
