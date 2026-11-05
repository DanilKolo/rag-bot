"""User persistence."""

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from models import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_user(
        self,
        telegram_id: int,
        username: str | None,
        first_name: str | None,
    ) -> User:
        """Insert a user or update username/first_name if it already exists."""
        stmt = (
            insert(User)
            .values(telegram_id=telegram_id, username=username, first_name=first_name)
            .on_conflict_do_update(
                index_elements=[User.telegram_id],
                set_={"username": username, "first_name": first_name},
            )
            .returning(User)
        )
        result = await self._session.execute(stmt)
        user = result.scalar_one()
        await self._session.commit()
        return user
