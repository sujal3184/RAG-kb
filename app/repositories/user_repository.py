"""Repository for the User table.

Adds user-specific queries on top of the generic CRUD from BaseRepository.
"""

from sqlalchemy import select

from app.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Data access methods for User rows."""

    model = User

    async def get_by_email(self, email: str) -> User | None:
        """Fetch a user by their email address, or None if not found.

        Used later (Module 4) during login to check if an account exists.
        """
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()