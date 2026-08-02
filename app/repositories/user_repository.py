"""Repository for the User table.

Adds user-specific queries on top of the generic CRUD from BaseRepository.
"""

from sqlalchemy import func,select

from app.models.user import User , UserRole
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


    async def list_all_paginated(
        self, *, limit: int = 50, offset: int = 0
    ) -> list[User]:
        """List all users, newest first — admin-only view."""
        stmt = select(User).order_by(User.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_all(self) -> int:
        """Count all registered users."""
        stmt = select(func.count()).select_from(User)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def count_by_status(self) -> dict[str, int]:
        """Count users grouped by active/verified status, for admin stats."""
        active_stmt = select(func.count()).select_from(User).where(User.is_active.is_(True))
        verified_stmt = select(func.count()).select_from(User).where(User.is_verified.is_(True))
        admin_stmt = select(func.count()).select_from(User).where(User.role == UserRole.ADMIN)

        active = (await self.session.execute(active_stmt)).scalar_one()
        verified = (await self.session.execute(verified_stmt)).scalar_one()
        admins = (await self.session.execute(admin_stmt)).scalar_one()

        return {"active": active, "verified": verified, "admins": admins}