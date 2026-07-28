"""Repository for RefreshToken rows."""

from datetime import UTC, datetime

from sqlalchemy import select
import uuid
from app.models.refresh_token import RefreshToken
from sqlalchemy import update
from app.repositories.base import BaseRepository


class RefreshTokenRepository(BaseRepository[RefreshToken]):
    """Data access methods for refresh tokens."""

    model = RefreshToken

    async def get_valid_token(self, token: str) -> RefreshToken | None:
        """Find a refresh token that is not revoked and not expired.

        Returns None if the token doesn't exist, was already revoked, or
        has expired — the caller doesn't need to know which case it was,
        just that the token can't be used.
        """
        stmt = select(RefreshToken).where(
            RefreshToken.token == token,
            RefreshToken.revoked.is_(False),
            RefreshToken.expires_at > datetime.now(UTC),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def revoke(self, refresh_token: RefreshToken) -> None:
        """Mark a refresh token as revoked (used on logout or rotation)."""
        refresh_token.revoked = True
        await self.session.flush()


    async def revoke_all_for_user(self, user_id: uuid.UUID) -> None:
        """Revoke every active refresh token belonging to a user.

        Called after a password reset — if a session was already stolen
        before the user reset their password, this logs that session out too.
        """
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked.is_(False))
            .values(revoked=True)
        )
        await self.session.execute(stmt)
        await self.session.flush()