"""Repository for VerificationToken rows."""

from datetime import UTC, datetime

from sqlalchemy import select, update

from app.models.verification_token import TokenPurpose, VerificationToken
from app.repositories.base import BaseRepository


class VerificationTokenRepository(BaseRepository[VerificationToken]):
    """Data access methods for email verification / password reset tokens."""

    model = VerificationToken

    async def get_valid_token(
        self, token: str, *, purpose: TokenPurpose
    ) -> VerificationToken | None:
        """Find a token that matches the given purpose, is unused, and unexpired."""
        stmt = select(VerificationToken).where(
            VerificationToken.token == token,
            VerificationToken.purpose == purpose,
            VerificationToken.used_at.is_(None),
            VerificationToken.expires_at > datetime.now(UTC),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def invalidate_active_tokens_for_user(
        self, user_id, *, purpose: TokenPurpose
    ) -> None:
        """Mark all of a user's existing unused tokens (for a purpose) as used.

        Called right before issuing a NEW token, so that if a user requests
        "resend verification email" multiple times, only the latest link
        works — old leaked links stop working immediately.
        """
        stmt = (
            update(VerificationToken)
            .where(
                VerificationToken.user_id == user_id,
                VerificationToken.purpose == purpose,
                VerificationToken.used_at.is_(None),
            )
            .values(used_at=datetime.now(UTC))
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def mark_used(self, token_row: VerificationToken) -> None:
        """Mark a specific token as used, so it can't be redeemed again."""
        token_row.used_at = datetime.now(UTC)
        await self.session.flush()