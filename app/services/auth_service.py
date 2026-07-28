"""Authentication business logic.

Coordinates repositories, security helpers, and email sending to implement
register, login, refresh, email verification, and password reset. Routes
stay thin — they only translate HTTP <-> this service.
"""

import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from app.config.settings import Settings
from app.core.exceptions import AuthenticationError, ConflictError, NotFoundError
from app.core.security import TokenType, create_token, hash_password, verify_password
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.models.verification_token import TokenPurpose, VerificationToken
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.repositories.verification_token_repository import VerificationTokenRepository
from app.schemas.auth import TokenResponse
from app.services.email.base import EmailSender

logger = logging.getLogger(__name__)


class AuthService:
    """Handles registration, login, token refresh, email verification, and
    password reset for users."""

    def __init__(
        self,
        user_repo: UserRepository,
        refresh_token_repo: RefreshTokenRepository,
        verification_token_repo: VerificationTokenRepository,
        email_sender: EmailSender,
        settings: Settings,
    ) -> None:
        """Store the dependencies this service needs.

        Args:
            user_repo: repository for reading/writing User rows.
            refresh_token_repo: repository for reading/writing refresh tokens.
            verification_token_repo: repository for verification/reset tokens.
            email_sender: abstraction for sending emails (see services/email/base.py).
            settings: app settings.
        """
        self.user_repo = user_repo
        self.refresh_token_repo = refresh_token_repo
        self.verification_token_repo = verification_token_repo
        self.email_sender = email_sender
        self.settings = settings

    # --- Registration & login -------------------------------------------

    async def register(self, *, email: str, password: str, full_name: str | None) -> User:
        """Create a new user account and send a verification email.

        Raises:
            ConflictError: if a user with this email already exists.
        """
        existing = await self.user_repo.get_by_email(email)
        if existing is not None:
            raise ConflictError(f"An account with email '{email}' already exists")

        user = User(
            email=email,
            hashed_password=hash_password(password),
            full_name=full_name,
        )
        created = await self.user_repo.create(user)
        logger.info("New user registered", extra={"user_id": str(created.id)})

        await self._send_verification_email(created)
        return created

    async def login(self, *, email: str, password: str) -> TokenResponse:
        """Verify credentials and issue a new access + refresh token pair.

        Note: we deliberately do NOT require `is_verified` to log in — an
        unverified account can still use the app, but future features may
        choose to gate specific actions behind verification.

        Raises:
            AuthenticationError: if the email/password don't match, or the
                account is inactive.
        """
        user = await self.user_repo.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            raise AuthenticationError("This account has been deactivated")

        return await self._issue_tokens(user)

    async def refresh(self, *, refresh_token: str) -> TokenResponse:
        """Exchange a valid refresh token for a new access + refresh token pair."""
        stored_token = await self.refresh_token_repo.get_valid_token(refresh_token)
        if stored_token is None:
            raise AuthenticationError("Invalid or expired refresh token")

        user = await self.user_repo.get_by_id(stored_token.user_id)
        if user is None or not user.is_active:
            raise AuthenticationError("Account no longer available")

        await self.refresh_token_repo.revoke(stored_token)
        return await self._issue_tokens(user)

    # --- Email verification ----------------------------------------------

    async def resend_verification_email(self, *, email: str) -> None:
        """Send a new verification email, if the account exists and isn't verified yet.

        Always behaves the same way from the outside (no error raised, no
        indication given) whether or not the email exists — this avoids
        leaking which emails are registered.
        """
        user = await self.user_repo.get_by_email(email)
        if user is None or user.is_verified:
            logger.info("Verification email skipped (unknown or already verified)")
            return

        await self._send_verification_email(user)

    async def verify_email(self, *, token: str) -> User:
        """Mark a user's email as verified, using a token from their email link.

        Raises:
            AuthenticationError: if the token is invalid, expired, or already used.
        """
        token_row = await self.verification_token_repo.get_valid_token(
            token, purpose=TokenPurpose.EMAIL_VERIFICATION
        )
        if token_row is None:
            raise AuthenticationError("Invalid or expired verification link")

        user = await self.user_repo.get_by_id(token_row.user_id)
        if user is None:
            raise AuthenticationError("Invalid or expired verification link")

        user.is_verified = True
        await self.user_repo.update(user)
        await self.verification_token_repo.mark_used(token_row)

        logger.info("Email verified", extra={"user_id": str(user.id)})
        return user

    async def _send_verification_email(self, user: User) -> None:
        """Generate a verification token and email the link to the user."""
        await self.verification_token_repo.invalidate_active_tokens_for_user(
            user.id, purpose=TokenPurpose.EMAIL_VERIFICATION
        )

        raw_token = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + timedelta(
            hours=self.settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS
        )
        await self.verification_token_repo.create(
            VerificationToken(
                user_id=user.id,
                token=raw_token,
                purpose=TokenPurpose.EMAIL_VERIFICATION,
                expires_at=expires_at,
            )
        )

        link = f"{self.settings.FRONTEND_BASE_URL}/verify-email?token={raw_token}"
        await self.email_sender.send(
            to=user.email,
            subject="Verify your email address",
            body=(
                f"Hi {user.full_name or 'there'},\n\n"
                f"Please verify your email by clicking this link:\n{link}\n\n"
                f"This link expires in {self.settings.EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS} hours."
            ),
        )

    # --- Password reset ----------------------------------------------------

    async def forgot_password(self, *, email: str) -> None:
        """Send a password reset email, if the account exists.

        Always behaves the same way from the outside whether or not the
        email exists — this avoids leaking which emails are registered.
        """
        user = await self.user_repo.get_by_email(email)
        if user is None:
            logger.info("Password reset skipped (unknown email)")
            return

        await self.verification_token_repo.invalidate_active_tokens_for_user(
            user.id, purpose=TokenPurpose.PASSWORD_RESET
        )

        raw_token = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + timedelta(
            hours=self.settings.PASSWORD_RESET_TOKEN_EXPIRE_HOURS
        )
        await self.verification_token_repo.create(
            VerificationToken(
                user_id=user.id,
                token=raw_token,
                purpose=TokenPurpose.PASSWORD_RESET,
                expires_at=expires_at,
            )
        )

        link = f"{self.settings.FRONTEND_BASE_URL}/reset-password?token={raw_token}"
        await self.email_sender.send(
            to=user.email,
            subject="Reset your password",
            body=(
                f"Hi {user.full_name or 'there'},\n\n"
                f"Someone requested a password reset. Click this link to choose a new password:\n"
                f"{link}\n\n"
                f"This link expires in {self.settings.PASSWORD_RESET_TOKEN_EXPIRE_HOURS} hour(s). "
                f"If you didn't request this, you can safely ignore this email."
            ),
        )

    async def reset_password(self, *, token: str, new_password: str) -> None:
        """Set a new password using a valid reset token.

        Also revokes ALL of the user's existing refresh tokens — if an
        attacker had stolen a session before the password reset, this logs
        them out everywhere.

        Raises:
            AuthenticationError: if the token is invalid, expired, or already used.
        """
        token_row = await self.verification_token_repo.get_valid_token(
            token, purpose=TokenPurpose.PASSWORD_RESET
        )
        if token_row is None:
            raise AuthenticationError("Invalid or expired reset link")

        user = await self.user_repo.get_by_id(token_row.user_id)
        if user is None:
            raise AuthenticationError("Invalid or expired reset link")

        user.hashed_password = hash_password(new_password)
        await self.user_repo.update(user)
        await self.verification_token_repo.mark_used(token_row)
        await self.refresh_token_repo.revoke_all_for_user(user.id)

        logger.info("Password reset completed", extra={"user_id": str(user.id)})

    # --- Internal helpers ---------------------------------------------------

    async def _issue_tokens(self, user: User) -> TokenResponse:
        """Create a new access token (JWT) and refresh token (stored in DB)."""
        access_token = create_token(
            subject=user.id, token_type=TokenType.ACCESS, settings=self.settings
        )
        refresh_token_value = create_token(
            subject=user.id, token_type=TokenType.REFRESH, settings=self.settings
        )

        expires_at = datetime.now(UTC) + timedelta(days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS)
        await self.refresh_token_repo.create(
            RefreshToken(user_id=user.id, token=refresh_token_value, expires_at=expires_at)
        )

        return TokenResponse(access_token=access_token, refresh_token=refresh_token_value)