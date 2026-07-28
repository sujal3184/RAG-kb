"""Shared FastAPI dependencies used across API routes."""

import uuid
from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.core.security import TokenType, decode_token
from app.db.dependencies import get_db_session
from app.models.user import User
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.repositories.verification_token_repository import VerificationTokenRepository
from app.services.auth_service import AuthService
from app.services.email.base import EmailSender
from app.services.email.console_email_sender import ConsoleEmailSender

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{get_settings().API_V1_PREFIX}/auth/login")


def get_user_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserRepository:
    return UserRepository(db)


def get_refresh_token_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> RefreshTokenRepository:
    return RefreshTokenRepository(db)


def get_verification_token_repository(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> VerificationTokenRepository:
    return VerificationTokenRepository(db)


def get_email_sender() -> EmailSender:
    """Provide the email-sending implementation.

    This is the ONE place to change when a real email provider is added
    later — swap `ConsoleEmailSender()` for e.g. `SesEmailSender(settings)`,
    and nothing else in the app needs to change.
    """
    return ConsoleEmailSender()


def get_auth_service(
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    refresh_token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
    verification_token_repo: Annotated[
        VerificationTokenRepository, Depends(get_verification_token_repository)
    ],
    email_sender: Annotated[EmailSender, Depends(get_email_sender)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AuthService:
    return AuthService(
        user_repo, refresh_token_repo, verification_token_repo, email_sender, settings
    )


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    user_repo: Annotated[UserRepository, Depends(get_user_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    payload = decode_token(token, expected_type=TokenType.ACCESS, settings=settings)

    user_id = payload.get("sub")
    if user_id is None:
        raise AuthenticationError("Token missing subject claim")

    user = await user_repo.get_by_id(uuid.UUID(user_id))
    if user is None or not user.is_active:
        raise AuthenticationError("User no longer available")

    return user