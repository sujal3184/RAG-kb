"""Authentication endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_auth_service, get_current_user
from app.models.user import User
from app.schemas.auth import (
    ForgotPasswordRequest,
    MessageResponse,
    RefreshTokenRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
    VerifyEmailRequest,
)
from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account and send a verification email",
)
async def register(
    body: UserRegisterRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> UserResponse:
    user = await auth_service.register(
        email=body.email, password=body.password, full_name=body.full_name
    )
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse, summary="Log in and receive tokens")
async def login(
    body: UserLoginRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    return await auth_service.login(email=body.email, password=body.password)


@router.post("/refresh", response_model=TokenResponse, summary="Exchange a refresh token")
async def refresh(
    body: RefreshTokenRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    return await auth_service.refresh(refresh_token=body.refresh_token)


@router.get("/me", response_model=UserResponse, summary="Get the current user's profile")
async def get_me(current_user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.post(
    "/verify-email",
    response_model=UserResponse,
    summary="Confirm an email address using the token from the verification link",
)
async def verify_email(
    body: VerifyEmailRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> UserResponse:
    """Marks the account as verified. The frontend calls this after the user
    clicks the link in their email (extracting `token` from the URL)."""
    user = await auth_service.verify_email(token=body.token)
    return UserResponse.model_validate(user)


@router.post(
    "/resend-verification",
    response_model=MessageResponse,
    summary="Request a new verification email",
)
async def resend_verification(
    body: ResendVerificationRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Always returns the same success message, regardless of whether the
    email exists or is already verified — this avoids leaking account info."""
    await auth_service.resend_verification_email(email=body.email)
    return MessageResponse(
        message="If an account with that email exists and isn't verified, a new link has been sent."
    )


@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="Request a password reset email",
)
async def forgot_password(
    body: ForgotPasswordRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """Always returns the same success message, regardless of whether the
    email exists — this avoids leaking which emails are registered."""
    await auth_service.forgot_password(email=body.email)
    return MessageResponse(
        message="If an account with that email exists, a password reset link has been sent."
    )


@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="Set a new password using the token from the reset link",
)
async def reset_password(
    body: ResetPasswordRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    await auth_service.reset_password(token=body.token, new_password=body.new_password)
    return MessageResponse(message="Password has been reset successfully. Please log in again.")