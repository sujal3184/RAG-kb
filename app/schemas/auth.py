"""Request/response models for authentication endpoints."""

import uuid

from pydantic import BaseModel, ConfigDict, EmailStr, Field
from app.models.user import UserRole


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    is_active: bool
    is_verified: bool
    role: UserRole

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    """A generic "it worked" response for actions with no useful data to return."""

    message: str


class VerifyEmailRequest(BaseModel):
    """What the client sends when the user clicks the verification link."""

    token: str


class ResendVerificationRequest(BaseModel):
    """What the client sends to request a new verification email."""

    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    """What the client sends to start a password reset."""

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """What the client sends to actually set a new password."""

    token: str
    new_password: str = Field(min_length=8, max_length=128)