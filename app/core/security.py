"""Password hashing and JWT token creation/verification.

This is the only file that touches `passlib` (hashing) and `jose` (JWTs) —
every other part of the app should call these helper functions instead of
using those libraries directly. That keeps the app easy to change later
(e.g., swapping bcrypt for argon2 someday means editing only this file).
"""

import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config.settings import Settings
from app.core.exceptions import AuthenticationError

# `CryptContext` handles hashing AND verifying, and can support multiple
# algorithms at once (useful if we ever want to migrate hash algorithms
# without breaking existing users' passwords).
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenType(StrEnum):
    """Distinguishes access tokens from refresh tokens inside the JWT payload.

    Without this, someone could take a refresh token and use it directly
    as an access token (or vice versa), since both are just JWTs signed
    with the same secret. Checking the "type" claim closes that gap.
    """

    ACCESS = "access"
    REFRESH = "refresh"


def hash_password(plain_password: str) -> str:
    """Turn a plain-text password into a secure hash for storage.

    Args:
        plain_password: the password as typed by the user.

    Returns:
        A bcrypt hash string, safe to store in the database.
    """
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check whether a plain-text password matches a stored hash.

    Args:
        plain_password: the password as typed by the user during login.
        hashed_password: the hash previously stored for that user.

    Returns:
        True if they match, False otherwise. Never raises on mismatch.
    """
    return _pwd_context.verify(plain_password, hashed_password)


def create_token(
    *,
    subject: uuid.UUID,
    token_type: TokenType,
    settings: Settings,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a signed JWT for a given user.

    Args:
        subject: the user's id — becomes the JWT's "sub" claim, the
            standard way to identify who a token belongs to.
        token_type: whether this is an access or refresh token, so
            `decode_token` can reject the wrong type being used in the
            wrong place.
        settings: app settings, providing the secret key, algorithm, and
            expiry durations.
        extra_claims: any additional data to embed in the token (rarely
            needed — keep tokens small).

    Returns:
        An encoded JWT string.
    """
    now = datetime.now(UTC)
    expire_delta = (
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        if token_type == TokenType.ACCESS
        else timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )

    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type.value,
        "iat": now,
        "exp": now + expire_delta,
        "jti": str(uuid.uuid4()),  # unique token id, useful for revocation later
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, *, expected_type: TokenType, settings: Settings) -> dict[str, Any]:
    """Decode and validate a JWT, returning its payload.

    Args:
        token: the raw JWT string received from the client.
        expected_type: whether we expect this to be an access or refresh
            token — rejects a refresh token being used where an access
            token is required, and vice versa.
        settings: app settings, providing the secret key and algorithm.

    Returns:
        The decoded token payload (a dict) if valid.

    Raises:
        AuthenticationError: if the token is expired, malformed, has an
            invalid signature, or is the wrong type.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as exc:
        raise AuthenticationError("Invalid or expired token") from exc

    if payload.get("type") != expected_type.value:
        raise AuthenticationError("Token type mismatch")

    return payload