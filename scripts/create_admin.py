"""Promote an existing user to admin.

The first admin must be created this way rather than via an API endpoint:
an endpoint that grants admin rights would either need admin auth already
(chicken-and-egg) or be an obvious security hole. Requiring shell access
is the standard, safe bootstrap.

Usage:
    uv run python scripts/create_admin.py user@example.com
"""

import asyncio
import sys

from app.db.session import async_session_factory
from app.models.user import UserRole
from app.repositories.user_repository import UserRepository


async def promote(email: str) -> None:
    """Promote the user with the given email to the admin role."""
    async with async_session_factory() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email(email)

        if user is None:
            print(f"No user found with email '{email}'. Register the account first.")
            sys.exit(1)

        if user.role == UserRole.ADMIN:
            print(f"'{email}' is already an admin.")
            return

        user.role = UserRole.ADMIN
        await repo.update(user)
        await session.commit()
        print(f"'{email}' has been promoted to admin.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/create_admin.py <email>")
        sys.exit(1)

    asyncio.run(promote(sys.argv[1]))
    