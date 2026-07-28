"""Alembic environment configuration.

This file connects Alembic (the migration tool) to OUR app's settings and
models, so:
1. The database URL comes from our `.env` (via `Settings`), not duplicated
   here.
2. Alembic can "see" all our SQLAlchemy models to auto-generate migrations
   by comparing them against the current database structure.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config.settings import get_settings
from app.db.base import Base


# Import every model module here so Base.metadata knows about all tables.
# Without this import, Alembic would think these tables don't exist yet
# and try to create migrations that delete them.
from app.models import document, knowledge_base, refresh_token, user, verification_token  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is what Alembic compares your models against to detect changes.
target_metadata = Base.metadata

# Inject our app's real database URL (built from .env) into Alembic's config.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)


def run_migrations_offline() -> None:
    """Generate SQL scripts without connecting to a database (rarely used)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations using an already-open database connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Connect to the real database and apply/generate migrations.

    Uses an async engine since our app is async, so Alembic's connection
    behavior matches how the app itself connects.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())