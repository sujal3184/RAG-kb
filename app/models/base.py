"""Shared building blocks (mixins) for database models.

A "mixin" is a small class that adds a few columns/behaviors, meant to be
combined with other classes — not used on its own. Every table in this app
will combine `Base` (from db/base.py) with these mixins instead of
re-declaring `id`, `created_at`, `updated_at` every time.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column


class UUIDPrimaryKeyMixin:
    """Adds a UUID primary key column named `id`.

    We use UUIDs (not auto-incrementing integers) as primary keys because:
    - They can be generated on the app side, before the row is even saved.
    - They don't leak information (an integer ID reveals "how many rows
      exist", a UUID doesn't).
    - They make it safe to merge data from multiple databases later
      (e.g., if we ever shard or replicate).
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


class TimestampMixin:
    """Adds `created_at` and `updated_at` columns, set automatically."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),  # let the DATABASE set this, not Python
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),  # auto-updates whenever the row changes
        nullable=False,
    )