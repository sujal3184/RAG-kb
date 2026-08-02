"""add user role column

Revision ID: 56024122bba8
Revises: f9d2e92168cd
Create Date: ...
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '56024122bba8'
down_revision: str | None = 'f9d2e92168cd'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Postgres enums are real database types that must be created BEFORE
    # any column can reference them. Alembic's autogenerate doesn't emit
    # this for add_column on an existing table, so we do it explicitly.
    userrole_enum = postgresql.ENUM("USER", "ADMIN", name="userrole")
    userrole_enum.create(op.get_bind(), checkfirst=True)

    op.add_column(
        "users",
        sa.Column(
            "role",
            userrole_enum,
            nullable=False,
            server_default="USER",  # backfills existing rows
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "role")

    userrole_enum = postgresql.ENUM("USER", "ADMIN", name="userrole")
    userrole_enum.drop(op.get_bind(), checkfirst=True)