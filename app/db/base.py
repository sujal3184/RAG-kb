"""SQLAlchemy declarative base.

Every database model (table) in the app must inherit from `Base`. This is
what lets SQLAlchemy (and Alembic) discover all our tables and generate
migrations automatically.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """The shared base class for all database models."""