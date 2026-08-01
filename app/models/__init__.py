"""Importing this package registers every model with SQLAlchemy's Base.

Any process that needs the FULL set of table relationships resolvable
(Alembic, the Celery worker) must import this package — not just the
specific model modules it happens to use directly — otherwise foreign
keys pointing at "unseen" tables fail with NoReferencedTableError.
"""

from app.models import (  # noqa: F401
    chunk,
    conversation,
    document,
    knowledge_base,
    message,
    refresh_token,
    user,
    verification_token,
)