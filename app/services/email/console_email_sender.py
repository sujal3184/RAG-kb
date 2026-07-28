"""A development-only EmailSender that logs instead of actually sending.

This project doesn't yet have a real email provider wired up (no SMTP
credentials, no SES/Postmark integration). Rather than blocking email
verification and password reset on that infrastructure work, we log the
"email" clearly to the console/logs so you can copy the link during
development and testing.

Swapping this for a real provider later means writing ONE new class that
implements `EmailSender`, and changing one line in `api/dependencies.py`
— nothing in `AuthService` or the API routes needs to change.
"""

import logging

from app.services.email.base import EmailSender

logger = logging.getLogger(__name__)


class ConsoleEmailSender(EmailSender):
    """Logs emails instead of sending them — for local development only."""

    async def send(self, *, to: str, subject: str, body: str) -> None:
        """Log the "email" so a developer can read/copy it from the console."""
        logger.info(
            "=== EMAIL (console mode — not actually sent) ===\n"
            "To: %s\nSubject: %s\n\n%s\n"
            "================================================",
            to,
            subject,
            body,
        )