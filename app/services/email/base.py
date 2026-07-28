"""Email sending interface.

Defines the CONTRACT for sending emails, without committing to any
specific provider. `AuthService` depends only on this interface — never
on a concrete implementation — so we can swap `ConsoleEmailSender` for a
real provider (SES, Postmark, SMTP) later by changing only the
dependency-injection wiring in `api/dependencies.py`.
"""

from abc import ABC, abstractmethod


class EmailSender(ABC):
    """Abstract base class for anything that can send an email."""

    @abstractmethod
    async def send(self, *, to: str, subject: str, body: str) -> None:
        """Send an email.

        Args:
            to: recipient email address.
            subject: email subject line.
            body: plain-text email body.
        """
        raise NotImplementedError