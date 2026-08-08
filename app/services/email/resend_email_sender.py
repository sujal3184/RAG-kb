"""Resend-based email sender.

Sends real emails via Resend's API — same EmailSender interface as
ConsoleEmailSender (Module 4b), so AuthService and everything upstream is
completely unaffected by this swap.

Note on Resend's free-tier sandbox: without a verified custom domain,
Resend only delivers to the email address you signed up with. To send to
arbitrary real users, verify a domain in Resend's dashboard and update
EMAIL_FROM_ADDRESS to use it.
"""

import logging

import resend

from app.services.email.base import EmailSender

logger = logging.getLogger(__name__)


class ResendEmailSender(EmailSender):
    """Sends emails via the Resend API."""

    def __init__(self, api_key: str, *, from_address: str, from_name: str) -> None:
        """Configure the Resend client.

        Args:
            api_key: Resend API key.
            from_address: the sender email address (must be on a domain
                verified in Resend, or the shared sandbox address).
            from_name: display name shown alongside the sender address.
        """
        resend.api_key = api_key
        self._from = f"{from_name} <{from_address}>"

    async def send(self, *, to: str, subject: str, body: str) -> None:
        """Send an email via Resend.

        Resend's Python SDK is synchronous, so we don't need asyncio.to_thread
        here for correctness, but we use it anyway to avoid blocking the
        event loop during the network call — consistent with how the rest
        of the app treats blocking I/O (see FileStorage, Module 6).

        Failures are logged but NOT raised — registration/password-reset
        flows should not fail outright just because email delivery had an
        issue; the token still exists and can be resent.
        """
        import asyncio

        try:
            await asyncio.to_thread(
                resend.Emails.send,
                {
                    "from": self._from,
                    "to": [to],
                    "subject": subject,
                    "text": body,
                },
            )
            logger.info("Email sent via Resend", extra={"to": to, "subject": subject})
        except Exception as exc:
            logger.error(
                "Failed to send email via Resend",
                extra={"to": to, "subject": subject, "error": str(exc)},
            )