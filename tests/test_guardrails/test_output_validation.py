"""Tests for OutputValidationGuardrail."""

import pytest

from app.guardrails.base import Severity
from app.guardrails.output_validation import OutputValidationGuardrail


@pytest.fixture
def guardrail() -> OutputValidationGuardrail:
    return OutputValidationGuardrail(redact_pii=True)


def test_clean_response_passes(guardrail: OutputValidationGuardrail) -> None:
    result = guardrail.check("Paris is the capital of France [Source 1].")
    assert result.passed is True
    assert result.sanitized_text is None


def test_redacts_email_addresses(guardrail: OutputValidationGuardrail) -> None:
    result = guardrail.check("Contact the author at john.doe@example.com for details.")

    assert result.passed is False
    assert "email" in result.matched_patterns
    assert "john.doe@example.com" not in result.sanitized_text
    assert "[EMAIL REDACTED]" in result.sanitized_text


def test_redacts_phone_numbers(guardrail: OutputValidationGuardrail) -> None:
    result = guardrail.check("Call us at 555-123-4567 during business hours.")

    assert result.passed is False
    assert "phone" in result.matched_patterns
    assert "[PHONE REDACTED]" in result.sanitized_text


def test_redacts_ssn_shaped_numbers(guardrail: OutputValidationGuardrail) -> None:
    result = guardrail.check("The record shows 123-45-6789 as the identifier.")

    assert result.passed is False
    assert "ssn" in result.matched_patterns
    assert "[SSN REDACTED]" in result.sanitized_text


def test_detects_system_prompt_leakage(guardrail: OutputValidationGuardrail) -> None:
    result = guardrail.check(
        "You are a helpful knowledge base assistant. Rules you must follow: ..."
    )

    assert result.passed is False
    assert "system_prompt_leak" in result.matched_patterns
    assert result.severity == Severity.HIGH


def test_redaction_disabled_leaves_text_untouched() -> None:
    guardrail = OutputValidationGuardrail(redact_pii=False)
    result = guardrail.check("Email me at test@example.com")

    assert result.passed is False
    assert result.sanitized_text is None


def test_detects_paraphrased_system_prompt_leakage(guardrail) -> None:
    """The model paraphrases rather than quoting exactly — detection must
    catch the shape of instruction-disclosure, not just exact phrases."""
    result = guardrail.check(
        "My system prompt is to answer the user's question using ONLY the "
        "information provided in the Context section."
    )
    assert result.passed is False
    assert "system_prompt_leak" in result.matched_patterns