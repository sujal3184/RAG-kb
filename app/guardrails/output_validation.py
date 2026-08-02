"""Output validation for LLM responses.

Checks generated answers for two problems before they reach the user:
1. System prompt leakage — the model echoing back its own instructions.
2. PII — personally identifiable information appearing in the response.

LIMITATION: PII detection covers email addresses, phone numbers, and
credit-card/SSN-shaped digit sequences — patterns with reliable shapes
and low false-positive rates. It does NOT detect names or addresses,
which would require named-entity-recognition models and produce
substantially more false positives. This is a deliberate scope choice.
"""

import re

from app.guardrails.base import Guardrail, GuardrailResult, Severity

# Phrases from our own system prompt (Module 14) — if these appear in a
# RESPONSE, the model is leaking its instructions back to the user.
_SYSTEM_PROMPT_LEAK_PATTERNS: list[tuple[str, str]] = [
    (r"my\s+(system\s+)?(prompt|instructions?|rules?|guidelines?)\s+(is|are)",
     "system_prompt_leak"),
    (r"i\s+(was|am|have\s+been)\s+(instructed|told|configured|programmed)\s+to",
     "system_prompt_leak"),
    (r"(my|the)\s+(rules?|instructions?|guidelines?)\s+(i\s+)?(must|should)\s+follow",
     "system_prompt_leak"),
    (r"you\s+are\s+a\s+helpful\s+knowledge\s+base\s+assistant", "system_prompt_leak"),
    (r"rules\s+you\s+must\s+follow", "system_prompt_leak"),
    (r"base\s+(your\s+)?answers?\s+strictly\s+on\s+the\s+provided\s+context",
     "system_prompt_leak"),
    (r"answer\s+the\s+user'?s?\s+question\s+using\s+only", "system_prompt_leak"),
]

_PII_PATTERNS: list[tuple[str, str, str]] = [
    # (regex, label, replacement)
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "email", "[EMAIL REDACTED]"),
    (
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "phone",
        "[PHONE REDACTED]",
    ),
    (
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "credit_card",
        "[CARD NUMBER REDACTED]",
    ),
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn", "[SSN REDACTED]"),
]

_COMPILED_LEAK_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in _SYSTEM_PROMPT_LEAK_PATTERNS
]

_COMPILED_PII_PATTERNS = [
    (re.compile(pattern), label, replacement)
    for pattern, label, replacement in _PII_PATTERNS
]


class OutputValidationGuardrail(Guardrail):
    """Detects system-prompt leakage and PII in LLM responses, redacting
    PII rather than discarding the whole answer."""

    def __init__(self, *, redact_pii: bool = True) -> None:
        """Configure the guardrail.

        Args:
            redact_pii: whether detected PII should be replaced with
                redaction markers in the returned text.
        """
        self._redact_pii = redact_pii

    @property
    def name(self) -> str:
        return "output_validation"

    def check(self, text: str) -> GuardrailResult:
        """Inspect an LLM response for leakage and PII.

        Args:
            text: the generated response text.

        Returns:
            A GuardrailResult. If PII was found and redaction is enabled,
            `sanitized_text` contains the redacted version — the caller
            should use that instead of the original.
        """
        matched_labels: list[str] = []

        leak_labels = [
            label for pattern, label in _COMPILED_LEAK_PATTERNS if pattern.search(text)
        ]
        matched_labels.extend(leak_labels)

        sanitized = text
        pii_labels: list[str] = []
        for pattern, label, replacement in _COMPILED_PII_PATTERNS:
            if pattern.search(sanitized):
                pii_labels.append(label)
                if self._redact_pii:
                    sanitized = pattern.sub(replacement, sanitized)

        matched_labels.extend(pii_labels)

        if not matched_labels:
            return GuardrailResult(passed=True)

        # System prompt leakage is more serious than incidental PII —
        # it indicates the model was successfully manipulated.
        severity = Severity.HIGH if leak_labels else Severity.MEDIUM

        reasons = []
        if leak_labels:
            reasons.append("response appears to contain system instructions")
        if pii_labels:
            reasons.append(f"response contains potential PII ({', '.join(sorted(set(pii_labels)))})")

        return GuardrailResult(
            passed=False,
            reason="; ".join(reasons),
            severity=severity,
            matched_patterns=sorted(set(matched_labels)),
            sanitized_text=sanitized if (self._redact_pii and pii_labels) else None,
        )