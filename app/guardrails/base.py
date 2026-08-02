"""Guardrail interfaces and shared result types.

A guardrail inspects text (user input or LLM output) and reports whether
it violates a policy, along with WHY — so violations can be logged,
explained to users, and used to tune detection over time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    """How serious a detected violation is."""

    LOW = "low"          # suspicious but likely benign
    MEDIUM = "medium"    # probable violation
    HIGH = "high"        # clear, deliberate violation


@dataclass
class GuardrailResult:
    """The outcome of running one guardrail check.

    Attributes:
        passed: True if the text is acceptable, False if it violates policy.
        reason: human-readable explanation of what triggered a violation
            (empty when passed).
        severity: how serious the violation is.
        matched_patterns: which specific detection rules fired — logged for
            debugging and pattern tuning, never shown to end users (it
            would tell an attacker exactly what to avoid).
        sanitized_text: for output guardrails, the text with problematic
            content redacted. None when no sanitization was applied.
    """

    passed: bool
    reason: str = ""
    severity: Severity = Severity.LOW
    matched_patterns: list[str] = field(default_factory=list)
    sanitized_text: str | None = None


class Guardrail(ABC):
    """Abstract base class for a single guardrail check."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this guardrail (used in logs/metrics)."""
        raise NotImplementedError

    @abstractmethod
    def check(self, text: str) -> GuardrailResult:
        """Inspect text and report whether it passes this guardrail.

        Args:
            text: the text to inspect (user query or LLM output).

        Returns:
            A GuardrailResult describing the outcome.
        """
        raise NotImplementedError