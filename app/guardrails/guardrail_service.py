"""Guardrail orchestration.

Applies input guardrails (blocking violations) and output guardrails
(sanitizing violations), with all outcomes logged and recorded as metrics
so false positives can be spotted and patterns tuned over time.
"""

import logging

from app.guardrails.base import Guardrail, GuardrailResult
from app.guardrails.exceptions import GuardrailViolationError
from app.observability.metrics import guardrail_checks_total

logger = logging.getLogger(__name__)


class GuardrailService:
    """Runs input and output guardrails around LLM interactions."""

    def __init__(
        self,
        input_guardrails: list[Guardrail],
        output_guardrails: list[Guardrail],
        *,
        enabled: bool,
        block_on_input_violation: bool,
    ) -> None:
        """Store the guardrails to run and how strictly to enforce them.

        Args:
            input_guardrails: checks applied to user queries.
            output_guardrails: checks applied to LLM responses.
            enabled: master switch — when False, all checks are skipped.
            block_on_input_violation: if True, input violations raise an
                error; if False, they're logged but allowed through
                (useful for observing false-positive rates before
                enforcing in a new deployment).
        """
        self.input_guardrails = input_guardrails
        self.output_guardrails = output_guardrails
        self._enabled = enabled
        self._block_on_input_violation = block_on_input_violation

    def check_input(self, text: str) -> None:
        """Run all input guardrails on a user query.

        Args:
            text: the user's query.

        Raises:
            GuardrailViolationError: if a guardrail fails and blocking is
                enabled.
        """
        if not self._enabled:
            return

        for guardrail in self.input_guardrails:
            result = guardrail.check(text)
            self._record(guardrail.name, result, direction="input")

            if not result.passed:
                logger.warning(
                    "Input guardrail violation",
                    extra={
                        "guardrail": guardrail.name,
                        "severity": result.severity.value,
                        "matched_patterns": result.matched_patterns,
                    },
                )
                if self._block_on_input_violation:
                    raise GuardrailViolationError(result.reason)

    def check_output(self, text: str) -> str:
        """Run all output guardrails on an LLM response.

        Args:
            text: the generated response.

        Returns:
            The response text, sanitized if any guardrail produced a
            sanitized version (e.g. PII redacted). Output violations
            never block — a redacted answer is more useful than none.
        """
        if not self._enabled:
            return text

        current_text = text
        for guardrail in self.output_guardrails:
            result = guardrail.check(current_text)
            self._record(guardrail.name, result, direction="output")

            if not result.passed:
                logger.warning(
                    "Output guardrail violation",
                    extra={
                        "guardrail": guardrail.name,
                        "severity": result.severity.value,
                        "matched_patterns": result.matched_patterns,
                    },
                )
                if result.sanitized_text is not None:
                    current_text = result.sanitized_text

        return current_text

    @staticmethod
    def _record(guardrail_name: str, result: GuardrailResult, *, direction: str) -> None:
        """Record a guardrail check outcome as a Prometheus metric."""
        outcome = "passed" if result.passed else "violated"
        guardrail_checks_total.labels(
            guardrail=guardrail_name, direction=direction, outcome=outcome
        ).inc()