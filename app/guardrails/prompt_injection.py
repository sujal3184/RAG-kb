"""Prompt injection detection for user input.

IMPORTANT LIMITATION — please read: this uses pattern matching, which
catches common, obvious injection attempts (the large majority of what
casual probing looks like) but is NOT a security guarantee. A determined
attacker who knows these patterns exist can phrase an injection to evade
them. For a system exposed to genuinely adversarial untrusted users at
scale, a dedicated classifier model or commercial guardrail service is
the appropriate tool. This module provides meaningful baseline protection
at near-zero latency/memory cost, which is the right trade-off for this
project's scope.
"""

import re

from app.guardrails.base import Guardrail, GuardrailResult, Severity

# Patterns targeting attempts to override system instructions.
# Patterns targeting attempts to override system instructions.
_INSTRUCTION_OVERRIDE_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+((all|the|any)\s+)*(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
     "instruction_override"),
    (r"disregard\s+((all|the|any)\s+)*(previous|prior|above|earlier)\s+(instructions?|context|rules?)",
     "instruction_override"),
    (r"forget\s+(everything|all)\s+(you|above|before)", "instruction_override"),
    (r"you\s+are\s+now\s+(a|an)\s+", "role_reassignment"),
    (r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)\s+", "role_reassignment"),
    (r"pretend\s+(to\s+be|you\s+are)", "role_reassignment"),
    (r"new\s+(instructions?|rules?|system\s+prompt)\s*:", "instruction_injection"),
]

# Patterns targeting attempts to extract the system prompt itself.
# Patterns targeting attempts to extract the system prompt itself.
_PROMPT_EXTRACTION_PATTERNS: list[tuple[str, str]] = [
    (r"(reveal|show|print|repeat|output|display|tell|give|state)\s+(me\s+)?(your|the)\s+"
     r"(system\s+)?(prompt|instructions?|rules?|guidelines?|directives?)", "prompt_extraction"),
    (r"what\s+(is|are|was|were)\s+(your|the)\s+(original\s+|initial\s+|exact\s+)?"
     r"(system\s+)?(prompt|instructions?|rules?|guidelines?|directives?)", "prompt_extraction"),
    (r"repeat\s+(everything\s+)?(above|before\s+this)", "prompt_extraction"),
    (r"(describe|explain|summarize)\s+(your|the)\s+(system\s+)?(prompt|instructions?|rules?)",
     "prompt_extraction"),
    (r"how\s+(were|are)\s+you\s+(instructed|configured|programmed|told)", "prompt_extraction"),
]

# Patterns targeting attempts to escape the RAG context constraint.
_CONTEXT_ESCAPE_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+the\s+(provided\s+)?(context|documents?|sources?)", "context_escape"),
    (r"(answer|respond)\s+(without|regardless\s+of)\s+(using\s+)?the\s+context",
     "context_escape"),
    (r"from\s+your\s+own\s+knowledge,?\s+(not|ignoring)", "context_escape"),
]

_ALL_PATTERNS = (
    _INSTRUCTION_OVERRIDE_PATTERNS
    + _PROMPT_EXTRACTION_PATTERNS
    + _CONTEXT_ESCAPE_PATTERNS
)

_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), label) for pattern, label in _ALL_PATTERNS
]


class PromptInjectionGuardrail(Guardrail):
    """Detects common prompt-injection attempts in user input."""

    def __init__(self, *, max_query_length: int = 4000) -> None:
        """Configure the guardrail.

        Args:
            max_query_length: queries longer than this are rejected —
                extremely long inputs are a common vector for burying
                injection payloads, and also waste tokens/cost.
        """
        self._max_query_length = max_query_length

    @property
    def name(self) -> str:
        return "prompt_injection"

    def check(self, text: str) -> GuardrailResult:
        """Inspect user input for prompt-injection patterns.

        Args:
            text: the user's query text.

        Returns:
            A GuardrailResult — `passed=False` with matched pattern labels
            if injection indicators were found.
        """
        if len(text) > self._max_query_length:
            return GuardrailResult(
                passed=False,
                reason=f"Query exceeds the maximum allowed length of {self._max_query_length} characters",
                severity=Severity.MEDIUM,
                matched_patterns=["excessive_length"],
            )

        matched_labels = [
            label for pattern, label in _COMPILED_PATTERNS if pattern.search(text)
        ]

        if not matched_labels:
            return GuardrailResult(passed=True)

        # Multiple distinct injection signals in one query strongly
        # suggests deliberate probing rather than an unlucky phrasing.
        severity = Severity.HIGH if len(set(matched_labels)) > 1 else Severity.MEDIUM

        return GuardrailResult(
            passed=False,
            reason="Your message appears to contain instructions that attempt to "
            "override how this assistant works. Please rephrase your question.",
            severity=severity,
            matched_patterns=sorted(set(matched_labels)),
        )