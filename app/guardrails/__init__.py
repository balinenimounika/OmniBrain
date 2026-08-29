"""
OmniBrain Guardrails Module

Provides document scope validation and guardrail checks for responses.
Uses NeMo Guardrails to restrict answers to document-related questions only.
"""

from app.guardrails.guardrails import (
    should_answer_question,
    get_guardrail_response,
    guardrail_status_summary
)

__all__ = ["should_answer_question", "get_guardrail_response", "guardrail_status_summary"]
