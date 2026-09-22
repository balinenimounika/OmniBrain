"""
OmniBrain Week 3 Guardrails
Document-scope guardrail without external API keys.
"""

from typing import Dict, Any, Optional
import re

OUT_OF_SCOPE_MESSAGE = "I can only answer questions based on the provided document."
INSUFFICIENT_CONTEXT_MESSAGE = "I could not find enough information about that in the provided document."


def get_words(text):
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))


def should_answer_question(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("user_query", "") or state.get("query", "")
    retrieved_context = state.get("retrieved_context", "")

    if isinstance(retrieved_context, list):
        context_text = " ".join(
            str(doc.get("text", "")) if isinstance(doc, dict) else str(doc)
            for doc in retrieved_context
        )
    else:
        context_text = str(retrieved_context)

    # No retrieved document context = block
    if not context_text.strip():
        state["guardrail_allowed"] = False
        state["guardrail_reason"] = "No relevant document content retrieved"
        state["guardrail_message"] = OUT_OF_SCOPE_MESSAGE
        return state

    query_words = get_words(query)
    context_words = get_words(context_text)

    stop_words = {
        "what", "is", "are", "was", "were", "the", "a", "an",
        "and", "or", "of", "to", "in", "on", "for", "from",
        "about", "does", "do", "how", "why", "when", "where",
        "which", "who", "can", "could", "would", "should",
        "this", "that", "document", "tell", "me"
    }

    meaningful_words = query_words - stop_words
    matches = meaningful_words.intersection(context_words)

    if matches:
        state["guardrail_allowed"] = True
        state["guardrail_reason"] = "Query is within provided document scope"
        state["guardrail_message"] = None
        state["guardrail_context_quality"] = "high"
    else:
        state["guardrail_allowed"] = False
        state["guardrail_reason"] = "Query is outside provided document scope"
        state["guardrail_message"] = OUT_OF_SCOPE_MESSAGE
        state["guardrail_context_quality"] = "empty"

    state["guardrail_score"] = 1.0
    return state


def get_guardrail_response(state: Dict[str, Any]) -> Optional[str]:
    if not state.get("guardrail_allowed", True):
        return state.get("guardrail_message", OUT_OF_SCOPE_MESSAGE)
    return None


def guardrail_status_summary(state: Dict[str, Any]) -> str:
    allowed = state.get("guardrail_allowed", "Unknown")
    reason = state.get("guardrail_reason", "No reason")
    status = "ALLOWED" if allowed else "BLOCKED"
    return f"{status} | Reason: {reason}"