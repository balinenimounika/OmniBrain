"""
NeMo Guardrails Integration for OmniBrain
Restricts responses to questions within the provided document scope.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Try to import NeMo Guardrails (may fail in some environments)
try:
    from nemoguardrails import LLMRails, RailsConfig
    from nemoguardrails.actions import action
    NEMO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NeMo Guardrails not fully available: {e}. Falling back to raw LLM checks.")
    LLMRails = None
    RailsConfig = None
    NEMO_AVAILABLE = False
    
    # Mock the @action decorator so functions still compile
    def action(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ==================== CONFIGURATION ====================
OUT_OF_SCOPE_MESSAGE = "I can only answer questions based on the provided document."
INSUFFICIENT_CONTEXT_MESSAGE = "I could not find enough information about that in the provided document."

# Initialize Rails Config
rails = None
if NEMO_AVAILABLE:
    try:
        config_path = os.path.dirname(os.path.abspath(__file__))
        config = RailsConfig.from_path(config_path)
        rails = LLMRails(config)
    except Exception as e:
        logger.error(f"Failed to initialize NeMo Guardrails: {e}")
        rails = None

@action(is_system_action=True, name="check_unrelated")
async def check_unrelated(query: str) -> bool:
    """Uses LLM to check if the query is general/unrelated."""
    try:
        llm = ChatOpenAI(temperature=0)
        messages = [
            SystemMessage(content="You are a classifier. Determine if the user query is a general knowledge question, a coding request, a joke, casual conversation, or anything unrelated to querying a specific document. Answer exactly 'yes' if it is unrelated, and 'no' if it sounds like a document query."),
            HumanMessage(content=query)
        ]
        response = llm.invoke(messages)
        return 'yes' in response.content.lower()
    except Exception as e:
        logger.error(f"Error in check_unrelated: {e}")
        return True  # Block on error

@action(is_system_action=True, name="check_answerable")
async def check_answerable(query: str, context: str) -> bool:
    """Uses LLM to check if the query can be answered with context."""
    if not context or not context.strip():
        return False
        
    try:
        llm = ChatOpenAI(temperature=0)
        messages = [
            SystemMessage(content="You are a classifier. Given a context and a question, determine if the question can be answered using ONLY the provided context. Answer exactly 'yes' if answerable, and 'no' if the context does not contain the answer."),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {query}")
        ]
        response = llm.invoke(messages)
        return 'yes' in response.content.lower()
    except Exception as e:
        logger.error(f"Error in check_answerable: {e}")
        return False  # Block on error

if rails:
    rails.register_action(check_unrelated, name="check_unrelated")
    rails.register_action(check_answerable, name="check_answerable")

def should_answer_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main guardrail enforcement function for LangGraph state.
    """
    try:
        query = state.get("user_query", "") or state.get("query", "")
        retrieved_context = state.get("retrieved_context", "")
        
        # Check if context is completely empty
        if not retrieved_context or not str(retrieved_context).strip():
            state["guardrail_allowed"] = False
            state["guardrail_reason"] = "No relevant document content retrieved"
            state["guardrail_message"] = OUT_OF_SCOPE_MESSAGE
            return state
            
        logger.info(f"Applying guardrail to query: '{query[:50]}...'")
        
        if rails and NEMO_AVAILABLE:
            # Run NeMo Guardrails dialog
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            messages = [
                {"role": "context", "content": {"retrieved_context": retrieved_context, "user_message": query}},
                {"role": "user", "content": query}
            ]
            
            response = loop.run_until_complete(
                rails.generate_async(messages=messages)
            )
            loop.close()
            
            bot_response = response.get("content", "")
        else:
            # Fallback to direct LLM execution if NeMo Guardrails is not installed/working
            logger.info("Using direct LLM guardrail fallback...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            is_unrelated = loop.run_until_complete(check_unrelated(query))
            if is_unrelated:
                bot_response = OUT_OF_SCOPE_MESSAGE
            else:
                is_answerable = loop.run_until_complete(check_answerable(query, retrieved_context))
                if not is_answerable:
                    bot_response = INSUFFICIENT_CONTEXT_MESSAGE
                else:
                    bot_response = "allow"
            loop.close()
        
        if bot_response == OUT_OF_SCOPE_MESSAGE or "only answer questions based on the provided document" in bot_response:
            state["guardrail_allowed"] = False
            state["guardrail_reason"] = "Out of scope query"
            state["guardrail_message"] = OUT_OF_SCOPE_MESSAGE
        elif bot_response == INSUFFICIENT_CONTEXT_MESSAGE or "could not find enough information" in bot_response:
            state["guardrail_allowed"] = False
            state["guardrail_reason"] = "Insufficient context"
            state["guardrail_message"] = INSUFFICIENT_CONTEXT_MESSAGE
        else:
            state["guardrail_allowed"] = True
            state["guardrail_reason"] = "Allowed by guardrails"
            state["guardrail_message"] = None
            
        state["guardrail_score"] = 1.0
        state["guardrail_context_quality"] = "high" if state["guardrail_allowed"] else "empty"
        
        return state
        
    except Exception as e:
        logger.error(f"Critical error in guardrail enforcement: {e}")
        state["guardrail_allowed"] = False
        state["guardrail_reason"] = f"Internal error: {str(e)}"
        state["guardrail_message"] = OUT_OF_SCOPE_MESSAGE
        return state

def get_guardrail_response(state: Dict[str, Any]) -> Optional[str]:
    if not state.get("guardrail_allowed", True):
        return state.get("guardrail_message", OUT_OF_SCOPE_MESSAGE)
    return None

def guardrail_status_summary(state: Dict[str, Any]) -> str:
    allowed = state.get("guardrail_allowed", "Unknown")
    reason = state.get("guardrail_reason", "No reason")
    status = "ALLOWED ✅" if allowed else "BLOCKED ❌"
    return f"{status} | Reason: {reason}"
