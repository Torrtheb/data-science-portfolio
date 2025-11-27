from fastapi import APIRouter, Depends, HTTPException
import os
from langchain_core.messages import HumanMessage, SystemMessage
from agent.llm import bound_llm_with_tools
from agent.prompts import SYSTEM_PROMPT
from app.core.auth import require_owner, TokenUser

router = APIRouter(prefix="/api/dev", tags=["dev"])

DEBUG_ENDPOINTS = os.getenv("DEBUG_ENDPOINTS", "0") == "1"


@router.get("/dry_run")
async def dry_run(q: str, user: TokenUser = Depends(require_owner)):
    """Invoke the bound LLM with tools using the system prompt (DEV only).

    Requires 'DEBUG_ENDPOINTS=1'. Returns message metadata and any tool-calls
    produced by the model for quick inspection during development.

    Args:
        q: User message to pass to the model.
        user: Authenticated owner (authorization enforced for consistency).

    Returns:
        Dict containing 'type', 'content', 'tool_calls', and 'kwargs' as seen
        on the LangChain response object.

    Raises:
        HTTPException: 404 if debug endpoints are disabled.
    """
    if not DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Not found")
    llm = bound_llm_with_tools()
    resp = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(q)])
    return {
        "type": getattr(resp, "type", None),
        "content": getattr(resp, "content", None),
        "tool_calls": getattr(resp, "tool_calls", None),
        "kwargs": getattr(resp, "additional_kwargs", None),
    }
