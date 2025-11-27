from __future__ import annotations
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State container passed between graph nodes.

    Keys
    - 'messages': Conversation history. The value is an annotated list of
      'BaseMessage' that merges via the 'add_messages' reducer.
    - 'summary': Optional running summary of the conversation.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    summary: NotRequired[str]
