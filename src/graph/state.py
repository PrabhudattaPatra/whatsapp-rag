# src/graph/state.py
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # Dictionary mapping category keys to the number of images shown to this user in this thread.
    image_offsets: dict