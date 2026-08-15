"""
Shared helper for pulling the "current" user question out of a message list.

Why this exists:
    The Postgres checkpointer persists the full message history per
    thread_id, so on any turn after the first, `state["messages"][0]` is
    whatever the user asked when the conversation *started* — not the
    live question. grade_documents/rewrite_question/generate_answer all
    need the question for the turn in progress, so they must scan from
    the end of the list rather than index into the front of it.
"""

from langchain_core.messages import BaseMessage, HumanMessage


def get_current_question(messages: list[BaseMessage]) -> str:
    """
    Return the content of the most recent HumanMessage in `messages`.

    On a fresh turn this is the user's latest message. Mid rewrite-loop,
    rewrite_question appends its rewritten query as a new HumanMessage,
    so this also picks that up as "the question" for the retry — which is
    the intended behavior, not a bug.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
        if isinstance(message, dict) and message.get("type") == "human":
            return message.get("content", "")
        if isinstance(message, tuple) and len(message) == 2 and message[0] in ("user", "human"):
            return message[1]

    # Nothing recognizable as a human turn — fall back to the first message
    # rather than raising, so a malformed/empty history doesn't crash a node.
    return messages[0].content if messages else ""
