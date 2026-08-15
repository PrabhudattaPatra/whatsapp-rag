import uuid

from langchain_core.messages import ToolMessage, AIMessage

import src.graph.workflow as workflow_module


async def rag_target(inputs: dict) -> dict:
    """Async target function for LangSmith evaluation — drives the real,
    compiled production graph (workflow_module.graph, set by init_graph()
    at process startup; read off the module rather than imported by name,
    since `graph` is None at import time and only becomes the compiled
    graph after init_graph() runs)."""
    question = inputs["question"]

    # Fresh thread_id per example: the Postgres checkpointer persists full
    # conversation history per thread_id. Reusing one across eval examples
    # would leak prior Q&A into later runs' state.
    thread_id = f"eval-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    final_state = await workflow_module.graph.ainvoke(
        {"messages": [("user", question)], "image_offsets": {}},
        config=config,
    )

    final_answer = None
    final_documents = []

    # Extract the AI's final answer and the retrieved documents
    for msg in final_state["messages"]:
        if isinstance(msg, ToolMessage) and msg.content:
            final_documents.append(msg.content)
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content

    return {
        "answer": final_answer,
        # retrieval_relevance.py assumes outputs["documents"] is a single
        # STRING (the joined ToolMessage content) — join with the same
        # separator classify_and_retrieve itself uses between chunks.
        "documents": "\n\n---\n\n".join(final_documents),
    }
