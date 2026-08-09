"""
LangGraph workflow: builds the StateGraph and compiles it with the
async Postgres checkpointer for persistent conversation memory.

Why split build_workflow / init_graph:
    `AsyncPostgresSaver` must be opened inside an async context manager
    (it starts a connection pool). `build_workflow()` is sync and just
    wires together nodes and edges — it cannot block on async I/O.
    `init_graph()` is async: it awaits the checkpointer, then compiles the
    graph. Call `init_graph()` once at application startup.

Usage:
    # In FastAPI lifespan or main async entrypoint:
    from src.graph.workflow import init_graph, graph
    await init_graph()
    # `graph` is now ready to use
"""

import logfire
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.graph.state import AgentState
from src.graph.nodes import (
    generate_query_or_respond,
    grade_documents,
    rewrite_question,
    generate_answer,
)
from src.retrieval.tools import classify_and_retrieve, get_college_images
from src.graph.checkpointer import get_checkpointer

# Instrument LangGraph with Logfire
logfire.instrument()

# The compiled graph — None until init_graph() is called at startup.
graph = None


def build_workflow() -> StateGraph:
    """
    Wire together all nodes and edges. Returns the raw StateGraph.
    Does NOT compile — the checkpointer is injected at startup in init_graph().
    """
    workflow = StateGraph(AgentState)

    # --- Nodes ---
    workflow.add_node("agent", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([classify_and_retrieve, get_college_images]))
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("generate", generate_answer)

    # --- Edges ---
    workflow.add_edge(START, "agent")

    # Route from agent: if the model issued tool calls, go to retrieve. Else, end.
    def route_agent(state: AgentState):
        last_msg = state["messages"][-1]
        if getattr(last_msg, "tool_calls", None):
            return "retrieve"
        return END

    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {"retrieve": "retrieve", END: END},
    )

    # After retrieval, grade the docs to decide whether to answer or rewrite.
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate",
            "rewrite_question": "rewrite",
        },
    )

    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)

    return workflow


async def init_graph():
    """
    Compile the workflow with the AsyncPostgresSaver checkpointer.

    Must be awaited exactly once at application startup, before any
    requests are handled. Sets the module-level `graph` variable.
    """
    global graph

    logfire.info("Building LangGraph workflow...")
    workflow = build_workflow()

    logfire.info("Attaching AsyncPostgresSaver checkpointer...")
    checkpointer = await get_checkpointer()

    graph = workflow.compile(checkpointer=checkpointer)
    logfire.info("LangGraph graph compiled and ready.")
    return graph