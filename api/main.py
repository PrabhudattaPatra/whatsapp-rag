"""
FastAPI app: chat API for the CGU RAG assistant.

Why this file exists:
    Exposes the LangGraph RAG pipeline over HTTP so the frontend (a
    separate static site — see frontend/, run independently via
    scripts/run_frontend.py) can talk to it. `POST /api/chat` runs one
    turn of the graph and returns the reply; a `session_id` (generated
    client-side, kept in localStorage) is used as the LangGraph
    `thread_id`, so the Postgres checkpointer gives each browser session
    its own persistent conversation memory.

    On startup, this module builds the same LangGraph `graph` that
    `test_graph.py` builds manually, and reuses it across every request.

Usage:
    uv run python -m scripts.run_api
    # (NOT `uvicorn api.main:app` directly on Windows — see that script's
    # docstring for why: psycopg needs a SelectorEventLoop, which has to be
    # forced before uvicorn picks its own loop class.)
    # then separately: uv run python -m scripts.run_frontend
"""

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.observability import configure_observability
from src.graph import workflow as graph_workflow
from src.graph.workflow import init_graph
from src.graph.checkpointer import close_checkpointer

# Origins the standalone frontend (scripts/run_frontend.py) can be served
# from. Add to this list if you serve the frontend from somewhere else.
FRONTEND_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

# get_college_images (src/retrieval/tools.py) returns local paths like
# "images/classroom/images (5).jpg" rewritten to "/media/classroom/...".
# Serve the project's images/ directory at that same /media prefix so
# those URLs are actually fetchable by the frontend.
IMAGES_DIR = Path(__file__).parent.parent / "images"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_observability()
    logfire.info("Starting up: initializing LangGraph workflow...")
    await init_graph()
    logfire.info("Startup complete.")
    yield
    logfire.info("Shutting down: closing checkpointer connection pool...")
    await close_checkpointer()


app = FastAPI(title="CGU RAG Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

if IMAGES_DIR.is_dir():
    app.mount("/media", StaticFiles(directory=str(IMAGES_DIR)), name="media")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@app.get("/api/health")
async def health():
    return {"status": "ok"}


def _sse(payload: dict) -> str:
    """Format a dict as one Server-Sent-Events frame."""
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    Run one turn of the RAG graph, streaming the final answer's tokens to
    the client as they're generated instead of blocking until the whole
    graph finishes.

    Why streaming, and why only the "generate" node's tokens:
        Every turn does several sequential steps before any answer text
        exists at all (tool routing, retrieval, grading — several seconds
        even after the checkpointer/vectorstore fixes elsewhere in this
        project). Streaming can't shrink that work, but it means the user
        sees the answer appear as it's written instead of staring at a
        typing indicator for the entire turn. Only src/graph/nodes/generator.py's
        `generate_answer` node actually streams tokens (via llm.astream);
        other paths (the agent answering directly without retrieval, or
        the image-tool path) still resolve to a single final message with
        no intermediate tokens — that's fine, they just skip straight to
        the "done" event below.

    `session_id` doubles as the LangGraph `thread_id`, so each browser
    session gets its own persistent conversation history in Postgres. If
    the client doesn't send one (first message of a new chat), a new one
    is generated and returned for the client to remember.

    Event shapes sent to the client (newline-delimited SSE `data:` frames):
        {"type": "token", "content": "..."}                              — one per streamed chunk
        {"type": "done", "session_id": "...", "reply": "...", "images": [...] | null}  — always sent last
        {"type": "error", "message": "..."}                              — only on an unhandled failure
    """
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    async def event_stream():
        try:
            graph = graph_workflow.graph
            async for chunk, metadata in graph.astream(
                {"messages": [("user", req.message)], "image_offsets": {}},
                config=config,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") != "generate":
                    continue
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    yield _sse({"type": "token", "content": content})

            # Streaming above only carries token deltas — pull the final
            # persisted state for the authoritative full reply/images. This
            # is also what covers non-streamed paths (image tool results,
            # or the agent answering directly without retrieval).
            state = await graph_workflow.graph.aget_state(config)
            last_message = state.values["messages"][-1]
            reply = last_message.content
            images = getattr(last_message, "additional_kwargs", {}).get("images") or None
            yield _sse({"type": "done", "session_id": session_id, "reply": reply, "images": images})
        except Exception as e:
            logfire.error("Failed to handle chat message", session_id=session_id, error=str(e))
            yield _sse({
                "type": "error",
                "message": "Sorry, I ran into an error answering that. Please try again.",
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
