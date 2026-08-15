"""
Generator node: generates final RAG answers or formats image results.
"""

import json
import logfire
from langchain_core.messages import AIMessage

from src.generation.llm import get_response_llm
from src.generation.prompt import GENERATE_PROMPT
from src.graph.message_utils import get_current_question
from src.graph.state import AgentState
from src.graph.timing import time_node


@time_node("generate")
@logfire.instrument("generate_answer_node")
async def generate_answer(state: AgentState):
    """
    Generate an answer from the user question and the retrieved context.
    If the last tool call retrieved images, handles formatting rather than calling the LLM.
    """
    logfire.info("✅ [Node] generate_answer")
    last_message = state["messages"][-1]

    # If the last tool call was the image retriever, don't run it through the LLM —
    # just pass the image data through as-is for the frontend to render.
    if getattr(last_message, "name", None) == "get_college_images":
        try:
            data = json.loads(last_message.content)
            images = data.get("images", [])
        except (json.JSONDecodeError, AttributeError, TypeError):
            images = []

        if images:
            reply = "Here are some pictures:"
        else:
            reply = "Sorry, I couldn't find any relevant pictures."

        logfire.info("Formatting image retrieval tool results response", count=len(images))
        return {
            "messages": [
                AIMessage(
                    content=reply,
                    additional_kwargs={"images": images},
                )
            ]
        }

    # Normal text-based answer generation
    question = get_current_question(state["messages"])
    context = last_message.content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    
    try:
        llm = get_response_llm()
        # Stream instead of .invoke() — this is what lets LangGraph's
        # stream_mode="messages" surface individual token chunks (tagged
        # with this node's name) to api/main.py, instead of the API only
        # being able to hand the frontend one big blob once the whole
        # answer is done.
        chunk = None
        async for piece in llm.astream([{"role": "user", "content": prompt}]):
            chunk = piece if chunk is None else chunk + piece
        if chunk is None:
            raise ValueError("LLM stream produced no chunks")
        logfire.info("Answer generated successfully")
        return {"messages": [chunk]}
    except Exception as e:
        logfire.error("Answer generation failed", error=str(e))
        return {
            "messages": [
                AIMessage(
                    content="I encountered an error trying to answer your question. Please try again later."
                )
            ]
        }
