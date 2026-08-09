"""
Generator node: generates final RAG answers or formats image results.
"""

import json
import logfire
from langchain_core.messages import AIMessage

from src.generation.llm import get_response_llm
from src.generation.prompt import GENERATE_PROMPT
from src.graph.state import AgentState


@logfire.instrument("generate_answer_node")
def generate_answer(state: AgentState):
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
    question = state["messages"][0].content
    context = last_message.content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    
    try:
        llm = get_response_llm()
        response = llm.invoke([{"role": "user", "content": prompt}])
        logfire.info("Answer generated successfully")
        return {"messages": [response]}
    except Exception as e:
        logfire.error("Answer generation failed", error=str(e))
        return {
            "messages": [
                AIMessage(
                    content="I encountered an error trying to answer your question. Please try again later."
                )
            ]
        }
