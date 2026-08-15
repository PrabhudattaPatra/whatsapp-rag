"""
Agent node: generate response or invoke retrieval tools.
Uses NeMo Guardrails and Portkey-based response LLM.
"""

import logfire
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages

from src.config import get_settings
from src.generation.llm import get_response_llm
from src.generation.prompt import AGENT_SYSTEM_PROMPT
from src.graph.state import AgentState
from src.graph.timing import time_node
from src.retrieval.tools import classify_and_retrieve, get_college_images

settings = get_settings()

# --- Initialize Guardrails and Response Model ---
logfire.info("Loading NeMo Guardrails config...", path=settings.guardrails_config_path)
config = RailsConfig.from_path(settings.guardrails_config_path)
guardrails = RunnableRails(config=config, passthrough=True)

# Build the guarded model with tools bound
response_model = get_response_llm()
guarded_response_model = (
    guardrails
    | response_model.bind_tools([classify_and_retrieve, get_college_images])
)
logfire.info("Guarded response model successfully built.")


def build_recent_history_text(messages: list, max_turns: int = 4) -> str:
    """Format the last few turns (excluding the current user message) as plain text."""
    recent = messages[:-1][-max_turns:]
    lines = []
    for m in recent:
        if isinstance(m, dict):
            role, content = m.get("role"), m.get("content")
        else:
            role, content = m.type, m.content
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


@time_node("agent")
@logfire.instrument("generate_query_or_respond_node")
async def generate_query_or_respond(state: AgentState):
    """
    Call the model to generate a response or call a retrieval tool.
    Passes the conversation history directly to the LLM without text manipulation.
    """
    logfire.info("🤖 [Node] generate_query_or_respond")
    messages = convert_to_messages(state["messages"])

    if not messages:
        return {"messages": []}

    # Pass all messages as-is to the guarded model, prefixed with a system
    # message describing its tools — without this, the model tends to
    # answer image requests with a generic "I can't display images"
    # disclaimer instead of calling get_college_images.
    # The LLM understands message arrays natively and will use the latest user message for tool calls
    response = await guarded_response_model.ainvoke([SystemMessage(content=AGENT_SYSTEM_PROMPT), *messages])

    # Initialize image_offsets if not already present
    image_offsets = state.get("image_offsets") or {}

    return {"messages": [response], "image_offsets": image_offsets}
