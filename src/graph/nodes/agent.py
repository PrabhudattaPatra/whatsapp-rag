"""
Agent node: generate response or invoke retrieval tools.
Uses NeMo Guardrails and Portkey-based response LLM.
"""

import logfire
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.messages import HumanMessage, convert_to_messages

from src.config import get_settings
from src.generation.llm import get_response_llm
from src.graph.state import AgentState
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


@logfire.instrument("generate_query_or_respond_node")
async def generate_query_or_respond(state: AgentState):
    """
    Call the model to generate a response or call a retrieval tool.
    Pre-processes context via recent message formatting before calling the guarded model.
    """
    logfire.info("🤖 [Node] generate_query_or_respond")
    messages = convert_to_messages(state["messages"])

    if not messages:
        return {"messages": []}

    last_msg = messages[-1]
    
    # Build context from recent messages if history exists
    if len(messages) > 1:
        history_text = build_recent_history_text(messages)
        combined_content = (
            f"Recent conversation context:\n{history_text}\n\n"
            f"Current user message: {last_msg.content}"
        )
    else:
        combined_content = last_msg.content

    # Replace only the last message for the LLM/Guardrails call
    guardrails_messages = messages[:-1] + [HumanMessage(content=combined_content)]

    # Invoke guarded model
    response = await guarded_response_model.ainvoke(guardrails_messages)
    
    # Initialize image_offsets if not already present
    image_offsets = state.get("image_offsets") or {}

    return {"messages": [response], "image_offsets": image_offsets}
