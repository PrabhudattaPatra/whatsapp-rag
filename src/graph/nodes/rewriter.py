"""
Rewriter node: reformulates the user query for better retrieval performance.
"""

import logfire
from langchain_core.messages import HumanMessage

from src.generation.llm import get_response_llm
from src.generation.prompt import REWRITE_PROMPT
from src.graph.state import AgentState


@logfire.instrument("rewrite_question_node")
def rewrite_question(state: AgentState):
    """
    Rewrite the original user question to capture semantic meaning better.
    """
    logfire.info("🔄 [Node] rewrite_question")
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    
    try:
        llm = get_response_llm()
        response = llm.invoke([{"role": "user", "content": prompt}])
        logfire.info("Question rewritten successfully", original=question, rewritten=response.content)
        return {"messages": [HumanMessage(content=response.content)]}
    except Exception as e:
        logfire.error("Question rewriting failed, returning original question", error=str(e))
        return {"messages": [HumanMessage(content=question)]}
