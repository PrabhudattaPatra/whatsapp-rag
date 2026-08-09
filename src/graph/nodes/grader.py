"""
Grader node: grade the relevance of retrieved documents to the query.
"""

from typing import Literal
import logfire
from pydantic import BaseModel, Field

from src.generation.llm import get_grader_llm
from src.generation.prompt import GRADE_PROMPT
from src.graph.state import AgentState


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


@logfire.instrument("grade_documents_node")
def grade_documents(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    """
    Determine whether the retrieved documents are relevant to the question.
    Returns the string representing the next node to transition to.
    """
    logfire.info("📝 [Node] grade_documents")
    last_message = state["messages"][-1]

    # Skip grading for image tool results — go straight to answer generation
    if getattr(last_message, "name", None) == "get_college_images":
        logfire.info("Skipping grading for image retrieval tool results")
        return "generate_answer"

    question = state["messages"][0].content
    context = last_message.content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    try:
        grader_llm = get_grader_llm()
        grader = grader_llm.with_structured_output(GradeDocuments)
        response = grader.invoke([{"role": "user", "content": prompt}])
        score = response.binary_score.strip().lower()
        logfire.info("Document relevance grade completed", binary_score=score)
        
        if score == "yes":
            return "generate_answer"
        return "rewrite_question"
    except Exception as e:
        logfire.error("Document grading failed, falling back to 'generate_answer'", error=str(e))
        return "generate_answer"
