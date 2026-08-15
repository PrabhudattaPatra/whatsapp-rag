"""
Grader node: grade the relevance of retrieved documents to the query.

Uses the Cohere rerank score already computed during retrieval
(classify_and_retrieve's artifact, src/retrieval/tools.py) to skip the
extra LLM grading call for clear-cut cases — but only the clear-cut ones.

Empirically (see calibration run against this project's actual corpus),
scores are extremely polarized: a genuinely well-matched query scored
0.9956, and unrelated queries ("what's the weather today") scored 0.0000.
But the low-but-nonzero middle isn't reliable — a real, useful answer
("what documents are required for admission") scored only 0.0004, in the
same ballpark as the nonsense queries. A single naive threshold would have
turned that good answer into an unnecessary rewrite loop. So: skip the LLM
call only when the score is unambiguous (very high, or ~exactly zero);
fall back to the original LLM-based grading for everything in between,
which is most of the real traffic. Smaller win than a naive threshold, but
doesn't trade away answer quality for it.
"""

from typing import Literal
import logfire
from pydantic import BaseModel, Field

from src.generation.llm import get_grader_llm
from src.generation.prompt import GRADE_PROMPT
from src.graph.message_utils import get_current_question
from src.graph.state import AgentState
from src.graph.timing import time_node

# Calibrated against this project's actual corpus/rerank model — not
# universal Cohere constants. Anything outside these bounds still goes
# through the LLM grader below.
HIGH_CONFIDENCE_RELEVANT = 0.5
HIGH_CONFIDENCE_IRRELEVANT = 1e-6


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


@time_node("grade_documents")
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

    # Fast path: the rerank score already computed during retrieval, when
    # it's unambiguous — see module docstring for why only the extremes.
    artifact = getattr(last_message, "artifact", None) or {}
    rerank_score = artifact.get("max_relevance_score")
    if rerank_score is not None:
        if rerank_score >= HIGH_CONFIDENCE_RELEVANT:
            logfire.info("Rerank score high-confidence relevant, skipping LLM grading", rerank_score=rerank_score)
            return "generate_answer"
        if rerank_score <= HIGH_CONFIDENCE_IRRELEVANT:
            logfire.info("Rerank score high-confidence irrelevant, skipping LLM grading", rerank_score=rerank_score)
            return "rewrite_question"
        logfire.info("Rerank score ambiguous, falling back to LLM grading", rerank_score=rerank_score)

    question = get_current_question(state["messages"])
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
