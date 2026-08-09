from langsmith import Client
from evaluation.target import rag_target
from evaluation.datasets import DATASET_NAME
from evaluation.evaluators.correctness import correctness
from evaluation.evaluators.relevance import relevance
from evaluation.evaluators.groundedness import groundedness
from evaluation.evaluators.retrieval_relevance import retrieval_relevance

def run():
    client = Client()
    results = client.evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[
            correctness,
            relevance,
            groundedness,
            retrieval_relevance,
        ],
        experiment_prefix="cgu-rag-eval",
        metadata={"version": "langgraph-prod"}
    )
    print(results.to_pandas())

if __name__ == "__main__":
    run()
