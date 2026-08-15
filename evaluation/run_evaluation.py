import asyncio
import sys

import src.config  # noqa: F401 -- ensures .env is loaded before Client()/settings reads
from langsmith import Client

from src.graph.workflow import init_graph
from evaluation.target import rag_target
from evaluation.datasets import DATASET_NAME, EXAMPLES, create_dataset
from evaluation.evaluators.correctness import correctness
from evaluation.evaluators.relevance import relevance
from evaluation.evaluators.groundedness import groundedness
from evaluation.evaluators.retrieval_relevance import retrieval_relevance


async def run():
    # Compile the graph once, before evaluation starts -- NOT per example.
    # rag_target() reads the module-level global this sets.
    print("[*] Initializing production graph...")
    await init_graph()
    print("[OK] Graph initialized.")

    client = Client()

    # Idempotent dataset bootstrap: create the dataset if it doesn't exist
    # yet, then top up any examples in EXAMPLES that aren't already present
    # remotely (by question text) -- covers the case where the dataset was
    # created earlier from a smaller/stale EXAMPLES list and needs syncing,
    # not just "exists vs doesn't".
    if not client.has_dataset(dataset_name=DATASET_NAME):
        print(f"[*] Dataset '{DATASET_NAME}' not found -- creating it...")
        create_dataset()
    else:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing_questions = {
            e.inputs.get("question") for e in client.list_examples(dataset_id=dataset.id)
        }
        missing = [
            ex for ex in EXAMPLES if ex["inputs"]["question"] not in existing_questions
        ]
        if missing:
            print(
                f"[*] Dataset '{DATASET_NAME}' exists with {len(existing_questions)} "
                f"examples -- adding {len(missing)} missing example(s)..."
            )
            client.create_examples(dataset_id=dataset.id, examples=missing)
        else:
            print(
                f"[OK] Dataset '{DATASET_NAME}' already has all "
                f"{len(existing_questions)} examples -- reusing it."
            )

    results = await client.aevaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=[
            correctness,
            relevance,
            groundedness,
            retrieval_relevance,
        ],
        experiment_prefix="cgu-rag-eval",
        metadata={"version": "langgraph-prod"},
    )
    print(results.to_pandas())


if __name__ == "__main__":
    # Windows fix: psycopg's AsyncPostgresSaver requires SelectorEventLoop,
    # not the default ProactorEventLoop. Must be set before asyncio.run().
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run())
