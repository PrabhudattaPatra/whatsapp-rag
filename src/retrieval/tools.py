# src/retrieval/tools.py
import json
from pathlib import PurePosixPath
import logfire
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from qdrant_client import models

from src.config import get_settings
from src.vector_store.qdrant_client import get_qdrant_client
from src.vector_store import get_image_vector_store, get_image_docstore
from src.vector_store.redis_store import IMAGE_DOC_ID_KEY
from src.embeddings import get_dense_embeddings
from src.retrieval.reranker import rerank_documents
from langchain_qdrant import QdrantVectorStore, RetrievalMode

settings = get_settings()
qdrant_client = get_qdrant_client()
dense_embeddings = get_dense_embeddings()

# Built once per process, not per call. QdrantVectorStore.__init__ validates
# the collection config by default (a GET .../collections/{name} call, plus
# an extra throwaway embedding call to check vector-size compatibility — see
# _validate_collection_for_dense in langchain_qdrant/qdrant.py). Rebuilding
# this inline inside classify_and_retrieve on every turn was paying that
# validation cost (~1.4s Qdrant round-trip + a wasted Fireworks embed call)
# on every single request for a schema that never changes at runtime.
text_vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=settings.qdrant_text_collection,
    embedding=dense_embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    vector_name="dense",
)

# --- 1. Query Classifier ---
class QueryCategory(BaseModel):
    doc_type: Literal["faq", "fee_structure", "notice_board", "examination_cell"] = Field(
        description="The single most relevant category"
    )

def get_classifier():
    llm = ChatGroq(model=settings.groq_model, temperature=0, api_key=settings.groq_api_key)
    return llm.with_structured_output(QueryCategory)

# --- 2. Text Retrieval Tool ---
# response_format="content_and_artifact" lets this tool return a second,
# LLM-invisible value alongside its text content — used here to carry the
# top Cohere rerank score out to grade_documents (src/graph/nodes/grader.py),
# which grades relevance with a plain threshold on that score instead of an
# extra LLM call. Every return path below must return a 2-tuple; that's a
# hard requirement of this response_format, not a style choice.
@tool(response_format="content_and_artifact")
def classify_and_retrieve(query: str) -> tuple[str, dict]:
    """Retrieve relevant college information for a student or parent query."""
    logfire.info("🔍 [Tool] classify_and_retrieve triggered", query=query)
    print(f"\n[DEBUG] classify_and_retrieve called with query: '{query}'")

    try:
        # Classify
        classifier = get_classifier()
        result = classifier.invoke([
            {"role": "system", "content": "Classify the query into exactly one category."},
            {"role": "user", "content": query},
        ])
        doc_type = result.doc_type
        logfire.info(f"🏷️ Classified as: {doc_type}")
        print(f"[DEBUG] Classified as: {doc_type}")

        # Retrieve (reuses the module-level text_vectorstore — see comment above)
        base_retriever = text_vectorstore.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.doc_type", match=models.MatchValue(value=doc_type)),
                        models.FieldCondition(key="metadata.academic_year", match=models.MatchValue(value="2026-27")),
                        models.FieldCondition(key="metadata.is_active", match=models.MatchValue(value=True)),
                    ]
                ),
            }
        )

        docs = base_retriever.invoke(query)

        # Rerank
        reranked_docs = rerank_documents(query, docs, top_n=3)

        if not reranked_docs:
            return f"No relevant documents found for category '{doc_type}'.", {
                "max_relevance_score": 0.0,
                "doc_type": doc_type,
            }

        # CohereRerank stamps metadata["relevance_score"] on every doc it
        # returns (src/retrieval/reranker.py) — take the best of the top-3.
        max_relevance_score = max(
            (doc.metadata.get("relevance_score") or 0.0) for doc in reranked_docs
        )
        content = "\n\n---\n\n".join(doc.page_content for doc in reranked_docs)
        return content, {"max_relevance_score": max_relevance_score, "doc_type": doc_type}

    except Exception as e:
        logfire.error("❌ [Tool] classify_and_retrieve failed", error=str(e))
        return "An error occurred while retrieving information.", {"max_relevance_score": 0.0}

# --- 3. Image Retrieval Tool ---
@tool
def get_college_images(query: str) -> str:
    """Retrieve images relevant to a student/parent query (e.g., classroom, canteen, campus, library, hostel)."""
    logfire.info("🖼️ [Tool] get_college_images triggered", query=query)

    try:
        # image_documents holds one Document per category (caption text,
        # e.g. "Classroom with desks, whiteboard, and chairs") — find the
        # closest matching category to the user's query.
        vector_store = get_image_vector_store()
        matches = vector_store.similarity_search(query, k=1)
        if not matches:
            return json.dumps({"images": [], "message": "No matching image category found."})

        doc = matches[0]
        doc_id = doc.metadata.get(IMAGE_DOC_ID_KEY)
        category = doc.metadata.get("category", "")

        # The actual file paths for that category live in Redis, keyed by
        # the same doc_id (see src/ingestion/image_indexer.py).
        docstore = get_image_docstore()
        raw = docstore.mget([doc_id])[0]
        if not raw:
            return json.dumps({"images": [], "message": f"No images found for '{category}'."})

        local_paths = json.loads(raw)

        # Local filesystem paths (e.g. "images/classroom/images (5).jpg")
        # aren't fetchable by a browser — rewrite to the /media URL the API
        # serves them under (see STATIC_MEDIA_MOUNT in api/main.py).
        image_urls = [
            "/media/" + PurePosixPath(*PurePosixPath(p.replace("\\", "/")).parts[1:]).as_posix()
            for p in local_paths
        ]

        logfire.info("Image retrieval matched category", category=category, count=len(image_urls))
        return json.dumps({
            "images": image_urls,
            "category": category,
            "message": f"Found {len(image_urls)} image(s) for '{category}'.",
        })
    except Exception as e:
        logfire.error("❌ [Tool] get_college_images failed", error=str(e))
        return json.dumps({"images": [], "message": "An error occurred while retrieving images."})