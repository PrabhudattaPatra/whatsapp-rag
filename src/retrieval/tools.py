# src/retrieval/tools.py
import json
import logfire
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from qdrant_client import models

from src.config import get_settings
from src.vector_store.qdrant_client import get_qdrant_client
from src.embeddings import get_dense_embeddings
from src.retrieval.reranker import rerank_documents
from langchain_qdrant import QdrantVectorStore, RetrievalMode

settings = get_settings()
qdrant_client = get_qdrant_client()
dense_embeddings = get_dense_embeddings()

# --- 1. Query Classifier ---
class QueryCategory(BaseModel):
    doc_type: Literal["faq", "fee_structure", "notice_board", "examination_cell"] = Field(
        description="The single most relevant category"
    )

def get_classifier():
    llm = ChatGroq(model=settings.groq_model, temperature=0, api_key=settings.groq_api_key)
    return llm.with_structured_output(QueryCategory)

# --- 2. Text Retrieval Tool ---
@tool
def classify_and_retrieve(query: str) -> str:
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

        # Retrieve
        vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name=settings.qdrant_text_collection,
            embedding=dense_embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
        )
        
        base_retriever = vectorstore.as_retriever(
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
            return f"No relevant documents found for category '{doc_type}'."

        return "\n\n---\n\n".join(doc.page_content for doc in reranked_docs)

    except Exception as e:
        logfire.error("❌ [Tool] classify_and_retrieve failed", error=str(e))
        return "An error occurred while retrieving information."

# --- 3. Image Retrieval Tool (Stub for now, we can flesh this out if you still need Redis) ---
@tool
def get_college_images(query: str) -> str:
    """Retrieve images relevant to a student/parent query (e.g., classroom, canteen, campus)."""
    logfire.info("🖼️ [Tool] get_college_images triggered", query=query)
    # TODO: Implement Qdrant image search + Redis lookup here if still needed.
    # For now, returning a placeholder to keep the graph compiling.
    return json.dumps({"images": ["path/to/sample_image.jpg"], "message": "Image retrieval placeholder"})