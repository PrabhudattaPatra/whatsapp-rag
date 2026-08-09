import asyncio
import pytest
import json
from src.observability import configure_observability
from src.retrieval.tools import classify_and_retrieve, get_college_images
import src.graph.workflow as workflow_module
from src.graph.workflow import init_graph
from src.graph.checkpointer import close_checkpointer

# Configure observability once for the test run
configure_observability()

# Session-scoped event loop — WindowsSelectorEventLoopPolicy is already
# applied in conftest.py so new_event_loop() returns a SelectorEventLoop.
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_graph():
    """Initialize the compiled LangGraph with Postgres checkpointer before any tests run."""
    await init_graph()
    yield
    await close_checkpointer()


def test_classify_and_retrieve_fees():
    """Verify that classify_and_retrieve classifies and retrieves fee structure information."""
    query = "What is the fee structure for Btech?"
    result = classify_and_retrieve.invoke({"query": query})
    assert isinstance(result, str)
    assert len(result) > 0
    # The result should contain some content retrieved from the database
    assert "no relevant active documents" not in result.lower()


def test_classify_and_retrieve_faq():
    """Verify that classify_and_retrieve classifies and retrieves faq information."""
    query = "What are the eligibility criteria for Btech?"
    result = classify_and_retrieve.invoke({"query": query})
    assert isinstance(result, str)
    assert len(result) > 0
    assert "no relevant active documents" not in result.lower()


def test_get_college_images():
    """Verify that get_college_images works and paginates correctly."""
    result = get_college_images.invoke({"query": "Give me image of classrooms", "limit": 2, "offset": 0})
    data = json.loads(result)
    assert "images" in data
    assert isinstance(data["images"], list)
    
    # Verify offset and pagination structure
    if "has_more" in data:
        assert isinstance(data["has_more"], bool)


@pytest.mark.asyncio
async def test_workflow_execution():
    """Verify end-to-end execution of the compiled LangGraph workflow."""
    thread_id = "test-thread-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    state = await workflow_module.graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "What is the fee structure for MCA?"}],
            "image_offsets": {}
        },
        config=config
    )
    
    assert "messages" in state
    assert len(state["messages"]) >= 2
    
    # The last message should be from the assistant generating the answer
    last_message = state["messages"][-1]
    assert last_message.type == "ai"
    assert len(last_message.content) > 0
    assert "error" not in last_message.content.lower()


@pytest.mark.asyncio
async def test_workflow_image_query():
    """Verify workflow execution for an image query."""
    thread_id = "test-thread-2"
    config = {"configurable": {"thread_id": thread_id}}
    
    state = await workflow_module.graph.ainvoke(
        {
            "messages": [{"role": "user", "content": "Give me image of classrooms"}],
            "image_offsets": {}
        },
        config=config
    )
    
    assert "messages" in state
    assert len(state["messages"]) >= 2
    
    last_message = state["messages"][-1]
    assert last_message.type == "ai"
    # It should say something about pictures and contain the structured images in additional_kwargs
    assert "pictures" in last_message.content.lower() or "images" in last_message.content.lower()
    assert "images" in last_message.additional_kwargs
