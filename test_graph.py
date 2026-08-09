"""
Quick test script to verify LangGraph workflow is working.

Usage:
    python test_graph.py
"""

import asyncio
from src.graph.workflow import init_graph


async def test_graph():
    """Test the graph with a simple query."""

    print("[*] Initializing graph...")
    try:
        graph = await init_graph()
        print("[OK] Graph initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize graph: {e}")
        return

    # Test configuration - use a NEW thread_id to avoid old conversation state
    import random
    thread_id = f"test-thread-{random.randint(1000, 9999)}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"[INFO] Using thread_id: {thread_id}")

    # Simple test query
    test_query = "What is the fee structure for Btech?"

    print(f"\n[TEST] Query: '{test_query}'")
    print("=" * 60)

    try:
        # Stream the graph execution
        async for event in graph.astream(
            {"messages": [("user", test_query)], "image_offsets": {}},
            config=config,
            stream_mode="values"
        ):
            # Print the latest message from each node
            if "messages" in event:
                last_msg = event["messages"][-1]
                print(f"\n[{last_msg.type.upper()}]: {last_msg.content[:200]}...")

        print("\n" + "=" * 60)
        print("[OK] Graph execution completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Graph execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if .env exists
    import os
    if not os.path.exists(".env"):
        print("[ERROR] .env file not found! Please create one with required API keys.")
        exit(1)

    # Windows fix: psycopg requires SelectorEventLoop, not ProactorEventLoop
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(test_graph())
