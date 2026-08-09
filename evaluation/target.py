from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from rag.graph import build_graph

def rag_target(inputs: dict) -> dict:
    """
    Target function for LangSmith evaluation.
    """
    question = inputs["question"]

    # Compile the graph
    graph = build_graph().compile()
    
    # Invoke the graph with the user question
    final_state = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })

    final_answer = None
    final_documents = []

    # Extract the AI's final answer and the retrieved documents
    for msg in final_state["messages"]:
        if isinstance(msg, ToolMessage):
            final_documents.append(msg.content)
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content

    return {
        "answer": final_answer,
        "documents": final_documents
    }
