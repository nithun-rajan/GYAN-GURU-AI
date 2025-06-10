# This module handles chat history management plus a lot of LangGraph functions and implementations (LangGraph state and persistence)

from config import ChatState
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional, Dict, Callable
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from query_classifier import classify_query, branch_logic
from faiss_retrieval import add_retrieval_to_graph
from file_processing import analyze_file

def generate_normal(state: ChatState, llm) -> ChatState:
    """Generates a normal LLM response without being a tutor or religious assistant."""
    # Include the full conversation history
    conversation_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"]])
    file_history = "\n".join([f"File {f['name']}: {f['content']}" for f in state.get("file_history", [])]) or "No file history."
    messages = [
        SystemMessage(content="You are a normal AI assistant. Answer queries clearly and in a structured manner."),
        HumanMessage(content=f"Conversation History:\n{conversation_history}\n\nFile History:\n{file_history}")
    ]
    response = llm.invoke(messages)
    state["messages"].append(AIMessage(content=response.content))
    return state

def generate_religious(state: ChatState, llm) -> ChatState:
    """Generates a response to a query about the sacred texts, explaining them based on the FAISS retrieval."""
    # Include the full conversation history
    conversation_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"]])
    context = state["context"] or "No sacred text context available."
    file_history = "\n".join([f"File {f['name']}: {f['content']}" for f in state.get("file_history", [])]) or "No file history."
    messages = [
        SystemMessage(content="""
        You are a tutor and expert on Swaminarayan wisdom for students. Your role is to explain and teach the students based on their questions.
        Provide responses with deep meaning, with examples if possible, while keeping it understandable for a child.
        """),
        HumanMessage(content=f"""
        Conversation History:
        {conversation_history}

        I have the following sacred text context: {context}
        I have the following file history: {file_history}
        Please provide a thoughtful response.
        """)
    ]
    response = llm.invoke(messages)
    state["messages"].append(AIMessage(content=response.content))
    return state

def generate_step(state: ChatState, llm) -> ChatState:
    conversation_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"]])
    context = state["context"] or "Unknown"
    file_history = "\n".join([f"File {f['name']}: {f['content']}" for f in state.get("file_history", [])]) or "No file history."
    messages = [
        SystemMessage(content="""
        You are a tutor guiding a student through a problem naturally and conversationally.
        - Provide the next logical guidance or explanation based on the student's progress and query.
        - Do not use numbered steps (e.g., "Step 1") unless the student asks for them.
        - Keep explanations clear, concise, and adaptive to the conversation.
        - Do not reveal the final answer unless explicitly asked.
        """),
        HumanMessage(content=f"""
        Conversation History:
        {conversation_history}
        Problem: {context}
        File content (if any): {file_history}
        Student's progress: {state.get('step_hint', 'No progress yet')}
        What should I do or understand next?
        """)
    ]
    response = llm.invoke(messages)
    state["step_hint"] = response.content.strip()
    state["messages"].append(AIMessage(content=state["step_hint"]))
    state["tutor_ai_count"] = state.get("tutor_ai_count", 0) + 1  # Increment
    return state

def check_student_response(state: ChatState, llm) -> ChatState:
    if not isinstance(state["messages"][-1], HumanMessage):
        return state  # Skip if no student input

    student_response = state["messages"][-1].content
    conversation_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"][:-1]])
    step_hint = state["step_hint"] or "No previous guidance provided yet."

    messages = [
        SystemMessage(content="""
        You are a tutor guiding a student through a problem conversationally.
        Based on the conversation history and the student’s latest response, determine their intent:
        - If they are asking to revisit a previous step or explain something in more detail, provide a clear, detailed explanation of the relevant part without moving forward.
        - If they are providing an answer to your latest guidance or asking to move forward, evaluate if it is correct:
          - If correct, reply 'CORRECT' and offer to proceed.
          - If incorrect, provide a hint or correction WITHOUT revealing the full answer.
          - If unclear, ask for clarification.
        Use the conversation history to ensure your response fits the context and the student’s current needs.
        """),
        HumanMessage(content=f"""
        Conversation History:
        {conversation_history}
        Previous guidance (if any): {step_hint}
        Student’s response: {student_response}
        """)
    ]

    response = llm.invoke(messages)
    feedback = response.content.strip()
    state["messages"].append(AIMessage(content=feedback))
    # Update state based on response
    if "CORRECT" in feedback:
        state["step_hint"] = None  # Clear for next guidance
    state["tutor_ai_count"] = state.get("tutor_ai_count", 0) + 1
    return state

def create_chat_graph(llm, extra_nodes: Optional[Dict[str, Callable]] = None, extra_edges: Optional[Dict[str, str]] = None, vectorstore=None):
    """Creates a structured tutoring chat graph with step-by-step logic."""
    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("add_retrieval_to_graph", lambda state: add_retrieval_to_graph(state, vectorstore))
    workflow.add_node("generate_religious", lambda state: generate_religious(state, llm))
    workflow.add_node("generate_step", lambda state: generate_step(state, llm))
    workflow.add_node("check_student_response", lambda state: check_student_response(state, llm))
    workflow.add_node("generate_normal", lambda state: generate_normal(state, llm))

    # Define edges
    workflow.set_entry_point("classify_query")
    workflow.add_edge("classify_query", "add_retrieval_to_graph")
    workflow.add_conditional_edges(
        "add_retrieval_to_graph",
        branch_logic,
        {
            "religious_response": "generate_religious",
            "tutor_path": "generate_step",
            "general_response": "generate_normal"
        }
    )
    workflow.add_edge("generate_religious", END)
    workflow.add_edge("generate_normal", END)
    workflow.add_edge("generate_step", "check_student_response")

    # Combined condition for check_student_response
    def tutor_path_condition(state: ChatState) -> str:
        if state["mode"] != "tutor":
            return "end"
        # Count messages since tutor mode started
        tutor_start_index = state.get("tutor_start_index", -1)
        messages_since_tutor_start = state["messages"][tutor_start_index:] if tutor_start_index >= 0 else state["messages"]
        # If this is the first tutor query (only one HumanMessage since start)
        if len(messages_since_tutor_start) == 1 and isinstance(messages_since_tutor_start[0], HumanMessage):
            return "generate_step"
        # Check tutor AI responses
        tutor_ai_count = state.get("tutor_ai_count", 0)
        if tutor_ai_count >= 5:  # Limit to 5 hints
            return "end"       
        # If no new student input after guidance, end
        if state["step_hint"] is not None and not isinstance(state["messages"][-1], HumanMessage):
            return "end"
        # Otherwise, check the student’s response
        return "check_student_response"
    
    workflow.add_conditional_edges(
        "check_student_response",
        tutor_path_condition,
        {
            "generate_step": "generate_step",
            "check_student_response": "check_student_response",
            "end": END
        }
    )
    # Add extra nodes and edges if provided
    if extra_nodes:
        for node_name, node_func in extra_nodes.items():
            workflow.add_node(node_name, node_func)
    if extra_edges:
        for start, end in extra_edges.items():
            workflow.add_edge(start, end)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

async def invoke_chat(graph, query: str, session_id: str, context: Optional[str] = None, config: Optional[dict] = None) -> str:
    if config is None:
        config = {"configurable": {"thread_id": session_id}}
    else:
        config["configurable"]["thread_id"] = session_id

    current_state = graph.get_state(config).values or {
        "messages": [],
        "context": context or None,
        "session_id": session_id,
        "step_hint": None,
        "mode": "",
        "file_type": None,
        "file_content": None,
        "tutor_start_index": -1,
        "tutor_ai_count": 0,
        "file_history": []  # Initialize file history
    }
    # Process file before graph invocation
    file = config["configurable"].get("file") if "configurable" in config else None
    if file:
        current_state = await analyze_file(current_state, file)
    current_state["messages"].append(HumanMessage(content=query))  # Add the query
    config["recursion_limit"] = 50  # Keep this if needed for debugging
    updated_state = await graph.ainvoke(current_state, config=config)
    return updated_state["messages"][-1].content