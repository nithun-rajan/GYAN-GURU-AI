from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from config import ChatState

# Classification LLM (lightweight for speed)
classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt for classification
classification_prompt = PromptTemplate(
    input_variables=["query", "history"],
    template="""
    You are a query classifier for a chatbot with three modes: 'general', 'tutor', and 'religious'.
    Given the latest user query and the conversation history, classify the query into one of these modes:
    - 'general': For casual conversation, facts, or non-educational queries.
    - 'tutor': For educational queries, problem-solving, or follow-ups to ongoing tutoring (e.g., math, science).
    - 'religious': For queries about religious topics or texts, specially Swaminarayan Religion.
    
    Rules:
    - If the conversation is already in 'tutor' mode (check history), keep it 'tutor' for follow-ups unless the query clearly shifts to a new unrelated topic.
    - Use the history to understand context and intent.
    
    Conversation History:
    {history}
    
    Latest Query:
    {query}
    
    Output only the mode name ('general', 'tutor', or 'religious').
    """
)

# Node to classify the query
def classify_query(state: ChatState) -> ChatState:
    query = state["messages"][-1].content
    conversation_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"][:-1]])
    classification_chain = classification_prompt | classifier_llm
    category = classification_chain.invoke({"query": query, "history": conversation_history}).content.strip()
    if category == "tutor" and state["mode"] != "tutor":
        state["tutor_start_index"] = len(state["messages"]) - 1  # Mark where tutor mode starts
        state["tutor_ai_count"] = 0  # Reset tutor AI message count
    state["mode"] = category
    return state

# Branching logic based on classification
def branch_logic(state: ChatState) -> str:
    category = state["mode"]
    if category == "religious":
        print(category)
        return "religious_response"
    elif category == "tutor":
        print(category)
        return "tutor_path"
    else:
        print(category)
        return "general_response"