#This module handles configuration (e.g., API keys, model settings)

import os
from typing import TypedDict, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load from .env in production
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = "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"
MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
RETRIEVER_K = 3
SIMILARITY_THRESHOLD = 0.6

class ChatState(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]
    context: Optional[str]  # Stores the current step in the solution
    session_id: str
    step_hint: Optional[str]  # Stores the last generated step
    mode: str
    file_type: Optional[str]  # Stores the input file type, if there is one
    file_content: Optional[str]  # Stores the input file, if there is one
    tutor_start_index: Optional[int]  # Index where tutor mode begins
    tutor_ai_count: Optional[int]    # Count of AI messages in tutor mode
    file_history: Optional[List[dict]]  # e.g., [{"name": "file1.pdf", "type": "application/pdf", "content": "..."}]
