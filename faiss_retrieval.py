from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K, SIMILARITY_THRESHOLD, ChatState

def initialize_faiss(file_path: str = "text.txt"):
    """
    Initialize FAISS vectorstore with a retriever.
    Returns retriever, vectorstore, and threshold from config.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load() or [Document(page_content="Fallback text", metadata={"source": "manual"})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding)
    return vectorstore  # Return raw vectorstore, not retriever

def retrieve_context(vectorstore: FAISS, query: str, similarity_threshold: float = SIMILARITY_THRESHOLD) -> str:
    """
    Retrieve context from FAISS using similarity_search_with_score.
    Only returns documents where similarity score >= threshold.
    """
    try:
        # Use similarity_search_with_score to get documents with their scores
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=RETRIEVER_K)
        # Filter documents based on similarity threshold
        # Note: FAISS returns distance (lower is better), so we convert to similarity (1 - distance)
        filtered_docs = [
            doc for doc, score in docs_with_scores 
            if (1 - score) >= similarity_threshold  # Convert distance to similarity
        ]
        return "\n".join([doc.page_content for doc in filtered_docs]) if filtered_docs else "No relevant context found."
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

def add_retrieval_to_graph(state: ChatState, vectorstore: FAISS) -> ChatState:
    """
    Add FAISS retrieval to the graph state.
    Always runs retrieval but only includes results above the threshold.
    """
    query = state["messages"][-1].content  # Only use the query, not file_content
    state["context"] = retrieve_context(vectorstore, query)
    return state