#main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from llm_utils import initialize_llm
from chat_memory import create_chat_graph, invoke_chat
from typing import Optional
from config import ChatState
from faiss_retrieval import initialize_faiss

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
# Initialize LLM
llm = initialize_llm()
vectorstore = initialize_faiss("text.txt")
graph = create_chat_graph(llm, vectorstore=vectorstore)

@app.post("/")
async def query_llm(
    query: str = Form(...),
    session_id: str = Form("default"),
    file: Optional[UploadFile] = File(None)
):
    # Pass the file directly to the workflow via config
    config = {"configurable": {"thread_id": session_id, "file": file}}
    # Invoke the LangGraph workflow
    response = await invoke_chat(graph, query, session_id, config=config)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)