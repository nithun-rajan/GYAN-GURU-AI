import base64
from typing import Optional
import io
import PyPDF2  # For PDF text extraction
from fastapi import UploadFile, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config import ChatState

# LLM for file content extraction
file_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def analyze_file(state: ChatState, file: Optional[UploadFile] = None) -> ChatState:
    """
    Analyzes an uploaded file, extracts content, and updates the chat state.
    Supports images (via LLM), PDFs (via PyPDF2), and plain text files.
    """
    if file:
        # Read file bytes asynchronously
        file_bytes = await file.read()
        if len(file_bytes) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit.")

        # Store file type
        state["file_type"] = file.content_type

        try:
            if file.content_type.startswith("image"):
                # Encode image as base64 for LLM
                base64_encoded = base64.b64encode(file_bytes).decode("utf-8")
                base64_string = f"data:{file.content_type};base64,{base64_encoded}"
                message_content = [
                    {"type": "text", "text": "Extract all text and describe the content of this image."},
                    {"type": "image_url", "image_url": {"url": base64_string}}
                ]
                response = file_llm.invoke([
                    SystemMessage(content="You are an assistant that extracts text and describes file content."),
                    HumanMessage(content=message_content)
                ])
                state["file_content"] = response.content
            elif file.content_type == "application/pdf":
                # Extract text from PDF using PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                if not text.strip():
                    state["file_content"] = "No readable text found in the PDF."
                else:
                    # Optional: Use LLM to summarize/describe if needed
                    response = file_llm.invoke([
                        SystemMessage(content="You are an assistant that describes file content."),
                        HumanMessage(content=f"Describe this PDF content:\n{text[:2000]}")  # Limit to avoid token overflow
                    ])
                    state["file_content"] = f"Extracted Text: {text}\nDescription: {response.content}"
            elif file.content_type.startswith("text"):
                # Decode plain text files directly
                text = file_bytes.decode("utf-8", errors="replace")
                state["file_content"] = f"Extracted Text: {text}"
            else:
                state["file_content"] = f"Unsupported file type: {file.content_type}"
            # Add to file history
            state["file_history"] = state.get("file_history", []) + [{
            "name": file.filename,
            "type": state["file_type"],
            "content": state["file_content"]}]
        except Exception as e:
            state["file_content"] = f"Error processing file: {str(e)}"
            # Optional: Log the error for debugging
            import logging
            logging.error(f"File processing error: {str(e)}")
    else:
        state["file_type"] = None
        state["file_content"] = None
    return state
