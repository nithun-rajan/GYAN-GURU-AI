markdown# Gyan Guru AI 🕉️

An intelligent tutoring and spiritual guidance chatbot powered by LangGraph, OpenAI, and FAISS vector search. Gyan Guru AI serves as both an educational tutor and a knowledgeable guide for Swaminarayan religious teachings.

## 🌟 Features

### 🎯 Multi-Modal Intelligence
- **General Assistant**: Handles everyday queries and casual conversation
- **Educational Tutor**: Provides step-by-step guidance for problem-solving with conversational tutoring
- **Religious Guide**: Expert knowledge on Swaminarayan philosophy, scriptures, and teachings

### 📁 File Processing
- **Image Analysis**: Extract text and analyze visual content using GPT-4o-mini
- **PDF Processing**: Text extraction and content analysis from PDF documents
- **Text Files**: Direct processing of plain text documents
- **File History**: Maintains context across multiple file uploads

### 🧠 Advanced Features
- **Vector Search**: FAISS-powered semantic search through religious texts
- **Conversation Memory**: Persistent chat history with session management
- **Smart Classification**: Automatic query categorization for appropriate response modes
- **Step-by-Step Tutoring**: Interactive problem-solving with hints and guidance

### Core Components

1. **Query Classifier**: Intelligently routes queries to appropriate handlers
2. **Chat Memory**: Manages conversation state and history
3. **File Processing**: Multi-format file analysis and content extraction
4. **FAISS Retrieval**: Semantic search through Swaminarayan religious texts
5. **LangGraph Workflow**: Orchestrates the entire conversation flow

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API Key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gyan-guru-ai.git
cd gyan-guru-ai

Install dependencies

bashpip install -r requirements.txt

Configure API Keys

Update config.py with your OpenAI API key
Or set environment variable: export OPENAI_API_KEY="your-api-key"


Run the server

bashpython main.py

Access the interface

Open index.html in your browser
Or visit http://127.0.0.1:8000 for API access



📖 Usage Guide
Web Interface

Enter your session ID (for conversation persistence)
Type your query or upload a file
Get intelligent responses based on query type:

Educational: Step-by-step tutoring
Religious: Swaminarayan teachings and guidance
General: Regular AI assistance



API Endpoints
POST /
Submit queries with optional file uploads
Parameters:

query (string): Your question or message
session_id (string): Unique identifier for conversation persistence
file (optional): Image, PDF, or text file for analysis

Example Request:
bashcurl -X POST "http://127.0.0.1:8000/" \
  -F "query=Explain the concept of Akshar-Purushottam" \
  -F "session_id=user123" \
  -F "file=@document.pdf"
🎓 Educational Features
Tutoring Mode

Adaptive Learning: Adjusts explanations based on student responses
Step-by-Step Guidance: Breaks down complex problems
Progress Tracking: Monitors understanding throughout the session
Hint System: Provides graduated assistance without giving away answers
Multi-Subject Support: Mathematics, science, and general problem-solving

Example Tutoring Flow
Student: "I need help with quadratic equations"
AI: "Let's work through this together! What do you know about quadratic equations?"
Student: "They have x squared terms"
AI: "Exactly! Now, can you tell me the standard form?"
🕉️ Religious Knowledge Base
Comprehensive Coverage

Scriptures: Vachanamrut, Shikshapatri, Satsangi Jivan
Philosophy: Akshar-Purushottam Darshan, Five Eternal Entities
History: Bhagwan Swaminarayan's life and teachings
Festivals: Detailed information on celebrations and rituals
Saints: Biographies and teachings of great spiritual leaders

Example Religious Queries

"What is the significance of Ekadashi fasting?"
"Explain the concept of Aksharbrahman"
"Tell me about Gunatitanand Swami's teachings"
"What are the five vartmans in Swaminarayan tradition?"

🔧 Configuration
Key Settings (config.py)
pythonOPENAI_API_KEY = "your-api-key-here"
MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
RETRIEVER_K = 3
SIMILARITY_THRESHOLD = 0.6
File Limits

Maximum file size: 5MB
Supported formats: Images (JPEG, PNG), PDFs, Text files

📁 Project Structure
gyan-guru-ai/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration and data models
├── chat_memory.py         # LangGraph workflow and conversation management
├── query_classifier.py    # Query classification logic
├── file_processing.py     # File upload and analysis
├── faiss_retrieval.py     # Vector search implementation
├── llm_utils.py          # LLM initialization and utilities
├── index.html            # Web interface
├── text.txt              # Swaminarayan knowledge base
├── requirements.txt      # Python dependencies
└── README.md            # This file
🔄 Workflow Details
LangGraph State Management
The application uses LangGraph to manage complex conversation flows:

Query Classification: Determines the appropriate response mode
Context Retrieval: Searches relevant information from knowledge base
Response Generation: Creates appropriate responses based on mode
State Persistence: Maintains conversation context across interactions

Tutoring Workflow
mermaidgraph TD
    A[Student Query] --> B[Classify as Tutoring]
    B --> C[Generate Step/Hint]
    C --> D[Wait for Student Response]
    D --> E[Check Understanding]
    E --> F{Correct?}
    F -->|Yes| G[Next Step]
    F -->|No| H[Provide Hint]
    G --> I[Continue or End]
    H --> D
🛠️ Advanced Features
File Processing Pipeline

Upload Validation: Size and format checking
Content Extraction:

Images: OCR and visual analysis
PDFs: Text extraction with PyPDF2
Text: Direct processing


Context Integration: Incorporates file content into responses
History Management: Maintains file context across sessions

Vector Search

FAISS Integration: Fast similarity search through religious texts
Semantic Understanding: Context-aware information retrieval
Threshold Filtering: Only returns highly relevant results
Chunked Processing: Efficient handling of large documents

🎨 User Interface
Features

Real-time Chat: Instant responses with typing indicators
File Upload: Drag-and-drop file processing
Math Rendering: KaTeX support for mathematical expressions
Markdown Support: Rich text formatting in responses
Session Management: Persistent conversations across page reloads

Mathematical Expression Support
The interface supports LaTeX mathematical notation:

Inline math: \(equation\)
Display math: $$equation$$

🔍 Troubleshooting
Common Issues
API Key Issues
Error: OpenAI API key not found
Solution: Check config.py or set OPENAI_API_KEY environment variable
File Upload Failures
Error: File size exceeds limit
Solution: Ensure files are under 5MB
Vector Search Not Working
Error: FAISS index not found
Solution: Ensure text.txt exists and is readable
Debug Mode
Enable detailed logging by setting:
pythonimport logging
logging.getLogger().setLevel(logging.DEBUG)
🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the repository
Create a feature branch: git checkout -b feature-name
Make your changes with proper documentation
Add tests if applicable
Submit a pull request with detailed description

Development Guidelines

Follow PEP 8 style guidelines
Add docstrings to all functions
Include type hints where appropriate
Update README for new features

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Swaminarayan Sampradaya for the spiritual wisdom and teachings
OpenAI for providing powerful language models
LangChain/LangGraph for conversation workflow management
FAISS for efficient vector search capabilities

📞 Support
For support and questions:

Create an issue on GitHub
Contact the development team
Check the documentation wiki


Made with ❤️ for spiritual learning and educational excellence
Jai Swaminarayan 🙏
RetryClaude does not have the ability to run the code it generates yet.Claude can make mistakes. Please double-check responses.
