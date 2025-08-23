
# AI Chatbot for Financial Policy Documents

This repository provides a professional, production-ready AI chatbot for answering questions about financial policy documents. It leverages state-of-the-art Retrieval-Augmented Generation (RAG) techniques, advanced language models, and semantic search to deliver accurate, referenced, and context-aware responses.

---

## Features

- **Agentic Document Extraction:** Robust PDF parsing and semantic chunking using the `agentic-doc` Python library.
- **Vector Search:** Efficient retrieval of relevant document sections using ChromaDB and HuggingFace embeddings.
- **RAG Pipeline:** Combines retrieval and LLM-based answer generation for robust, explainable Q&A.
- **Streamlit UI:** Modern, interactive chat interface for seamless user experience.
- **Source Attribution:** Every answer includes supporting context, source page, and confidence level.

---

## Technical Implementation Details

### 1. Document Extraction & Preprocessing
- **PDF Parsing:**
	- The `agentic-doc` library parses the policy PDF, extracting text and tables with metadata.
	- Custom logic groups content by page and type, producing LangChain `Document` objects for each page.
- **Semantic Chunking:**
	- Documents are split into semantically meaningful chunks using `SemanticChunker` from `langchain_experimental`.
	- Embeddings are generated with `HuggingFaceEmbeddings` (model: `BAAI/bge-large-en-v1.5`).

### 2. Vector Store
- **ChromaDB:**
	- All document chunks are stored in a persistent ChromaDB vector store for fast similarity search.
	- The vector store is created at startup and reused for all queries.

### 3. Retrieval-Augmented Generation (RAG)
- **Retriever:**
	- For each user query, the top-k most relevant chunks are retrieved using vector similarity.
	- A cross-encoder re-ranker (`BAAI/bge-reranker-base`) further refines the context.
- **LLM Integration:**
	- The system uses the Groq API with the `llama-3.3-70b-versatile` model for answer generation.
	- The RAG chain is history-aware, reformulating queries as needed and providing context-rich answers.
- **Prompt Engineering:**
	- System prompts enforce answer structure, require supporting context, and demand source attribution and confidence scoring.

### 4. User Interface
- **Streamlit:**
	- The app provides a chat interface, displaying both user and assistant messages.
	- Each answer includes direct response, supporting context, source, and confidence.

---

## Agentic Document Extraction – Python Library & Environment Details

- **Library:** [`agentic-doc`](https://pypi.org/project/agentic-doc/)
	- Used for robust PDF parsing, including text, tables, and metadata extraction.
	- Integrates seamlessly with LangChain document objects.
- **Environment Variables:**
	- `GROQ_API_KEY` – Required for LLM access (Groq API).
	- `VISION_AGENT_API_KEY` – (Optional) For vision-based document extraction, if needed.
	- Store these in a `.env` file at the project root.
- **Python Version:** 3.10+
- **Dependencies:** See `requirements.txt` for all required packages.

---

## Project Structure

```
├── app.py                # Main Streamlit app 
├── modules/
│   ├── ragchain.py       # RAG pipeline and LLM integration
│   ├── vector_store.py   # Vector store and embedding logic
│   └── extract_pdf.py    # PDF parsing and document creation
├── chroma-db/            # ChromaDB persistent storage
├── context_pdf/
|    └── Policy file.pdf       # Example policy document
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Setup & Installation

1. **Clone the repository:**
	 ```sh
	 git clone https://github.com/sheikhmdsamiul/AI-Chatbot-financial-policy-document.git
	 cd AI-Chatbot-financial-policy-document
	 ```
2. **Create and activate a virtual environment:**
	 ```sh
	 python -m venv env1
	 .\env1\Scripts\activate
	 ```
3. **Install dependencies:**
	 ```sh
	 pip install -r requirements.txt
	 ```
4. **Set up environment variables:**
	- Create a `.env` file and add your API keys.
	- **GROQ_API_KEY:** Required for LLM access. You can obtain a free API key by signing up at [https://console.groq.com/keys](https://console.groq.com/keys).
5. **Run the app:**
	 ```sh
	 streamlit run app.py
	 ```

---

## Usage

- Open the Streamlit app in your browser.
- Ask questions about the uploaded policy document.
- The chatbot will respond with answers, supporting context, and source references.

---

## Requirements

- Python 3.10+
- GROQ_API_KEY in .env file
- See `requirements.txt` for all dependencies.

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://www.trychroma.com/)
- [HuggingFace](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- [agentic-doc](https://pypi.org/project/agentic-doc/)



