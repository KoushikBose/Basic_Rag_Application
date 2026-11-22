# Basic RAG Application

A small Retrieval-Augmented Generation (RAG) app. It answers questions from PDFs in the `knowledge base/` folder. It has a Flask backend, a LangChain pipeline, a FAISS vector store and a plain HTML/JS chat UI.

## Architecture

```mermaid
flowchart LR
    subgraph Client["Frontend (browser)"]
        UI["index.html / script.js / style.css<br/>chat UI + collapsible sources"]
    end

    subgraph Server["Backend (Flask - server.py)"]
        API["POST /api/ask"]
        STATIC["GET / and /&lt;file&gt;<br/>serves frontend"]
    end

    subgraph RAG["RAG pipeline (backend.py - LangChain LCEL)"]
        RET["FAISS retriever<br/>similarity, k=3"]
        PROMPT["Prompt template<br/>context + question"]
        LLM["ChatGroq<br/>openai/gpt-oss-120b"]
        PARSE["StrOutputParser"]
    end

    subgraph Index["Indexing (runs once at startup)"]
        PDFS["knowledge base/*.pdf"] --> LOAD["PyPDFLoader"]
        LOAD --> SPLIT["RecursiveCharacterTextSplitter<br/>300 chars, 50 overlap"]
        SPLIT --> EMB["HuggingFaceEmbeddings<br/>all-MiniLM-L6-v2"]
        EMB --> VS[("FAISS vector store<br/>in memory")]
    end

    UI -- "question (JSON)" --> API
    API --> RET
    VS --> RET
    RET -- "top-3 chunks" --> PROMPT --> LLM --> PARSE
    PARSE -- "answer" --> API
    RET -- "source chunks" --> API
    API -- "answer + sources (JSON)" --> UI
    STATIC --> UI
```

### Request flow

1. **Indexing (startup).** Importing `backend.py` loads every PDF in `knowledge base/` and splits it into 300-character chunks with 50 characters of overlap. Each chunk is embedded with `all-MiniLM-L6-v2` and stored in an in-memory FAISS index. The index is rebuilt on every start and is not saved to disk.
2. **Question.** The browser sends `POST /api/ask` with `{ "question": "..." }`.
3. **Retrieval.** The retriever returns the 3 chunks most similar to the question.
4. **Generation.** The chunks are inserted into a prompt that tells the LLM to answer only from the context. If the answer isn't in the context, the LLM replies: *"I don't know, it's not in my knowledge base."* The prompt goes to Groq's `openai/gpt-oss-120b`, with `max_tokens=500`.
5. **Response.** The server returns `{ answer, sources: [{source, page, text}] }`. The UI shows the answer with a collapsible list of the source chunks and page numbers.

## Project structure

```
Basic Rag Project/
├── backend/
│   ├── backend.py        # RAG pipeline: load -> chunk -> embed -> FAISS -> prompt -> LLM
│   └── server.py         # Flask app: /api/ask endpoint + static frontend hosting
├── frontend/
│   ├── index.html        # Chat page
│   ├── script.js         # Calls /api/ask, renders answer and sources
│   └── style.css
├── knowledge base/       # PDFs that are indexed (add your own here)
├── requirements.txt
├── .env                  # API keys (not committed)
└── README.md
```

## Tech stack

| Layer           | Technology                                           |
|-----------------|------------------------------------------------------|
| Frontend        | HTML, CSS, vanilla JavaScript                        |
| API server      | Flask                                                |
| Orchestration   | LangChain (LCEL)                                     |
| Document loader | `PyPDFLoader` (pypdf)                                |
| Chunking        | `RecursiveCharacterTextSplitter`                     |
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2`             |
| Vector store    | FAISS (CPU)                                          |
| LLM             | Groq, `openai/gpt-oss-120b`                          |

## Getting started

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/)
- A Hugging Face token (optional; it avoids rate limits when downloading the embedding model)

### Setup

```bash
git clone https://github.com/KoushikBose/Basic_Rag_Application.git
cd Basic_Rag_Application

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
```

### Run

```bash
cd backend
python server.py
```

Open <http://127.0.0.1:5000>. The first start takes a while because it downloads the embedding model and builds the index.

To test the pipeline from the command line without the UI, run `python backend.py`. It prints the vector DB as a table and answers a sample question.

## API

`POST /api/ask`

Request:
```json
{ "question": "What is RAG?" }
```

Response:
```json
{
  "answer": "...",
  "sources": [{ "source": "file.pdf", "page": 0, "text": "..." }]
}
```

An empty question returns `400`. A pipeline error returns `500`. Both have the form `{ "error": "..." }`.

## Customizing

- **Add knowledge:** drop more PDFs into `knowledge base/` and restart the server.
- **Retrieval depth:** change `k` in `search_kwargs` in `backend.py`.
- **Chunking:** change `chunk_size` and `chunk_overlap` in `backend.py`.
- **Model:** change the model name passed to `ChatGroq`.
