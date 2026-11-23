"""⚙️ Single source of truth for paths and indexing settings.

backend.py (query time) and ingest.py (indexing) both read these values, so the
chunking and embedding settings can never drift apart between the two.
"""
import os

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# 📂 Where the PDFs live (same folder as before)
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "knowledge base")

# ✂️ Chunking + 🧠 embedding settings (unchanged values)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 💾 Persisted vector store. Override with VECTOR_STORE_DIR to put it on a fast local
# disk (e.g. outside OneDrive, which may lock files while syncing).
STORE_DIR = os.getenv("VECTOR_STORE_DIR", os.path.join(PROJECT_ROOT, "vector_store"))
FAISS_DIR = os.path.join(STORE_DIR, "faiss")              # index.faiss + index.pkl
MANIFEST_DB = os.path.join(STORE_DIR, "manifest.sqlite3")  # what has been indexed
LOCK_FILE = os.path.join(STORE_DIR, ".ingest.lock")        # one indexer at a time

# 🚀 Throughput / durability knobs
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "512"))     # chunks per embedding call
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "50000"))   # chunks between index saves
