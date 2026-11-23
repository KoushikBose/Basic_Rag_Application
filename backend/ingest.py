"""⚡ Incremental indexing for the knowledge base.

Only new or changed PDFs are parsed and embedded. Unchanged PDFs are skipped, and
the chunks of deleted PDFs are removed. The FAISS index is saved to disk, and a
SQLite manifest records which file produced which chunks.

Run on its own (e.g. from cron / a scheduler) or let backend.py call it on startup:
    python ingest.py            # incremental sync
    python ingest.py --rebuild  # throw everything away and re-embed from scratch

Layout on disk (STORE_DIR):
    faiss/            index.faiss + index.pkl  (LangChain FAISS.save_local)
    manifest.sqlite3  files(file_key, size, mtime_ns, sha256, ...) + chunks(chunk_id, file_key)
    .ingest.lock      present while an ingestion is running
"""
import fnmatch
import hashlib
import logging
import os
import shutil
import sqlite3
import time
from contextlib import closing, contextmanager

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    CHECKPOINT_EVERY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    FAISS_DIR,
    KNOWLEDGE_BASE_DIR,
    LOCK_FILE,
    MANIFEST_DB,
    STORE_DIR,
)

log = logging.getLogger("ingest")

# 🔑 Any setting that changes how chunks or vectors are produced. If it differs from the
# value stored in the manifest, the old vectors are incompatible, so we rebuild.
INDEX_FINGERPRINT = (
    f"schema=1;model={EMBEDDING_MODEL_NAME};chunk_size={CHUNK_SIZE};chunk_overlap={CHUNK_OVERLAP}"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta  (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    file_key   TEXT PRIMARY KEY,   -- file name inside the knowledge base folder
    size       INTEGER NOT NULL,
    mtime_ns   INTEGER NOT NULL,
    sha256     TEXT    NOT NULL,
    num_chunks INTEGER NOT NULL,
    indexed_at REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,     -- FAISS docstore id
    file_key TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_key);
"""


# ───────────────────────────── helpers ─────────────────────────────

def _connect():
    db = sqlite3.connect(MANIFEST_DB)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(_SCHEMA)
    return db


@contextmanager
def _ingest_lock():
    """Stop two indexers from writing the same index at the same time."""
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"Another ingestion is already running (lock file: {LOCK_FILE}). "
            "If a previous run crashed, delete the lock file and try again."
        ) from None
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    try:
        yield
    finally:
        os.remove(LOCK_FILE)


def _file_sha256(path, block_size=1 << 20):
    """Hash the file in 1 MB blocks so large PDFs never sit in memory whole."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def _chunk_id(file_key, file_sha, i):
    """Deterministic id: same file content -> same ids, so re-running a batch is idempotent."""
    return hashlib.sha1(f"{file_key}\x00{file_sha}\x00{i}".encode()).hexdigest()


def _scan_pdfs(kb_dir):
    """Return the files glob('*.pdf') would match, with their stat info, in one directory pass."""
    found = {}
    with os.scandir(kb_dir) as entries:
        for e in entries:
            if not e.name.startswith(".") and fnmatch.fnmatch(e.name, "*.pdf") and e.is_file():
                found[e.name] = (os.path.join(kb_dir, e.name), e.stat())
    return found


def _recover_interrupted_swap():
    """If we crashed between the two renames in _save_index, put the last good index back."""
    old = FAISS_DIR + ".old"
    if not os.path.isdir(FAISS_DIR) and os.path.isdir(old):
        os.replace(old, FAISS_DIR)


def _save_index(store):
    """Atomic save: write to a temp dir, then swap it in. Readers never see a half-written index."""
    tmp, old = FAISS_DIR + ".tmp", FAISS_DIR + ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    store.save_local(tmp)
    if os.path.isdir(FAISS_DIR):
        os.replace(FAISS_DIR, old)
    os.replace(tmp, FAISS_DIR)
    shutil.rmtree(old, ignore_errors=True)


def _load_or_reset(db, embedding_model, force_rebuild):
    """Load the persisted index if it was built with the current settings. Otherwise start empty."""
    row = db.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
    if not force_rebuild and row and row[0] == INDEX_FINGERPRINT and os.path.isdir(FAISS_DIR):
        log.info("Loading persisted FAISS index from %s", FAISS_DIR)
        # The pickle was written by this program, so deserialising it is safe.
        return FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)

    if force_rebuild:
        reason = "--rebuild requested"
    elif row is None:
        reason = "no index yet"
    elif row[0] != INDEX_FINGERPRINT:
        reason = f"settings changed ({row[0]} -> {INDEX_FINGERPRINT})"
    else:
        reason = "index files missing"
    log.warning("Full build: %s", reason)
    db.execute("DELETE FROM chunks")
    db.execute("DELETE FROM files")
    db.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('fingerprint', ?)", (INDEX_FINGERPRINT,)
    )
    return None


# ───────────────────────────── indexer ─────────────────────────────

class _Indexer:
    """Buffers chunks across files so the embedding model always gets full batches,
    and checkpoints (save index, then commit manifest) every CHECKPOINT_EVERY chunks."""

    def __init__(self, db, store, embedding_model):
        self.db = db
        self.store = store
        self.embedding_model = embedding_model
        self.buf_docs, self.buf_ids = [], []
        self.since_checkpoint = 0
        self.dirty = False  # in-memory index differs from the one on disk

    def has(self, chunk_id):
        return self.store is not None and isinstance(self.store.docstore.search(chunk_id), Document)

    def reconcile(self):
        """Fix drift left by a crash between 'index saved' and 'manifest committed'.
        Cheap when nothing is wrong: it only compares two counts."""
        if self.store is None:
            return
        n_tracked = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if len(self.store.index_to_docstore_id) == n_tracked:
            return
        log.warning("Index and manifest disagree; reconciling")
        indexed = set(self.store.index_to_docstore_id.values())
        tracked = {}
        for chunk_id, file_key in self.db.execute("SELECT chunk_id, file_key FROM chunks"):
            tracked[chunk_id] = file_key
        # Vectors nobody owns -> delete them.
        self.delete([cid for cid in indexed if cid not in tracked])
        # Files whose vectors are missing -> forget them so they get re-indexed.
        for file_key in {fk for cid, fk in tracked.items() if cid not in indexed}:
            self.db.execute("DELETE FROM chunks WHERE file_key = ?", (file_key,))
            self.db.execute("DELETE FROM files  WHERE file_key = ?", (file_key,))

    def delete(self, chunk_ids):
        """One FAISS delete call per batch of ids (each call is O(index size))."""
        ids = [cid for cid in chunk_ids if self.has(cid)]
        if ids:
            self.store.delete(ids)
            self.dirty = True

    def add(self, docs, ids):
        for doc, cid in zip(docs, ids):
            if self.has(cid):  # already embedded by an earlier, interrupted run
                continue
            self.buf_docs.append(doc)
            self.buf_ids.append(cid)
            if len(self.buf_docs) >= EMBED_BATCH_SIZE:
                self.flush()
        self.since_checkpoint += len(docs)
        if self.since_checkpoint >= CHECKPOINT_EVERY:
            self.checkpoint()

    def flush(self):
        """Embed the buffered chunks and add them to FAISS."""
        if not self.buf_docs:
            return
        if self.store is None:
            self.store = FAISS.from_documents(self.buf_docs, self.embedding_model, ids=self.buf_ids)
        else:
            self.store.add_documents(self.buf_docs, ids=self.buf_ids)
        self.dirty = True
        self.buf_docs, self.buf_ids = [], []

    def checkpoint(self):
        """Save the index first, then commit the manifest. If we crash in between, the
        next run sees extra vectors and reconcile() fixes it. Never the other way round."""
        self.flush()
        if self.dirty:
            if self.store is not None:
                _save_index(self.store)
            self.dirty = False
        elif self.store is None:
            shutil.rmtree(FAISS_DIR, ignore_errors=True)  # full rebuild with nothing to index
        self.db.commit()
        self.since_checkpoint = 0


# ───────────────────────────── entry point ─────────────────────────────

def sync_vector_store(kb_dir, splitter, embedding_model, force_rebuild=False):
    """Bring the persisted FAISS index in line with the PDFs in kb_dir and return it.

    The steps:
      1. Load the saved index (or start empty if settings changed / first run).
      2. Scan the folder. size+mtime unchanged -> skip without reading the file.
         Otherwise hash it: same hash -> skip; new hash -> (re)index.
      3. Delete the vectors of removed and modified files in one FAISS pass.
      4. Parse, split and embed ONLY the new/modified files, in batches.
      5. Checkpoint: save the index atomically, then commit the manifest.
    """
    os.makedirs(STORE_DIR, exist_ok=True)
    started = time.time()

    with _ingest_lock(), closing(_connect()) as db:
        # 1️⃣ Load
        _recover_interrupted_swap()
        idx = _Indexer(db, _load_or_reset(db, embedding_model, force_rebuild), embedding_model)
        idx.reconcile()

        # 2️⃣ Detect changes
        on_disk = _scan_pdfs(kb_dir)
        known = {
            key: (size, mtime_ns, sha)
            for key, size, mtime_ns, sha in db.execute(
                "SELECT file_key, size, mtime_ns, sha256 FROM files"
            )
        }
        removed = [key for key in known if key not in on_disk]
        todo = []  # (file_key, path, stat, sha256) of new or modified files
        for key, (path, st) in sorted(on_disk.items()):
            prev = known.get(key)
            if prev and prev[0] == st.st_size and prev[1] == st.st_mtime_ns:
                continue  # fast path: untouched, not even read
            sha = _file_sha256(path)
            if prev and prev[2] == sha:  # touched/copied but same bytes
                db.execute(
                    "UPDATE files SET size = ?, mtime_ns = ? WHERE file_key = ?",
                    (st.st_size, st.st_mtime_ns, key),
                )
                continue
            todo.append((key, path, st, sha))
        modified = [t[0] for t in todo if t[0] in known]
        log.info(
            "Knowledge base: %d PDFs on disk | %d new | %d modified | %d removed | %d unchanged",
            len(on_disk), len(todo) - len(modified), len(modified), len(removed),
            len(on_disk) - len(todo),
        )

        # 3️⃣ Drop stale vectors (removed + modified files)
        stale_ids = []
        for key in removed + modified:
            stale_ids += [r[0] for r in db.execute("SELECT chunk_id FROM chunks WHERE file_key = ?", (key,))]
            db.execute("DELETE FROM chunks WHERE file_key = ?", (key,))
            db.execute("DELETE FROM files  WHERE file_key = ?", (key,))
        idx.delete(stale_ids)

        # 4️⃣ Embed only what is new
        added_chunks = failed = 0
        for n, (key, path, st, sha) in enumerate(todo, 1):
            try:
                # Same loader + splitter as before -> identical chunks and metadata.
                chunks = splitter.split_documents(PyPDFLoader(path).load())
            except Exception:
                failed += 1
                log.exception("Skipping %s: could not be parsed (will retry next run)", key)
                continue
            ids = [_chunk_id(key, sha, i) for i in range(len(chunks))]
            db.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id, file_key) VALUES (?, ?)",
                [(cid, key) for cid in ids],
            )
            db.execute(
                "INSERT OR REPLACE INTO files(file_key, size, mtime_ns, sha256, num_chunks, indexed_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (key, st.st_size, st.st_mtime_ns, sha, len(chunks), time.time()),
            )
            idx.add(chunks, ids)
            added_chunks += len(chunks)
            if n % 100 == 0 or n == len(todo):
                log.info("Indexed %d/%d files (%d chunks so far)", n, len(todo), added_chunks)

        # 5️⃣ Persist
        idx.checkpoint()
        store = idx.store

    if store is None:
        raise RuntimeError(f"No PDFs could be indexed from {kb_dir!r}; the vector store is empty.")
    log.info(
        "Vector store ready: %d chunks total, %d new chunks embedded, %d files failed, %.1fs",
        store.index.ntotal, added_chunks, failed, time.time() - started,
    )
    return store


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parser = argparse.ArgumentParser(description="Incrementally index the knowledge base.")
    parser.add_argument("--rebuild", action="store_true", help="re-embed everything from scratch")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    sync_vector_store(
        KNOWLEDGE_BASE_DIR,
        RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        force_rebuild=args.rebuild,
    )
