import os
from flask import Flask, jsonify, request, send_from_directory

from backend import rag_chain, retriever

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="Please enter a question."), 400
    try:
        answer = rag_chain.invoke(question)
        sources = [
            {
                "source": os.path.basename(d.metadata.get("source", "unknown")),
                "page": d.metadata.get("page"),
                "text": d.page_content,
            }
            for d in retriever.invoke(question)
        ]
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(answer=answer, sources=sources)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
