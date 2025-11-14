# api.py

from flask import Flask, request, jsonify
import os
import traceback
import subprocess

# Enable CORS for GitHub Pages
from flask_cors import CORS
from query import answer_query   # Lazy import removed for simplicity

# Flask App
app = Flask(__name__)
CORS(app)  # Allow frontend to call the API


# ----------------------------
# Health Check
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ----------------------------
# Info Endpoint (Chroma status)
# ----------------------------
@app.route("/info", methods=["GET"])
def info():
    try:
        import chromadb
        from chromadb.config import Settings

        chroma_client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chromadb_store")
        )

        collections = chroma_client.list_collections()
        names = [c.name for c in collections]

        stats = {"collections": names}

        if "mzu_knowledge" in names:
            coll = chroma_client.get_collection("mzu_knowledge")
            try:
                stats["mzu_knowledge_count"] = coll.count()
            except:
                stats["mzu_knowledge_count"] = None

        return jsonify({"status": "ok", "chroma": stats})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "details": str(e)}), 500


# ----------------------------
# Chat Endpoint
# ----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)

        if "query" not in data:
            return jsonify({"error": "No query provided"}), 400

        question = data["query"]
        k = int(data.get("k", 3))

        # Call RAG pipeline
        answer = answer_query(question, k=k)

        return jsonify({"answer": answer})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "internal_error", "details": str(e)}), 500


# ----------------------------
# Build Vector DB on Render
# ----------------------------
@app.route("/builddb", methods=["POST"])
def builddb():
    """
    Runs embed_store.py on the server.
    This generates embeddings and stores them in chromadb_store.
    """
    try:
        proc = subprocess.run(
            ["python", "embed_store.py"],
            capture_output=True,
            text=True
        )

        return jsonify({
            "returncode": proc.returncode,
            "stdout": proc.stdout[-500:],
            "stderr": proc.stderr[-500:]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "build_failed", "details": str(e)}), 500


# ----------------------------
# Local Dev Server
# ----------------------------
if __name__ == "__main__":
    print("Running API locally...")
    app.run(host="0.0.0.0", port=5000, debug=True)
