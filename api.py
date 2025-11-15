# api.py – FINAL UPDATED VERSION
# Clean, stable, Render-ready backend

from flask import Flask, request, jsonify
import os
import traceback
import subprocess
from flask_cors import CORS

from query import answer_query   # NEW updated answer system

app = Flask(__name__)
CORS(app)


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
    """
    Returns collection name + document count.
    Works even if Chroma folder is empty.
    """
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

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
    """
    Main chat endpoint used by your frontend.
    {
      "query": "your question",
      "k": 3
    }
    """
    try:
        data = request.get_json(force=True)

        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400

        question = data["query"]
        k = int(data.get("k", 3))

        # Call full RAG + scraping + LLM chain
        answer = answer_query(question, k=k)

        return jsonify({"answer": answer})

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "internal_error",
            "details": str(e)
        }), 500


# ----------------------------
# Build Vector DB Endpoint
# ----------------------------
@app.route("/builddb", methods=["POST"])
def builddb():
    """
    Rebuilds vector DB using embed_store.py.
    You should call this ONCE after Render deploy.
    """
    try:
        proc = subprocess.run(
            ["python", "embed_store.py"],
            capture_output=True,
            text=True
        )

        return jsonify({
            "returncode": proc.returncode,
            "stdout": proc.stdout[-800:],
            "stderr": proc.stderr[-800:]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "build_failed", "details": str(e)}), 500


# ----------------------------
# Local Dev Server
# ----------------------------
if __name__ == "__main__":
    print("Running API locally at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
