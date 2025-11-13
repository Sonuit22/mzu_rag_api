# api.py

from flask import Flask, request, jsonify, abort
import os
import traceback

# Optional CORS (safe for web frontend)
try:
    from flask_cors import CORS
    cors_available = True
except:
    cors_available = False

# Lazy import to avoid circular issues
def _import_answer():
    from query import answer_query
    return answer_query

app = Flask(__name__)

if cors_available:
    CORS(app)

API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

def require_auth(request):
    if not API_AUTH_TOKEN:
        return True
    auth = request.headers.get("Authorization")
    if not auth:
        return False
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1]
    else:
        token = auth
    return token == API_AUTH_TOKEN


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/info', methods=['GET'])
def info():
    try:
        if not require_auth(request):
            return jsonify({'error': 'unauthorized'}), 401

        import chromadb
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        collections = chroma_client.list_collections()
        names = [c.name for c in collections]

        stats = {"collections": names}
        if "mzu_knowledge" in names:
            coll = chroma_client.get_collection("mzu_knowledge")
            try:
                count = coll.count()
            except:
                count = None
            stats["mzu_knowledge_count"] = count

        return jsonify({"status": "ok", "chroma": stats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "details": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not require_auth(request):
            return jsonify({'error': 'unauthorized'}), 401

        data = request.get_json(force=True)
        if "query" not in data:
            return jsonify({"error": "No query provided"}), 400

        q = data["query"]
        answer_query = _import_answer()
        ans = answer_query(q)

        return jsonify({"answer": ans})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "internal_error", "details": str(e)}), 500


# Do NOT run Flask dev server on Render
# Gunicorn will run the app automatically
if __name__ == "__main__":
    print("Running local dev server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
