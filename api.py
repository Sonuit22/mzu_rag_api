# api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from query import answer_query

app = Flask(__name__)
CORS(app)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "no query"}), 400

    query = data["query"]
    k = int(data.get("k", 3))
    answer = answer_query(query, k)
    return jsonify({"answer": answer})

# Disable builddb (not needed for lightweight RAG)
@app.route("/builddb", methods=["POST"])
def builddb():
    return jsonify({
        "status": "disabled",
        "message": "Embedding building disabled on Render. Generate embeddings locally using create_embeddings_json.py"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
