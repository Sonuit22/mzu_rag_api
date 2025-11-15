from flask import Flask, request, jsonify
from flask_cors import CORS
from query import answer_query
import traceback
import logging

app = Flask(__name__)

# Allow GitHub Pages frontend (recommended)
CORS(app, resources={r"/*": {"origins": "*"}})

# Render-friendly logging
logging.basicConfig(level=logging.INFO)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Safely parse JSON
        data = request.get_json(silent=True)

        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data["query"]
        k = int(data.get("k", 3))

        app.logger.info(f"[QUERY] {query}")

        answer = answer_query(query, k)

        return jsonify({"answer": answer}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "internal_error",
            "details": str(e)
        }), 500


@app.route("/builddb", methods=["POST"])
def builddb():
    return jsonify({
        "status": "disabled",
        "message": "Embedding building disabled on Render. "
                   "Generate embeddings locally using create_embeddings_json.py"
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
