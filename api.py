from flask import Flask, request, jsonify
from flask_cors import CORS
from query import answer_query
import traceback
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Better logging
logging.basicConfig(level=logging.INFO, format='[API] %(message)s')


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)

        if not data or "query" not in data:
            return jsonify({"answer": "⚠ No query provided"}), 200

        query = data["query"]
        k = int(data.get("k", 3))

        app.logger.info(f"Query received → {query}")

        answer = answer_query(query, k)

        return jsonify({"answer": answer}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "answer": f"⚠ Internal server error: {e}"
        }), 200


@app.route("/builddb", methods=["POST"])
def builddb():
    return jsonify({
        "status": "disabled",
        "message": "Embedding building is disabled on Render (local only)"
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
