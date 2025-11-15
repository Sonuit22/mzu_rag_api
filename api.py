from flask import Flask, request, jsonify
from flask_cors import CORS
from query import answer_query
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)


# --- GLOBAL CORS FIX FOR CLOUDFLARE ---
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# --- HEALTH CHECK ---
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# --- CHAT ROUTE ---
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():

    # Handle preflight before reading JSON
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(silent=True)

        if not data or "query" not in data:
            return jsonify({"answer": "⚠ Invalid request. Send JSON {query: 'your question'}"}), 200
        
        query = data["query"]
        app.logger.info(f"QUERY → {query}")

        answer = answer_query(query)
        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"answer": f"⚠ Server error: {e}"}), 200


# --- DISABLED DB ROUTE ---
@app.route("/builddb", methods=["POST"])
def builddb():
    return jsonify({
        "status": "disabled",
        "message": "Embedding building is disabled on Render"
    }), 200


# --- RUN LOCAL ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
