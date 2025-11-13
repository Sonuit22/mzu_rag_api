# api.py
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
from flask import Flask, request, jsonify
from query import answer_query

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    q = data.get('query')
    if not q:
        return jsonify({'error': 'No query provided'}), 400
    ans = answer_query(q)
    return jsonify({'answer': ans})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
