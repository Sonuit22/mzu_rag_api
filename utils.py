import os
import json
from pathlib import Path


# Ensure required folders always exist
def ensure_dirs():
    Path('mzu_docs').mkdir(parents=True, exist_ok=True)
    Path('data').mkdir(parents=True, exist_ok=True)


# Run once when imported
ensure_dirs()


# Read a file safely
def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# Write file safely
def write_file(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


# Load JSON embeddings safely (needed for RAG search)
def load_embeddings(path='data/embeddings.json'):
    if not os.path.exists(path):
        return {"ids": [], "docs": [], "vectors": []}

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
