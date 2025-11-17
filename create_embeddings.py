# create_embeddings.py
# Generate embeddings locally (NO OPENAI REQUIRED)

import os
import json
from sentence_transformers import SentenceTransformer
from utils import read_file
from chunk import chunk_text

MODEL_NAME = "paraphrase-MiniLM-L6-v2"  # lightweight, no GPU required
embedder = SentenceTransformer(MODEL_NAME)

DATA_PATH = "data/mzu_raw.txt"
OUT_PATH = "data/embeddings.json"

# 1. Load text
text = read_file(DATA_PATH)

# 2. Chunk text
chunks = chunk_text(text, chunk_size=900, overlap=150)
print(f"Total chunks created: {len(chunks)}")

ids, docs, vectors = [], [], []

# 3. Encode locally
for i, chunk in enumerate(chunks):
    vec = embedder.encode(chunk).tolist()
    ids.append(f"chunk_{i}")
    docs.append(chunk)
    vectors.append(vec)

# 4. Save embeddings.json
payload = {
    "ids": ids,
    "docs": docs,
    "vectors": vectors
}

os.makedirs("data", exist_ok=True)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print(f"Generated {len(ids)} chunks → {OUT_PATH}")
