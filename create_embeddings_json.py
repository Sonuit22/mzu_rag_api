# create_embeddings_json.py
# Run this LOCALLY to generate embeddings.json

import json
from sentence_transformers import SentenceTransformer
from utils import read_file
from chunk import chunk_text

MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL)

DATA_PATH = "data/mzu_raw.txt"
OUT_PATH = "data/embeddings.json"

text = read_file(DATA_PATH)
chunks = chunk_text(text, chunk_size=900, overlap=150)

ids = []
docs = []
vectors = []

for i, c in enumerate(chunks):
    emb = embedder.encode(c).tolist()
    ids.append(f"chunk_{i}")
    docs.append(c)
    vectors.append(emb)

payload = {
    "ids": ids,
    "docs": docs,
    "vectors": vectors
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False)

print(f"Generated {len(ids)} chunks → {OUT_PATH}")
