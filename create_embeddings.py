# create_embeddings.py — HASH EMBEDDINGS (Render Safe)

import json
import numpy as np
from utils import read_file
from chunk import chunk_text

DATA_PATH = "data/mzu_raw.txt"
OUT_PATH = "data/embeddings.json"

def embed_hash(text):
    vec = np.zeros(300)
    for w in text.lower().split():
        vec[hash(w) % 300] += 1
    return vec.tolist()

# 1. Load text
text = read_file(DATA_PATH)

# 2. Chunk text
chunks = chunk_text(text)
print(f"Total chunks created: {len(chunks)}")


ids, docs, vectors = [], [], []

print(f"Total chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks):
    ids.append(f"chunk_{i}")
    docs.append(chunk)
    vectors.append(embed_hash(chunk))

payload = {
    "ids": ids,
    "docs": docs,
    "vectors": vectors
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print(f"Generated {len(ids)} chunks → {OUT_PATH}")
