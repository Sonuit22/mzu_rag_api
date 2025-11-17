# create_embeddings.py — HASH EMBEDDINGS (Render Safe)

import os
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

text = read_file(DATA_PATH)
chunks = chunk_text(text, chunk_size=900, overlap=150)

ids, docs, vectors = [], [], []

for i, chunk in enumerate(chunks):
    ids.append(f"chunk_{i}")
    docs.append(chunk)
    vectors.append(embed_hash(chunk))

payload = {"ids": ids, "docs": docs, "vectors": vectors}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

print(f"Generated {len(ids)} chunks → {OUT_PATH}")
