# embed_store.py – FINAL OFFLINE VERSION
# Uses SentenceTransformers + Chroma (NO API, NO COST)

import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chunk import chunk_text
from utils import read_file


# -------------------------------------
# Embedding model (FREE, OFFLINE)
# -------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------------
# Chroma persistent DB
# -------------------------------------
CHROMA_DIR = "./chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

COLLECTION_NAME = "mzu_knowledge"

try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except:
    collection = chroma_client.create_collection(COLLECTION_NAME)


# -------------------------------------
# Load training source file(s)
# -------------------------------------
DATA_PATH = "data/mzu_raw.txt"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Training file missing: {DATA_PATH}")

full_text = read_file(DATA_PATH)

# Split text into chunks
chunks = chunk_text(full_text, chunk_size=900, overlap=150)

print(f"Total chunks: {len(chunks)}")

ids = []
docs = []
metadatas = []
embeddings = []


# -------------------------------------
# Embed and store chunks
# -------------------------------------
for idx, chunk in enumerate(tqdm(chunks)):
    vec = embedder.encode(chunk).tolist()

    ids.append(f"chunk_{idx}")
    docs.append(chunk)
    metadatas.append({"source": "mzu_docs", "chunk": idx})
    embeddings.append(vec)


# Clear existing collection and re-create
try:
    chroma_client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = chroma_client.create_collection(COLLECTION_NAME)

# Upload in single batch
collection.add(
    ids=ids,
    documents=docs,
    embeddings=embeddings,
    metadatas=metadatas
)

try:
    chroma_client.persist()
except:
    pass

print("Stored", len(ids), "chunks in Chroma (OFFLINE MODE)")
