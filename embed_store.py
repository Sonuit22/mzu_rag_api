# embed_store.py
import os
from openai import OpenAI
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")

from tqdm import tqdm
from chunk import chunk_text
from utils import read_file

API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError('Set OPENAI_API_KEY in environment')

client = OpenAI(api_key=API_KEY)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection_name = 'mzu_knowledge'
existing = [c.name for c in chroma_client.list_collections()]
if collection_name in existing:
    collection = chroma_client.get_collection(collection_name)
else:
    collection = chroma_client.create_collection(collection_name)

text = read_file('data/mzu_raw.txt')
chunks = chunk_text(text, chunk_size=1000, overlap=200)

ids = []
docs = []
metadatas = []
embeddings = []

batch_size = 16
for i in tqdm(range(0, len(chunks), batch_size)):
    batch = chunks[i:i+batch_size]
    resp = client.embeddings.create(model='text-embedding-3-small', input=batch)
    for j, item in enumerate(resp.data):
        idx = i + j
        ids.append(f'chunk_{idx}')
        docs.append(batch[j])
        metadatas.append({'source': 'mzu_docs', 'chunk': idx})
        embeddings.append(item.embedding)

collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)

try:
    chroma_client.persist()
except Exception:
    pass

print('Stored', len(ids), 'chunks in Chroma')
