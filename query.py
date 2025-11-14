# query.py
import os
import chromadb
from openai import OpenAI
from utils import read_file

# Load API key
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Chroma persistent DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("mzu_knowledge")

SYSTEM_PROMPT = (
    "You are an assistant for Mizoram University (MZU). Answer the user's question using ONLY the provided context. "
    "If the answer is not in the context, say you don't know and suggest a way to find it. Keep answers concise and factual."
)


def answer_query(query, k=3):

    # Embedding
    qresp = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    qemb = qresp.data[0].embedding

    # Vector search
    results = collection.query(
        query_embeddings=[qemb],
        n_results=k
    )

    docs = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]

    context = "\n\n".join([
        f"[Source: {m.get('source','mzu')} chunk:{m.get('chunk')}]\n{d}"
        for d, m in zip(docs, metadatas)
    ])

    # Chat completion using new API
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    # New OpenAI Responses API
    resp = client.responses.create(
        model="gpt-4o",
        input=messages,
        max_output_tokens=400
    )

    return resp.output_text


if __name__ == "__main__":
    q = input("Question: ")
    print(answer_query(q))
