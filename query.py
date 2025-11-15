# query.py  (FULLY UPDATED – NO OPENAI, FREE, SCRAPING + RAG)

import os
import requests
import chromadb
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# Load ENV variables
# -----------------------------
LLM_API_URL = os.getenv("LLM_API_URL")  # e.g., https://api.groq.com/openai/v1/chat/completions
LLM_API_KEY = os.getenv("LLM_API_KEY")  # your Groq / HF key
LLM_MODEL   = os.getenv("LLM_MODEL", "llama3-8b")

if not LLM_API_URL:
    raise ValueError("❌ LLM_API_URL not set in environment")
if not LLM_API_KEY:
    raise ValueError("❌ LLM_API_KEY not set in environment")

# -----------------------------
# Embedding Model (offline)
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Chroma Vector DB
# -----------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = chroma_client.get_collection("mzu_knowledge")
except:
    collection = chroma_client.create_collection("mzu_knowledge")

# -----------------------------
# LIVE Web Scraper (safe)
# -----------------------------
def scrape_mzu():
    url = "https://mzu.edu.in"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator=" ")
        text = " ".join(text.split())   # remove extra spaces
        return text[:5000]              # limit chars to avoid overload
    except:
        return "Live data unavailable."

# -----------------------------
# Query Function
# -----------------------------
def answer_query(query, k=3):

    # 1) Embed user query
    q_vec = embedder.encode(query).tolist()

    # 2) Search in RAG DB
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=k
    )

    docs = results.get("documents", [[]])[0]
    sources = results.get("metadatas", [[]])[0]

    rag_context = "\n\n".join(
        f"[Source: {src.get('source','mzu')}]\n{doc}"
        for doc, src in zip(docs, sources)
    )

    # 3) Live scrape
    live_data = scrape_mzu()

    # 4) Build prompt
    final_prompt = f"""
You are the Official MZU Assistant.

Use the OFFLINE knowledge base + LIVE scraped website data to answer.

OFFLINE RAG DATA:
{rag_context}

LIVE SCRAPED DATA:
{live_data}

QUESTION:
{query}

Give a clear, short, factual answer. 
If unsure, say "Information not available".
"""

    # 5) Call free LLM provider (Groq / HF / local server)
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are the MZU Assistant."},
            {"role": "user", "content": final_prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }

    response = requests.post(LLM_API_URL, json=body, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()

    # Extract answer from any OpenAI-style API
    try:
        answer = data["choices"][0]["message"]["content"]
    except:
        answer = str(data)

    return answer


# -------------- TEST MODE ----------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk MZU Bot: ")
        print("\nAnswer:", answer_query(q), "\n")
