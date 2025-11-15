# query.py
# Lightweight RAG for Render — no heavy ML libraries

import os
import json
import numpy as np
import requests
from bs4 import BeautifulSoup

# LLM variables from Render
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b")

# Load precomputed embeddings
EMB_PATH = "data/embeddings.json"

if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        DATA = json.load(f)
else:
    DATA = {"ids": [], "docs": [], "vectors": []}

DOCS = DATA["docs"]
VECS = np.array(DATA["vectors"], dtype=np.float32)

def scrape_mzu():
    """Scrape mzu.edu.in home page text."""
    try:
        res = requests.get("https://mzu.edu.in", timeout=7)
        soup = BeautifulSoup(res.text, "html.parser")
        for s in soup(["script", "style", "img"]):
            s.decompose()
        text = soup.get_text(separator=" ")
        text = " ".join(text.split())
        return text[:4000]
    except:
        return ""

def simple_keyword_search(query, k=3):
    """Lightweight keyword-based RAG search."""
    q = query.lower()
    scores = []

    for i, doc in enumerate(DOCS):
        d = doc.lower()
        score = sum(d.count(w) for w in q.split() if len(w) > 3)
        scores.append((score, i))

    scores.sort(reverse=True)
    top_docs = [DOCS[i] for score, i in scores[:k] if score > 0]

    if not top_docs:
        top_docs = DOCS[:k]

    return top_docs

def answer_query(query, k=3):
    """RAG → LLM final answer."""
    offline_docs = simple_keyword_search(query, k)
    live = scrape_mzu()

    system_prompt = "You are the MZU University Assistant. Answer using the provided documents."
    user_prompt = f"""
User question: {query}

Offline context:
{''.join(offline_docs)}

Live website extract:
{live}

Give a short, factual answer.
"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    try:
        r = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=30)
        data = r.json()

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]

        return str(data)
    except Exception as e:
        return f"⚠ LLM Error: {e}"
