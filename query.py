# query.py
# Lightweight RAG + Live Scraping + Groq LLM (llama-3.1-8b-instant)

import os
import json
import numpy as np
import requests
from bs4 import BeautifulSoup

# ====== ENV VARS (Render) ======
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# IMPORTANT: Latest valid Groq model
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")


# ====== LOAD EMBEDDINGS ======
EMB_PATH = "data/embeddings.json"

if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        DATA = json.load(f)
else:
    DATA = {"ids": [], "docs": [], "vectors": []}

DOCS = DATA["docs"]
VECS = np.array(DATA.get("vectors", []), dtype=np.float32)


# ====== LIVE SCRAPER ======
def scrape_mzu():
    """Scrape mzu.edu.in homepage for fresh real-time info."""
    try:
        res = requests.get("https://mzu.edu.in", timeout=7)
        soup = BeautifulSoup(res.text, "html.parser")

        # remove unnecessary tags
        for tag in soup(["script", "style", "img", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ")
        text = " ".join(text.split())

        return text[:4000]
    except:
        return ""


# ====== LIGHTWEIGHT RAG SEARCH ======
def simple_keyword_search(query, k=3):
    q_words = [w for w in query.lower().split() if len(w) > 3]
    scores = []

    for i, doc in enumerate(DOCS):
        score = sum(doc.lower().count(w) for w in q_words)
        scores.append((score, i))

    scores.sort(reverse=True)
    top_docs = [DOCS[i] for score, i in scores[:k] if score > 0]

    return top_docs if top_docs else DOCS[:k]


# ====== FINAL ANSWER GENERATOR ======
def answer_query(query, k=3):
    offline_docs = simple_keyword_search(query, k)
    live_extract = scrape_mzu()

    system_prompt = (
        "You are the official Mizoram University Assistant. "
        "Use ONLY the provided offline text and live website extract. "
        "Be short, factual, and accurate."
    )

    user_prompt = f"""
User Question:
{query}

Offline Knowledge:
{''.join(offline_docs)}

Live Website Extract:
{live_extract}

Answer concisely.
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

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return str(data)

    except Exception as e:
        return f"⚠ LLM Error: {e}"
