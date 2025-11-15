# query.py — Fast Version (works with GitHub Pages + Render Free)

import os
import json
import requests
import numpy as np
from bs4 import BeautifulSoup

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Load small embeddings file
EMB_PATH = "data/embeddings.json"

if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        DATA = json.load(f)
else:
    DATA = {"docs": [], "vectors": []}

DOCS = DATA["docs"]


def scrape_mzu():
    """FAST scrape of homepage only."""
    try:
        res = requests.get("https://mzu.edu.in", timeout=4)
        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "img"]):
            tag.decompose()

        text = " ".join(soup.get_text(" ").split())
        return text[:3000]
    except:
        return ""
    

def simple_keyword_search(query, k=3):
    q = query.lower()
    scores = []

    for i, doc in enumerate(DOCS):
        score = sum(doc.lower().count(w) for w in q.split() if len(w) > 3)
        scores.append((score, i))

    scores.sort(reverse=True)
    return [DOCS[i] for score, i in scores[:k] if score > 0] or DOCS[:k]


def answer_query(query, k=3):

    offline_docs = simple_keyword_search(query, k)
    live_data = scrape_mzu()

    system_prompt = "You are the official Mizoram University Assistant. Answer shortly and accurately."

    user_prompt = f"""
User question:
{query}

Offline data:
{''.join(offline_docs)}

Live website extract:
{live_data}

Answer concisely.
"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Groq-Version": "2024-10-14",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.25,
        "max_tokens": 350
    }

    try:
        r = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=8)
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return str(data)
    except Exception as e:
        return f"⚠ Server busy. Try again.\n{e}"
