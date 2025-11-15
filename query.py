# query.py — Fast MZU RAG with lightweight multi-page scraper

import os
import json
import numpy as np
import requests
from bs4 import BeautifulSoup

# ========= ENV ==========
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# ========= LOAD EMBEDDINGS ==========
EMB_PATH = "data/embeddings.json"

if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        DATA = json.load(f)
else:
    DATA = {"docs": [], "vectors": []}

DOCS = DATA["docs"]


# ========= LIGHT SCRAPER ==========
FAST_PAGES = [
    "https://mzu.edu.in",
    "https://mzu.edu.in/contact-us/",
    "https://mzu.edu.in/department-of-information-technology/",
    "https://mzu.edu.in/examination-news-results/",
    "https://mzu.edu.in/message-by-vice-chancellor/"
]

def scrape_page(url):
    try:
        res = requests.get(url, timeout=4)
        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "img", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ")
        return " ".join(text.split())[:2000]

    except:
        return ""


def scrape_live_data():
    text = ""
    for url in FAST_PAGES:
        text += scrape_page(url) + "\n\n"
    return text[:4000]   # limit for speed


# ========= LIGHT RETRIEVAL ==========
def simple_keyword_search(query, k=3):
    q_words = [w for w in query.lower().split() if len(w) > 3]
    scores = []

    for i, doc in enumerate(DOCS):
        d = doc.lower()
        score = sum(d.count(w) for w in q_words)
        scores.append((score, i))

    scores.sort(reverse=True)
    top_docs = [DOCS[i] for score, i in scores[:k] if score > 0]

    return top_docs if top_docs else DOCS[:k]


# ========= FINAL ANSWER ==========
def answer_query(query, k=3):

    offline_docs = simple_keyword_search(query, k)
    live_data = scrape_live_data()

    system_prompt = (
        "You are the official Mizoram University Assistant. "
        "Answer using the offline text + live website data. "
        "If publicly known (like NIRF rank, departments), you MAY use general knowledge. "
        "Stay factual and short."
    )

    user_prompt = f"""
User question:
{query}

Offline documents:
{''.join(offline_docs)}

Live website data:
{live_data}

Give a short and accurate answer.
"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "Groq-Version": "2024-10-14"
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
        r = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=10)
        data = r.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return str(data)

    except Exception as e:
        return f"⚠ Server busy. Try again.\nError: {e}"
