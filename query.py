# query.py — Updated Version (Local RAG + Fallback + Lightweight)

import os
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# ENV
# -----------------------------
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
SIM_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.40))

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
EMB_PATH = "data/embeddings.json"

if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        DATA = json.load(f)
else:
    DATA = {"docs": [], "vectors": []}

DOCS = DATA.get("docs", [])
VECS = DATA.get("vectors", [])


# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -----------------------------
# SCRAPE MZU HOMEPAGE (fallback)
# -----------------------------
def scrape_mzu():
    try:
        res = requests.get("https://mzu.edu.in", timeout=4)
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "img"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:3000]
    except:
        return ""


# -----------------------------
# LOCAL EMBEDDING (MATCHES CREATE_EMBEDDINGS)
# -----------------------------
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def embed_query(text):
    """Generate embedding for user query (LOCAL MINI LM)"""
    try:
        return embedder.encode(text).tolist()
    except:
        return None


def local_search(query):
    """Return best matching local chunk using cosine similarity"""
    if not DOCS or not VECS:
        return None, 0.0

    qvec = embed_query(query)
    if qvec is None:
        return None, 0.0

    scores = []
    for i, vec in enumerate(VECS):
        sim = cosine(qvec, vec)
        scores.append((sim, DOCS[i]))

    scores.sort(reverse=True)
    best_sim, best_doc = scores[0]
    return best_doc, best_sim


# -----------------------------
# MAIN ANSWER FUNCTION
# -----------------------------
def answer_query(query):
    # 1) Try LOCAL DATA FIRST
    local_answer, score = local_search(query)

    if score >= SIM_THRESHOLD:
        return f"📘 **Local Data Answer:**\n{local_answer}"

    # 2) FALLBACK: scrape + LLM
    offline_docs = DOCS[:3]  # small context
    live_data = scrape_mzu()

    system_prompt = "You are the official Mizoram University Assistant. Answer shortly and accurately."

    user_prompt = f"""
User question:
{query}

Local data top match (score={score:.2f}):
{local_answer}

Offline docs:
{" ".join(offline_docs)}

Live website extract:
{live_data}

If you don't know, say you don't know.
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
