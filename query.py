# query.py — Local RAG + Fallback (Render Safe, concise local answers)

import os
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re

load_dotenv()

# -----------------------------
# ENV
# -----------------------------
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
SIM_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.01))  # tuned for hash embeddings

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
        return 0.0
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
# LIGHTWEIGHT HASH EMBEDDING (Render Safe)
# -----------------------------
def embed_query(text):
    vec = np.zeros(300)
    for w in text.lower().split():
        vec[hash(w) % 300] += 1
    return vec.tolist()


# -----------------------------
# HELPER: extract most relevant short answer from a matched chunk
# -----------------------------
def extract_best_sentence(query, chunk_text):
    """
    Break chunk into candidate lines/sentences and score them by keyword overlap.
    Return short, human-like sentence (or chunk summary if nothing else).
    """
    # normalize
    q = query.lower()
    # keywords: words length > 3 (common approach)
    keywords = [w for w in re.findall(r"[a-zA-Z0-9]+", q) if len(w) > 3]

    # split chunk into lines and sentences
    # prefer splitting on newline/semicolon/dash and sentences
    candidates = []
    # keep original newline-separated lines as top priority
    for part in re.split(r"\n|;|-|\u2022", chunk_text):
        part = part.strip()
        if part:
            candidates.append(part)
    # if still large, also try sentence-splitting by periods
    if len(candidates) < 3:
        for s in re.split(r"(?<=[.!?])\s+", chunk_text):
            s = s.strip()
            if s:
                candidates.append(s)

    # scoring: count occurrences of keywords (longer words weigh more)
    def score_text(t):
        tl = t.lower()
        score = 0
        for k in keywords:
            score += tl.count(k)  # frequency helps
        # also reward short answers a bit (avoid giant paragraphs)
        length_penalty = max(0, (len(t.split()) - 40) / 40)  # penalize very long candidates
        return score - length_penalty

    # compute scores
    scored = [(score_text(c), c) for c in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])

    if scored and scored[0][0] > 0:
        best = scored[0][1]
        # short cleanup: if best is long, take first 25 words
        words = best.split()
        if len(words) > 40:
            return " ".join(words[:40]) + "..."
        return best

    # fallback heuristics:
    # - if chunk contains "VC of" or "Vice", try to extract that phrase
    m = re.search(r"(VC of [^,.\n]+|Vice[- ]Chancellor[^,.\n]+|Head of Department[^,.\n]+|NIRF Ranking[^,.\n]+)", chunk_text, re.IGNORECASE)
    if m:
        return m.group(0).strip()

    # otherwise return the first line but keep it short
    first_line = candidates[0] if candidates else chunk_text
    words = first_line.split()
    if len(words) > 40:
        return " ".join(words[:40]) + "..."
    return first_line


# -----------------------------
# LOCAL SEARCH
# -----------------------------
def local_search(query):
    """Return best matching local chunk (and similarity) using cosine similarity."""
    if not DOCS or not VECS:
        return None, 0.0

    qvec = embed_query(query)
    scores = []
    for i, vec in enumerate(VECS):
        sim = cosine(qvec, vec)
        scores.append((sim, i))

    scores.sort(reverse=True, key=lambda x: x[0])
    best_sim, best_idx = scores[0]
    best_doc = DOCS[best_idx] if best_idx is not None and best_idx < len(DOCS) else None
    return best_doc, float(best_sim)


# -----------------------------
# MAIN ANSWER FUNCTION
# -----------------------------
def answer_query(query):
    # 1) Try LOCAL DATA FIRST
    local_answer_chunk, score = local_search(query)

    if local_answer_chunk and score >= SIM_THRESHOLD:
        # extract a concise sentence from the chunk
        short = extract_best_sentence(query, local_answer_chunk)
        # return a short, precise answer (no huge chunk)
        return f"📘 Local Data Answer:\n{short}"

    # 2) FALLBACK: scrape + LLM
    offline_docs = DOCS[:3]  # small context
    live_data = scrape_mzu()

    system_prompt = "You are the official Mizoram University Assistant. Answer shortly and accurately."

    user_prompt = f"""
User question:
{query}

Local data best match (score={score:.2f}):
{local_answer_chunk}

Small offline context:
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
        "temperature": 0.2,
        "max_tokens": 350
    }

    try:
        r = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=10)
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return str(data)
    except Exception as e:
        return f"⚠ Server busy. Try again.\n{e}"
