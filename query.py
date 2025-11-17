# query.py — Hybrid Local RAG + LLM fallback (precise answers)

import os
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re

load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
SIM_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.01))

EMB_PATH = "data/embeddings.json"
if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        DATA = json.load(f)
else:
    DATA = {"docs": [], "vectors": []}

DOCS = DATA.get("docs", [])
VECS = DATA.get("vectors", [])

# Precompute lowercase versions for fast text matching
DOCS_LC = [d.lower() for d in DOCS]

def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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

def embed_query(text):
    vec = np.zeros(300)
    for w in text.lower().split():
        vec[hash(w) % 300] += 1
    return vec.tolist()

def extract_best_sentence(query, chunk_text):
    q = query.lower()
    keywords = [w for w in re.findall(r"[a-zA-Z0-9]+", q) if len(w) > 3]
    # split chunk into lines first
    candidates = [p.strip() for p in re.split(r"\n|;|-|\u2022", chunk_text) if p.strip()]
    if len(candidates) < 3:
        for s in re.split(r"(?<=[.!?])\s+", chunk_text):
            s = s.strip()
            if s:
                candidates.append(s)
    def score_text(t):
        tl = t.lower()
        score = 0
        for k in keywords:
            score += tl.count(k)
        length_penalty = max(0, (len(t.split()) - 40) / 40)
        return score - length_penalty
    scored = [(score_text(c), c) for c in candidates]
    scored.sort(reverse=True, key=lambda x: x[0])
    if scored and scored[0][0] > 0:
        best = scored[0][1]
        words = best.split()
        if len(words) > 40:
            return " ".join(words[:40]) + "..."
        return best
    m = re.search(r"(VC of [^,.\n]+|Vice[- ]Chancellor[^,.\n]+|Head of Department[^,.\n]+|NIRF Ranking[^,.\n]+)", chunk_text, re.IGNORECASE)
    if m:
        return m.group(0).strip()
    first_line = candidates[0] if candidates else chunk_text
    words = first_line.split()
    if len(words) > 40:
        return " ".join(words[:40]) + "..."
    return first_line

# ------------------------
# New: exact / synonym mapping and simple substring search
# ------------------------
SYNONYMS = {
    "vc": ["vice-chancellor", "vice chancellor", "vc", "vicechancellor"],
    "hod": ["head of department", "hod", "head of dept", "head of"],
    "dean": ["dean", "dean, school", "dean of"],
    "nirf": ["nirf", "ranking"],
    "naac": ["naac"],
    "hostel": ["hostel", "hostels", "hall of residence"],
    "placement": ["placement", "placed", "package"],
    "contact": ["contact", "email", "phone"],
    "library": ["library", "rfid", "books"],
    "students": ["student", "students", "undergraduate", "postgraduate", "doctoral"],
}

def simple_keyword_search(query):
    q = query.lower()
    # 1) direct substring match (highest priority)
    for i, d in enumerate(DOCS_LC):
        if q in d:
            return DOCS[i], 1.0  # perfect match
    # 2) synonyms-based match (map short queries like "vc" -> find chunk containing vice chancellor)
    for key, tokens in SYNONYMS.items():
        for tok in tokens:
            if tok in q:
                # find chunk that contains any of the tokens
                for i, d in enumerate(DOCS_LC):
                    if any(t in d for t in tokens):
                        return DOCS[i], 0.9
    # 3) token-overlap scoring (fast deterministic fallback before vectors)
    q_words = [w for w in re.findall(r"[a-zA-Z0-9]+", q) if len(w) > 3]
    if not q_words:
        return None, 0.0
    scores = []
    for i, d in enumerate(DOCS_LC):
        score = sum(d.count(w) for w in q_words)
        scores.append((score, i))
    scores.sort(reverse=True)
    best_score, best_idx = scores[0]
    if best_score > 0:
        return DOCS[best_idx], float(best_score)
    return None, 0.0

def local_search(query):
    # 1) Try simple deterministic search first
    doc, score = simple_keyword_search(query)
    if doc is not None and score > 0:
        # return early with high confidence (we'll still extract a concise sentence)
        return doc, score
    # 2) vector cosine fallback
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

def answer_query(query):
    # local search
    local_chunk, score = local_search(query)
    # If simple search returned an integer count (token overlap) we treat it as match
    # For substring/synonym we returned 0.9/1.0 which is already > SIM_THRESHOLD
    if local_chunk and (score >= SIM_THRESHOLD or isinstance(score, float) and score >= 0.1):
        # extract short sentence for human-like response
        short = extract_best_sentence(query, local_chunk)
        return f"📘 Local Data Answer:\n{short}"
    # fallback to scraping + LLM
    offline_docs = DOCS[:3]
    live_data = scrape_mzu()
    system_prompt = "You are the official Mizoram University Assistant. Answer shortly and accurately."
    user_prompt = f"""
User question:
{query}

Local data best match (score={score:.2f}):
{local_chunk}

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
