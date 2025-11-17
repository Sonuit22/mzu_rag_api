# query.py — Perfect Accuracy Version (Knowledge → RAG → LLM)

import os
import json
import requests
import numpy as np
import re
import difflib
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import string

load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
SIM_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.01))


# ============================================================
# 1) LOAD KNOWLEDGE JSON (TOP PRIORITY)
# ============================================================
KB_PATH = "data/knowledge.json"
if os.path.exists(KB_PATH):
    with open(KB_PATH, "r", encoding="utf-8") as f:
        KB = json.load(f)
else:
    KB = {}

# Normalize keys
KB_NORM = {key.lower().strip(): value for key, value in KB.items()}


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def norm(t):
    t = t.lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    return " ".join(t.split())


def tokens(t):
    return [w for w in norm(t).split() if len(w) > 2]


# ============================================================
# 2) KNOWLEDGE BASE SEARCH
# ============================================================
def search_kb(query):
    q = norm(query)

    # exact match
    if q in KB_NORM:
        return KB_NORM[q]

    # partial match (key inside query)
    for key in KB_NORM:
        if key in q:
            return KB_NORM[key]

    # fuzzy match (high similarity only)
    keys = list(KB_NORM.keys())
    match = difflib.get_close_matches(q, keys, n=1, cutoff=0.85)
    if match:
        return KB_NORM[match[0]]

    # token overlap with key (≥ 60%)
    qtok = set(tokens(q))
    if not qtok:
        return None

    for key in keys:
        ktok = set(tokens(key))
        if not ktok:
            continue
        overlap = len(qtok & ktok) / len(ktok)
        if overlap >= 0.6:
            return KB_NORM[key]

    return None


# ============================================================
# 3) LOAD EMBEDDINGS (SECOND PRIORITY)
# ============================================================
EMB_PATH = "data/embeddings.json"
if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        DATA = json.load(f)
else:
    DATA = {"docs": [], "vectors": []}

DOCS = DATA.get("docs", [])
VECS = DATA.get("vectors", [])
DOCS_NORM = [norm(d) for d in DOCS]


# ============================================================
# VECTOR UTILS
# ============================================================
def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_query(text):
    vec = np.zeros(300)
    for w in tokens(text):
        vec[hash(w) % 300] += 1
    return vec.tolist()


# ============================================================
# SIMPLE KEYWORD SEARCH (STRICT)
# ============================================================
def simple_keyword_search(query):
    qtok = set(tokens(query))
    if not qtok:
        return None, 0.0

    best_idx = None
    best_score = 0

    for i, d in enumerate(DOCS_NORM):
        dtok = set(tokens(d))
        if not dtok:
            continue

        overlap = len(qtok & dtok) / len(qtok)  # ratio match
        if overlap > best_score:
            best_score = overlap
            best_idx = i

    if best_score >= 0.5:  # require ≥50% token match
        return DOCS[best_idx], best_score

    return None, 0.0


# ============================================================
# VECTOR SEARCH
# ============================================================
def vector_search(query):
    if not VECS:
        return None, 0.0

    qvec = embed_query(query)
    sims = [(cosine(qvec, v), i) for i, v in enumerate(VECS)]
    sims.sort(reverse=True)

    best_sim, idx = sims[0]
    return DOCS[idx], best_sim


# ============================================================
# BEST SENTENCE EXTRACTION
# ============================================================
def extract_best_sentence(query, chunk):
    qtok = set(tokens(query))

    lines = [x.strip() for x in re.split(r"\n|;|-|\u2022|\.", chunk) if x.strip()]
    if not lines:
        return chunk[:80] + "..."

    best = None
    best_score = -999

    for line in lines:
        lt = norm(line)
        ltok = set(tokens(lt))
        if not ltok:
            continue

        overlap = len(qtok & ltok) / len(qtok)
        length_penalty = len(line.split()) / 35.0
        score = overlap - length_penalty

        if score > best_score:
            best_score = score
            best = line

    if not best:
        return lines[0][:60] + "..."

    words = best.split()
    return " ".join(words[:30]) + ("..." if len(words) > 30 else "")


# ============================================================
# SCRAPE WEBSITE (FALLBACK)
# ============================================================
def scrape_mzu():
    try:
        r = requests.get("https://mzu.edu.in", timeout=4)
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "img"]):
            t.decompose()
        return " ".join(soup.get_text(" ").split())[:3000]
    except:
        return ""


# ============================================================
# FINAL ANSWER PIPELINE
# ============================================================
def answer_query(query):

    # 1) KNOWLEDGE JSON (perfect)
    kb = search_kb(query)
    if kb:
        return f"{kb}"

    # 2) SIMPLE KEYWORD SEARCH (strict)
    doc, score = simple_keyword_search(query)
    if doc and score >= 0.5:
        return extract_best_sentence(query, doc)

    # 3) VECTOR SEARCH
    vdoc, vscore = vector_search(query)
    if vdoc and vscore >= SIM_THRESHOLD:
        return extract_best_sentence(query, vdoc)

    # 4) LLM FALLBACK
    offline = " ".join(DOCS[:3])
    live = scrape_mzu()

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Answer briefly and accurately."},
            {"role": "user", "content": f"Question: {query}\nOffline: {offline}\nWebsite: {live}"}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Groq-Version": "2024-10-14",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=10)
        data = r.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return "I couldn't fetch a proper answer."
    except:
        return "Website unreachable. Try again later."
