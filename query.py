# query.py
# Full MZU RAG: Web Scraping + Multi-Page + PDF Reading + Groq LLM

import os
import json
import numpy as np
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF (for PDFs)

# ==========================
# ENV VARIABLES FROM RENDER
# ==========================
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# ==========================
# LOAD LOCAL EMBEDDINGS
# ==========================
EMB_PATH = "data/embeddings.json"

if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        DATA = json.load(f)
else:
    DATA = {"docs": [], "vectors": []}

DOCS = DATA["docs"]
VECS = np.array(DATA.get("vectors", []), dtype=np.float32)


# ==========================
# PDF TEXT EXTRACTION
# ==========================
def extract_pdf_text(url):
    """Download and read PDF content."""
    try:
        r = requests.get(url, timeout=10)
        pdf = fitz.open(stream=r.content, filetype="pdf")

        txt = ""
        for page in pdf:
            txt += page.get_text()

        return txt
    except:
        return ""


# ==========================
# SCRAPE MZU MULTIPLE PAGES
# ==========================
MZU_PAGES = [
    "https://mzu.edu.in",
    "https://mzu.edu.in/refund-policy/",
    "https://mzu.edu.in/admission-brochures/",
    "https://mzu.edu.in/contact-us/",
    "https://mzu.edu.in/schools-departments/",
    "https://mzu.edu.in/department-of-information-technology/",
    "https://mzu.edu.in/department-of-computer-engineering/",
    "https://mzu.edu.in/department-of-electronics-communication-engineering/",
    "https://mzu.edu.in/department-of-civil-engineering-department/",
    "https://mzu.edu.in/department-of-electrical-engineering/",
    "https://mzu.edu.in/dean-students-welfare/",
    "https://mzu.edu.in/examination-news-results/",
    "https://mzu.edu.in/sports/",
    "https://stjmzu.org/index.php/journal",
    "https://mzu.edu.in/message-by-vice-chancellor/",
    "https://mzu.edu.in/visitor/",
    "https://mzu.edu.in/registrar/",
    "https://lib.mzu.edu.in/",
    "https://mzu.edu.in/certifications-accreditations/"
    "https://mzu.edu.in/affiliated-institutes/",
    "https://sites.google.com/mzu.edu.in/gallery?usp=sharing"
]

def scrape_page(url):
    """Scrape a single webpage."""
    try:
        res = requests.get(url, timeout=8)
        soup = BeautifulSoup(res.text, "html.parser")

        # Remove junk
        for tag in soup(["script", "style", "img", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ")
        text = " ".join(text.split())
        return text
    except:
        return ""


def scrape_live_data():
    """Scrape multiple MZU pages + PDF links."""
    collected = ""

    # Step 1 — scrape HTML pages
    for url in MZU_PAGES:
        collected += scrape_page(url) + "\n\n"

    # Step 2 — detect PDF links in main site
    try:
        res = requests.get("https://mzu.edu.in", timeout=8)
        soup = BeautifulSoup(res.text, "html.parser")
        pdfs = [a["href"] for a in soup.find_all("a", href=True) if a["href"].lower().endswith(".pdf")]

        for link in pdfs[:5]:  # limit for safety
            pdf_url = link if link.startswith("http") else f"https://mzu.edu.in/{link}"
            collected += extract_pdf_text(pdf_url)
    except:
        pass

    return collected[:8000]   # limit for LLM


# ==========================
# LIGHTWEIGHT KEYWORD SEARCH
# ==========================
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


# ==========================
# FINAL ANSWER GENERATOR
# ==========================
def answer_query(query, k=3):

    offline_docs = simple_keyword_search(query, k)
    live_data = scrape_live_data()

    system_prompt = (
        "You are the official Mizoram University Assistant. "
        "Use the offline documents + live website data to answer. "
        "If something is obvious or publicly known (like NIRF ranking, departments, HOD names), "
        "you MAY use general knowledge as well. "
        "Keep answers short, factual, and accurate."
    )

    user_prompt = f"""
User question:
{query}

Offline documents:
{''.join(offline_docs)}

Live website data:
{live_data}

Give the BEST factual answer.
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
        "temperature": 0.2,
        "max_tokens": 500
    }

    try:
        r = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=30)
        data = r.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        return str(data)

    except Exception as e:
        return f"⚠ LLM Error: {e}"
