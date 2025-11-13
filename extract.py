# extract.py
import fitz
import os
from utils import ensure_dirs, write_file

ensure_dirs()

INPUT_DIR = 'mzu_docs'
OUT_FILE = 'data/mzu_raw.txt'

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        t = page.get_text()
        if t:
            text.append(t)
    return '\n'.join(text)

all_text = []
for fname in sorted(os.listdir(INPUT_DIR)):
    if fname.lower().endswith('.pdf'):
        path = os.path.join(INPUT_DIR, fname)
        print('Extracting', path)
        text = extract_text_from_pdf(path)
        header = f"\n\n=== SOURCE: {fname} ===\n\n"
        all_text.append(header + text)

write_file(OUT_FILE, '\n'.join(all_text))
print('Wrote', OUT_FILE)
