import os
from pathlib import Path


# Ensure required folders always exist
def ensure_dirs():
    Path('mzu_docs').mkdir(parents=True, exist_ok=True)
    Path('data').mkdir(parents=True, exist_ok=True)


# Run once on import
ensure_dirs()


# Read a file safely
def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# Write file safely
def write_file(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
