import os
from pathlib import Path

def ensure_dirs():
    Path('mzu_docs').mkdir(parents=True, exist_ok=True)
    Path('data').mkdir(parents=True, exist_ok=True)

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
