# chunk.py
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == '__main__':
    from utils import read_file
    text = read_file('data/mzu_raw.txt')
    ch = chunk_text(text)
    print('Chunks:', len(ch))
