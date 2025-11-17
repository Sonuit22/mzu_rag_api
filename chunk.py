# chunk.py — clean text chunker for embeddings

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Splits text into overlapping chunks.
    - chunk_size: number of words per chunk
    - overlap: repeated words between chunks
    """
    words = text.split()
    n = len(words)

    if n == 0:
        return []

    chunks = []
    start = 0

    while start < n:
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # next chunk starts overlap before the end
        start = end - overlap
        if start < 0:
            start = 0

    return chunks
