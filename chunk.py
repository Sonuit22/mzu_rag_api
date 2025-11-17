# chunk.py — line-based chunking (best for structured facts)

def chunk_text(text):
    """
    Splits text into chunks separated by blank lines.
    Each block becomes one chunk.
    """
    raw_blocks = text.split("\n\n")  # split on blank line

    # Clean blocks
    chunks = []
    for block in raw_blocks:
        block = block.strip()
        if len(block) > 5:  # avoid tiny lines
            chunks.append(block)

    return chunks
