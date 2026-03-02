import hashlib
import re

MAX_CHARS = 800
MIN_CHARS = 200

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def semantic_chunk(text: str):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]

    chunks = []
    buffer = ""

    for p in paragraphs:
        if len(buffer) + len(p) <= MAX_CHARS:
            buffer += " " + p
        else:
            if len(buffer) >= MIN_CHARS:
                chunks.append(buffer.strip())
            buffer = p

    if len(buffer) >= MIN_CHARS:
        chunks.append(buffer.strip())

    return chunks


def deduplicate(chunks):
    seen = set()
    unique = []

    for c in chunks:
        h = hashlib.md5(normalize(c).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)

    return unique