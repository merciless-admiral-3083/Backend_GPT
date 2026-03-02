import os
import faiss
import json
from sentence_transformers import SentenceTransformer

DATA_DIR = "rag_data"
INDEX_DIR = "rag_index"
os.makedirs(INDEX_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
sources = []

import os

import tiktoken
enc = tiktoken.get_encoding("gpt2")

MAX_TOKENS_PER_CHUNK = 80   # 👈 key knob (60–120 is ideal)

def chunk_text(text, max_tokens=80):
    tokens = enc.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield enc.decode(tokens[i:i+max_tokens])
        
for root, _, files in os.walk(DATA_DIR):
    for fname in files:
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(root, fname)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            for chunk in chunk_text(content, MAX_TOKENS_PER_CHUNK):
                chunk = chunk.strip()
                if len(chunk) > 30:
                    texts.append(chunk)
                    sources.append(path)

print(f"Loaded {len(texts)} chunks")

embeddings = model.encode(texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

with open(os.path.join(INDEX_DIR, "data.json"), "w", encoding="utf-8") as f:
    json.dump(
        [{"text": t, "source": s} for t, s in zip(texts, sources)],
        f,
        indent=2
    )

print("RAG index built successfully")