# rag/convert_instruction_json.py

import json
import os

INPUT_FILE = "data/instruction_clean.json"
OUTPUT_DIR = "rag_data/from_instructions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

seen = set()
chunks = []

for item in data:
    text = item.get("output", "").strip()

    # basic quality filters
    if len(text) < 40:
        continue
    if text in seen:
        continue

    seen.add(text)
    chunks.append(text)

BATCH_SIZE = 200
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    fname = f"instructions_{i//BATCH_SIZE}.txt"
    with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write("\n\n".join(batch))

print(f"Converted {len(chunks)} instruction outputs into RAG text files")