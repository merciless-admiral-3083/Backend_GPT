---
title: RAG_GPT_DEPLOY
app_file: gradio_app.py
sdk: gradio
sdk_version: 6.8.0
---
# GPT-RAG Assistant (Strictly context-grounded generation)

## 📋 Project Overview

GPT-RAG is a comprehensive framework for building a **Retrieval-Augmented Generation (RAG) system** that combines a custom-trained GPT language model with a semantic search mechanism to provide factual, Hallucination-resistant responses. The system is designed to answer questions using only information from a provided knowledge base, with built-in safeguards to refuse answering out-of-scope queries.

### Key Innovation
Rather than relying solely on the language model's training data (which can lead to hallucinations), this system:
1. **Retrieves** relevant context from a knowledge base using semantic similarity
2. **Gates** the response based on confidence thresholds
3. **Generates** answers exclusively from retrieved context
4. **Refuses** gracefully when knowledge is unavailable

---

## 🎯 Key Features

- **Custom GPT Model**: Transformer-based language model built from scratch using PyTorch
- **Semantic Search**: FAISS-indexed vector search using MiniLM embeddings for fast context retrieval
- **Confidence-based Gating**: Distance threshold mechanism to prevent answering without sufficient context
- **Instruction Fine-tuning**: Specialized training on instruction-response pairs for chat applications
- **Large-scale Pretraining**: Supports large-scale pretraining on FineWeb-Edu (10B tokens)
- **QA**: Context-grounded generative QA with strict retrieval gating
- **Safe Refusal**: Explicit "I don't know" responses for out-of-scope questions

---

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  1. Query Embedding             │
│     (Sentence Transformers)     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  2. Semantic Search             │
│     (FAISS Index)               │
│     Retrieve top-k documents    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  3. Confidence Gating           │
│     Distance < Threshold?       │
└────────┬─────────────────────────┤
         │ YES                 NO │
         ▼                        ▼
    ┌─────────┐         ┌──────────────┐
    │ Proceed │         │ Safe Refusal │
    └────┬────┘         └──────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  4. Generate Answer             │
│     GPT Model with Context      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘
```

---

## 📁 Project Structure

```
GPT_RAG/
├── train.py                    # Core GPT model implementation
├── chat.py                     # Interactive chat interface
├── instruction_train.py        # Fine-tuning on instruction datasets
├── fineweb.py                  # FineWeb-Edu dataset download & tokenization
├── test_rag.py                 # RAG system testing
├── diagnose.py                 # Diagnostic tools
├── generate_instructions_fixed.py  # Instruction dataset generation
│
├── rag/                        # RAG System Module
│   ├── config.py              # RAG configuration (thresholds, top-k, etc.)
│   ├── rag_retriever.py       # Semantic retrieval implementation
│   ├── build_rag_index.py     # FAISS index creation
│   └── convert_instruction_json.py  # Format conversion utilities
│
├── data/                       # Training Data
│   ├── instruction_clean.json  # Cleaned instruction-response pairs
│   ├── instructions.txt        # Raw instructions
│   ├── text_corpus.txt         # Text corpus for knowledge base
│   ├── train.npy               # Tokenized training data
│   ├── raw/                    # Raw data files
│   └── clean/                  # Cleaned data files
│
├── edu_fineweb10B/             # FineWeb-Edu Dataset (10B tokens)
│   ├── edufineweb_train_000001.npy  # Tokenized shards
│   ├── edufineweb_train_000002.npy
│   └── ... (57 total shards)
│
├── rag_data/                   # RAG-specific data
├── rag_index/                  # FAISS index & metadata
│   ├── index.faiss             # Vector index
│   └── data.json               # Document metadata
│
├── log/                        # Model checkpoints & logs
│   ├── config.json             # Training configuration
│   └── model_chat.pt           # Fine-tuned chat model
│
├── model_chat/                 # Chat model artifacts
├── tokenizer/                  # GPT-2 tokenizer files
│
└── README.md                   # This file
```

---

## 🛠️ Component Details

### 1. **train.py** - GPT Language Model
Implements a transformer-based GPT model from scratch:

**Architecture:**
- **Causal Self-Attention**: Multi-head attention with causal masking for autoregressive generation
- **MLP Layers**: Feed-forward networks with GELU activation
- **Block Stacking**: 4 transformer blocks (configurable)
- **Configuration**:
  - Block size: 128 tokens
  - Embedding dim (n_embd): 256
  - Attention heads (n_head): 4
  - Layers (n_layer): 4
  - Vocabulary: 50,257 (GPT-2 tokenizer)

**Key Classes:**
- `CausalSelfAttention`: Multi-head self-attention with causal masking
- `MLP`: Feed-forward layer (4x expansion)
- `Block`: Transformer block combining attention + MLP
- `GPTConfig`: Configuration dataclass
- `GPT`: Full model class with weight initialization, optimizer configuration, and pretrained loading

**Features:**
- Support for loading pretrained GPT-2 weights
- Gradient clipping and weight decay optimization
- Custom weight initialization scheme (NANOGPT_SCALE_INIT)

### 2. **chat.py** - Interactive Chat Interface
`chat.py` is the interactive entrypoint that wires together the FAISS-based RAG retriever and the local GPT model. It implements a hybrid strategy that
- attempts to answer from retrieved context first (RAG), and
- falls back to constrained GPT generation when RAG confidence is low.

**Highlights (from the current implementation):**
- Loads the fine-tuned chat model checkpoint (default: `log/main/model_chat.pt`) when available; otherwise runs in RAG-only mode.
- Uses `tiktoken` GPT-2 encoding for prompt/token handling.
- Applies a strict context validation and sentence-extraction pipeline to avoid low-quality context.
- Hybrid decision logic: computes a RAG confidence score and decides between the RAG answer and GPT-generated answer.

**Minimal prompt used by `chat.py`:**
```
Context: {context[:350]}

Q: {question}
A:
```

This implementation does not rely on an external `prompt_template` file or variable — the prompt is constructed inline as shown above.

**Command-line Arguments (as implemented):**
```
--model        Path to model checkpoint (default: log/main/model_chat.pt)
--device       Device to run on (default: cpu)
--temperature  Sampling temperature (default: 0.4)
--top_k        Top-k filtering parameter for sampling (default: 50)
--max_tokens   Maximum tokens to generate per query (default: 100)
--rag_weight   RAG confidence weight threshold (default: 0.90)
--debug        Enable debug prints
```

**Runtime behavior notes:**
- Retrieved documents are filtered by distance threshold and a context-quality check before joining into a single context.
- The GPT generator uses top-k sampling, token-level repetition checks, and early stopping heuristics to avoid loops and low-quality outputs.
- When updating docs or examples, prefer the compact prompt above rather than the older multi-line template.

### 3. **instruction_train.py** - Instruction Fine-tuning
Specialized training script for adapting the model to follow instructions:

**Process:**
1. Load base pretrained GPT model
2. Build instruction-response pairs from JSON dataset
3. Mask loss computation to only train on response tokens
4. Train for 3 epochs with learning rate 1e-5
5. Save instruction-tuned model

**Key Features:**
- **Selective Loss Masking**: Only compute loss on the response portion, not the instruction
- **Batch Building**: Pads sequences to block size and creates x, y, y_masked tensors
- **Gradient Clipping**: Prevents exploding gradients
- **Validation**: Skips malformed examples (< 10 tokens)

### 4. **fineweb.py** - Large-scale Dataset Processing
Downloads and tokenizes the FineWeb-Edu 10B token dataset:

**Dataset:**
- Source: HuggingFace FineWeb-Edu (sample-10BT split)
- Size: 10 billion tokens
- Tokenizer: GPT-2 encoder
- Shard Size: 100M tokens per shard (results in 100 shards)

**Features:**
- Multiprocessing tokenization for speed
- Memory-efficient shard writing
- End-of-text token separation between documents
- Uint16 token representation for storage efficiency

### 5. **rag/rag_retriever.py** - Semantic Search
FAISS-based vector retrieval system:

**Architecture:**
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Index**: FAISS for fast approximate nearest neighbor search
- **Metadata**: JSON file mapping index positions to documents

**Methods:**
- `retrieve(query, top_k)`: Return top-k most similar documents with similarity distances

### 6. **rag/config.py** - RAG Configuration
System parameters for controlling retrieval and gating:

```python
BEST_DISTANCE_THRESHOLD = 1.2  # Confidence threshold
TOP_K_RETRIEVAL = 3             # Number of documents to retrieve
MAX_CHUNKS_RETURNED = 2         # Maximum chunks in final context
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch
pip install tiktoken
pip install transformers
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu
pip install numpy
pip install datasets  # For FineWeb download
pip install tqdm
```

### Step 1: Prepare Training Data

#### Option A: Use FineWeb-Edu (10B tokens)
```bash
python fineweb.py
```
This downloads and tokenizes the FineWeb-Edu dataset into `edu_fineweb10B/` directory.

#### Option B: Use Your Own Data
Prepare a text file and tokenize it using the tokenization utilities in `data/`.

### Step 2: Pretrain Base Model
```bash
python train.py \
    --data_dir=edu_fineweb10B \
    --out_dir=log/main \
    --epochs=1 \
    --batch_size=32 \
    --device=cuda
```

This trains a base GPT model on the FineWeb-Edu data.

### Step 3: Build RAG Index

First, prepare your knowledge base text in `data/text_corpus.txt`, then:
```bash
---
title: RAG-GPT Assistant
---

# RAG-GPT Assistant — Retrieval-Augmented Generation (RAG)

Brief: A self-contained repository that combines a local GPT model with a FAISS-backed semantic retriever to produce context-grounded, low-hallucination answers. This README documents setup, usage, and architecture for reproducible experiments and local deployment.

**Contents**
- **Overview** — what this project does and design goals
- **Quick Start** — minimal steps to run the demo
- **Detailed Setup** — environment, dependencies, data, and GPU notes
- **Core Workflows** — training, fine-tuning, RAG index build, and chat/Gradio
- **Code Map** — description of important files and directories
- **Configuration & Tuning** — parameters to adjust
- **Troubleshooting** — common issues and fixes
- **Contributing & License**

This fine-tunes the base model on the instruction dataset for better chat performance.

### Step 5: Run Interactive Chat
```bash
python chat.py \
    --model=log/main/model_chat.pt \
    --device=cuda \
    --temperature=0.2
```

---

## 💬 Usage Examples

### Interactive Chat Mode

**Example 1: Factual Question (with Context)**
```
Q: What is recursion?
Retrieved Context: "Recursion is a programming technique where a function calls itself..."
A: Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems.
```

**Example 2: Out-of-Scope Question (no Context)**
```
Q: Who invented electricity?
Retrieved Context: (distance > threshold, insufficient match)
A: I don't know based on the given context.
```

**Example 3: Partial Context**
```
Q: What are the main types of machine learning?
Retrieved Context: "Supervised learning uses labeled data. Unsupervised learning finds patterns..."
A: Based on the provided context, there are at least two main types: supervised learning (which uses labeled data) and unsupervised learning (which finds patterns without labels).
**Quick Start**

1) Create and activate a Python environment (recommended):

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (cmd)
.\.venv\Scripts\activate.bat
# or on Unix:
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Build or load a RAG index (see 'Build the RAG index') and place it under `rag_index/`.

4) Run the interactive chat (CPU example):

```bash
python chat.py --device=cpu
```

5) Or run the Gradio demo locally:

```bash
python gradio_app.py
```

---

**Notes**: If you already have a trained model checkpoint (e.g., `model_final.pt` or `log/main/model_chat.pt`), pass its path to `--model` flags when running scripts.

---

**Detailed Setup**

- Python: 3.8+ recommended
- Primary libs: `torch`, `transformers` (optional), `sentence-transformers`, `faiss-cpu` (or `faiss-gpu`), `tiktoken`, `numpy`, `tqdm`, `gradio` (for UI)

Install the typical dependencies (CPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If you have a CUDA-capable GPU, install the appropriate `torch` wheel for your CUDA version for faster training and generation.

Environment tips:
- Use `faiss-gpu` if you have a GPU and want faster retrieval.
- For reproducible results, pin package versions in `requirements.txt` and use a venv/conda env.

---

## Core Workflows

1) Prepare knowledge base

- Place plain text files (one document per line or long text) into `rag_data/` or create `data/text_corpus.txt`.

2) Build the RAG index

```bash
cd rag
python build_rag_index.py --input ../data/text_corpus.txt --out ../rag_index
```

This script chunks documents, computes embeddings (all-MiniLM-like), and writes a FAISS index plus metadata JSON into `rag_index/`.

3) Train / pretrain model (optional)

The repo contains `train.py` for training a GPT-like model from tokenized data.

Example (quick test/train run using CPU):

```bash
python train.py --data_dir=data --out_dir=log/test --epochs=1 --batch_size=8 --device=cpu
```

4) Instruction fine-tuning

Use `instruction_train.py` to adapt the base model to instruction-following datasets (instruction/response JSON format):

```bash
python instruction_train.py --base_model=log/test/model_final.pt --out_dir=log/test --device=cpu
```

5) Run the chat interface

```bash
python chat.py --model=log/test/model_chat.pt --device=cpu --temperature=0.2
```

6) Run Gradio UI (optional)

```bash
python gradio_app.py
```

---

## Configuration & Parameters

Primary configuration lives in `rag/config.py` and top-level script flags.

- `BEST_DISTANCE_THRESHOLD` — retrieval distance threshold controlling confidence gating (lower = stricter)
- `TOP_K_RETRIEVAL` — how many nearest neighbors to fetch
- `MAX_CHUNKS_RETURNED` — cap on final context chunks used for generation

Generation parameters (CLI flags to `chat.py`): `--temperature`, `--top_k`, `--max_tokens`, `--rag_weight`

Tune these to balance specificity vs. creativity; for strict factual answers prefer low `temperature` (0.0–0.3) and stricter distance thresholds.

---

## Code Map (important files)

- `train.py` — model training and low-level GPT implementation
- `instruction_train.py` — fine-tuning on instruction-response datasets
- `chat.py` — CLI interactive RAG+GPT chat runner
- `gradio_app.py` — Gradio demo frontend
- `rag/` — RAG module with:
  - `build_rag_index.py` — create FAISS index and metadata
  - `rag_retriever.py` — retriever wrapper used at query time
  - `config.py` — retrieval and gating config
- `rag_data_clean/` — raw/clean documents used for indexing (example dataset)
- `rag_index/` — built FAISS index and JSON metadata (output)
- `model_final.pt` — example trained model checkpoint (if present)

---

## Examples

Programmatic retrieval + generation (minimal):

```python
from rag.rag_retriever import RAGRetriever
import torch

# load retriever
rag = RAGRetriever('rag_index/index.faiss', 'rag_index/data.json')
results = rag.retrieve('What is gradient descent?', top_k=3)
context = '\n'.join([r['text'] for r in results])

# load model
ckpt = torch.load('log/main/model_chat.pt', map_location='cpu')
# build prompt and generate as in chat.py
```

CLI example:

```bash
python chat.py --device=cpu --model=log/main/model_chat.pt
```

---

## Troubleshooting & Tips

- Index build failure / memory errors: chunk your corpus into smaller pieces and use `faiss-cpu` or `faiss-gpu` with batching.
- Slow embedding computation: cache embeddings to disk. Reuse caches when rebuilding indexes.
- Model checkpoint not loading: confirm the checkpoint `config` matches model code (block size, n_embd, etc.).
- Non-deterministic outputs: set random seeds for `numpy`, `torch`, and Python `random` when evaluating reproducibility.

Common quick fixes:

```bash
# Recreate venv and reinstall dependencies
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducibility

- Save `log/` directory for checkpoints and configs.
- Keep `rag_index/` and `rag_data_clean/` archived with versioning for dataset provenance.

---

## Contributing

1. Open an issue describing the feature or bug.
2. Create a branch `feature/your-feature`.
3. Add tests where applicable and update `README.md` if behavior changes.

---

## License & Attribution

This repository does not include any third-party model weights by default. Ensure you comply with licenses for datasets and models you download (e.g., Hugging Face, Sentence-Transformers, FAISS).

If you want, I can also:
- run a quick smoke test (build small index + run chat)
- add example scripts for dataset preprocessing
- create a small `.env` or `config.example.json` to centralize paths

---

End of README.
- [ ] Implement reranking stage for context quality
- [ ] Support for dynamic knowledge base updates
- [ ] Integration with vector databases (Pinecone, Weaviate)
- [ ] Web search integration for current events
- [ ] Multilingual support
- [ ] Streaming generation for real-time responses
- [ ] Caching system for common queries
- [ ] Metrics & evaluation harness (BLEU, ROUGE, F1)

---

## 📝 License & Attribution

This project combines custom implementations with community datasets:
- **Model**: Custom PyTorch implementation
- **Data**: FineWeb-Edu (HuggingFace)
- **Embeddings**: Sentence Transformers (BAAI)
- **Search**: FAISS (Meta)

---

## 🤝 Contributing

To extend this project:

1. **Add new datasets**: Update `data/` and `fineweb.py`
2. **Improve retrieval**: Modify `rag/rag_retriever.py`
3. **Enhance generation**: Tune parameters in `chat.py`
4. **Add features**: Create new modules in appropriate directories

---

## 📧 Contact & Support

For issues, questions, or suggestions:
- Check existing test files
- Review diagnostic output
- Consult troubleshooting section above
