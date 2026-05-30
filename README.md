# FileRAG Memory Architecture 🧠

> **The .txt file as the soul of a personal AI.**  
> A novel memory architecture for LLMs that stores conversation history as a distilled plain-text file and retrieves it using hybrid BM25 + semantic RAG.

**By [Dharanidharan J (JD)](https://www.linkedin.com/in/dharanidharan-j-7757a9321/) — Full Stack & AI Engineer | Building Jarvix**

---

## The Problem

Every AI memory approach today has a fatal flaw:

| Approach | Problem |
|---|---|
| Dict | Lost on restart. 0% accuracy after 500 turns |
| Redis | Persistent but buries facts under noise |
| Vector DB | 4+ GB at 1000 turns. Needs infrastructure |
| **FileRAG** | **18 KB soul file. 100% accuracy at 1000 turns** |

---

## The Idea

Instead of storing messages — store a **relationship**.

Every conversation is distilled into a plain `.txt` soul file:

```
[Turns 1-5]
- User's name is JD, software engineer
- Building FileRAG on Pop!_OS with NVIDIA GPU
- Has a cat named Pixel

[Turns 6-10]
- Paused TaskNest due to burnout
- Now focused on AgenticMesh
- Interested in Mars colonisation
```

Human readable. Editable. Private. Owned by the user.

---

## Architecture

```
User message
     ↓
Topic Drift Check (cosine similarity < 0.25?)
     ├── YES → Distill buffer immediately
     └── NO  → continue
     ↓
Hybrid Retrieval (BM25 + ChromaDB) from soul file
     ↓
Inject context → LLM responds
     ↓
Append to buffer → save buffer to disk
     ↓
Every 5 turns → Distill → Append to soul file → Update ChromaDB
     ↓
On exit (Ctrl+C) → Emergency distillation → Soul file committed
```

### Key Innovations

- **Topic-Drift Distillation** — distills immediately when topic changes, not just every N turns
- **Deduplication** — skips chunks >92% similar to existing ones, keeps soul file clean
- **Emergency Exit Handler** — SIGINT/SIGTERM intercepted, nothing lost on crash
- **Hybrid Retrieval** — BM25 for exact keywords + semantic for meaning

---

## Benchmark Results

### Accuracy across scales

| Scale | Dict | Redis | Vector DB | **FileRAG** |
|---|---|---|---|---|
| 20 turns | 100% | 100% | 67% | **100%** |
| 500 turns | **0% ❌** | **0% ❌** | 67% | **100%** |
| 1000 turns | **0% ❌** | **0% ❌** | 67% | **100%** |

### Storage at 1000 turns

| Approach | Total Storage |
|---|---|
| Dict | 69 KB (lost on restart) |
| Redis | 67 KB (TTL-based) |
| Vector DB | **4,338 KB** |
| FileRAG soul file | **18 KB** |

---

## Stack

```
LLM           → Groq (llama3-70b-8192)
Distillation  → Groq every 5 turns or on topic drift
Embeddings    → sentence-transformers/all-MiniLM-L6-v2
Vector Store  → ChromaDB (persistent local)
Retrieval     → Hybrid BM25 + Cosine Semantic
Memory        → {user_id}.txt — the soul file
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/dharanidh75/filerag-memory
cd filerag-memory

# Install
uv venv && source .venv/bin/activate.fish
uv add groq chromadb rank-bm25 sentence-transformers python-dotenv fakeredis tabulate

# Set env vars
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Run chatbot
python main.py

# Run benchmarks
python benchmark.py
```

---

## Project Structure

```
filerag-memory/
├── main.py          ← full chatbot (single file)
├── benchmark.py     ← compare Dict vs Redis vs VectorDB vs FileRAG
├── .env.example     ← env vars template
├── requirements.txt
├── README.md
└── users/
    └── {user_id}.txt   ← soul file (auto-created)
```

---

## Where It Shines

| Use Case | Why FileRAG |
|---|---|
| Local AI assistants | No cloud, no infra, full privacy |
| Developer tools | Remembers your stack across sessions |
| Mental health apps | Data stays on device, human readable |
| Educational tutors | Adapts to student over a semester |
| Research companions | Tracks papers, questions, hypotheses |

**Not recommended for:** High concurrency SaaS, sub-millisecond latency requirements.

---

## Roadmap

- [ ] v2 — Structured memory schema (facts / preferences / timeline)
- [ ] v3 — Multi-user support
- [ ] v4 — Fine-tuned distillation model
- [ ] v5 — FastAPI production layer

---

## Built By

**Dharanidharan J (JD)** — Full Stack & AI Engineer  
[LinkedIn](https://www.linkedin.com/in/dharanidharan-j-7757a9321/) · [GitHub](https://www.github.com/dharanidh75) · Building [Jarvix](https://github.com/dharanidh75/jarvix) — a local-first voice AI assistant for Pop!_OS

---

*The soul file is 18 KB. Your AI deserves better than a dict.*
