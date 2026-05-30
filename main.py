import os
import sys
import signal
import json
import numpy as np
from groq import Groq
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "your_groq_key_here")
GROQ_MODEL         = "llama-3.3-70b-versatile"
DISTILL_EVERY      = 5           # fallback turn-count trigger
TOPIC_DRIFT_THRESH = 0.25        # cosine similarity below this = topic drift
TOP_K              = 3
BM25_THRESHOLD     = 0.5
SEMANTIC_THRESHOLD = 0.25
USER_ID            = "jd"
SOUL_FILE          = f"users/{USER_ID}.txt"
BUFFER_FILE        = f"users/{USER_ID}_buffer.json"
os.makedirs("users", exist_ok=True)

# ─── INIT CLIENTS ─────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)
embedder    = SentenceTransformer("all-MiniLM-L6-v2")
chroma      = chromadb.PersistentClient(path="./chroma_db")
collection  = chroma.get_or_create_collection(f"user_{USER_ID}", metadata={"hnsw:space": "cosine"})

print("Clients ready.\n")

# ─── SOUL FILE ────────────────────────────────────────────
if not os.path.exists(SOUL_FILE):
    with open(SOUL_FILE, "w") as f:
        f.write(f"# Soul File — {USER_ID}\n\n")

def read_soul():
    with open(SOUL_FILE, "r") as f:
        return f.read()

def append_soul(summary: str, turn_range: str):
    with open(SOUL_FILE, "a") as f:
        f.write(f"[Turns {turn_range}]\n{summary}\n\n")

def get_chunks() -> list[str]:
    content = read_soul()
    blocks  = content.strip().split("\n\n")
    return [b.strip() for b in blocks if b.strip() and not b.startswith("#")]

# ─── BUFFER PERSISTENCE ───────────────────────────────────
def save_buffer(buf: list):
    with open(BUFFER_FILE, "w") as f:
        json.dump(buf, f)

def load_buffer() -> list:
    if os.path.exists(BUFFER_FILE):
        with open(BUFFER_FILE) as f:
            return json.load(f)
    return []

# ─── EMBED ────────────────────────────────────────────────
def embed(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts, normalize_embeddings=True).tolist()

# ─── DISTILLATION ─────────────────────────────────────────
def distill(turns: list[dict], reason: str = "scheduled") -> str:
    if not turns:
        return ""
    formatted = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in turns])
    prompt = f"""You are an intelligent memory engine for a personal AI assistant.
Analyze the conversation below and extract a precise memory summary.

Rules:
- Extract explicit facts: name, job, projects, tools, OS, preferences, pets, interests
- Also INFER facts logically:
  * Mentions bash scripts, apt, NVIDIA drivers, terminal → infer Linux/Pop!_OS environment
  * Mentions Rust, FastAPI, Python → log as tech stack
  * Mentions a project name → log it with any details shared
  * Mentions a pet, hobby, personal event → log it
- Do NOT include filler, greetings, or generic conversation
- Be specific — "User uses Pop!_OS" not "User uses Linux"
- Output ONLY bullet points starting with "- "
- Max 6 bullet points

Conversation:
{formatted}"""

    res = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.2
    )
    return res.choices[0].message.content.strip()

def run_distillation(buf: list, turn_range: str, reason: str = "scheduled"):
    if not buf:
        return
    print(f"\n[Memory] Distilling ({reason})...")
    summary = distill(buf, reason)
    append_soul(summary, turn_range)
    sync_vector_store()
    save_buffer([])   # clear persisted buffer
    print(f"[Memory] Soul file updated ✓")

# ─── TOPIC DRIFT DETECTION ────────────────────────────────
def detect_topic_drift(buf: list, new_message: str) -> bool:
    """
    Compare the semantic meaning of the current buffer
    vs the new incoming message.
    If similarity is below threshold → topic has drifted.
    """
    if len(buf) < 4:   # need at least 2 turns to compare
        return False
    # summarise buffer context into one string
    buf_text    = " ".join([t["content"] for t in buf[-4:]])
    vecs        = embed([buf_text, new_message])
    v1, v2      = np.array(vecs[0]), np.array(vecs[1])
    similarity  = float(np.dot(v1, v2))   # already normalised
    if similarity < TOPIC_DRIFT_THRESH:
        print(f"\n[Memory] Topic drift detected (similarity={similarity:.2f}) — triggering early distillation...")
        return True
    return False

# ─── HYBRID RETRIEVAL ─────────────────────────────────────
def hybrid_retrieve(query: str) -> str:
    chunks = get_chunks()
    if not chunks:
        return ""

    # BM25
    bm25        = BM25Okapi([c.lower().split() for c in chunks])
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top    = [
        chunks[i] for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
        if bm25_scores[i] >= BM25_THRESHOLD
    ]

    # Semantic
    q_vec   = embed([query])[0]
    sem_top = []
    if collection.count() > 0:
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "distances"]
        )
        if results["documents"]:
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                if (1 - dist) >= SEMANTIC_THRESHOLD:
                    sem_top.append(doc)

    seen, merged = set(), []
    for chunk in bm25_top + sem_top:
        if chunk not in seen:
            seen.add(chunk)
            merged.append(chunk)

    return "\n\n".join(merged[:TOP_K * 2])

# ─── UPDATE VECTOR STORE ──────────────────────────────────
def sync_vector_store():
    chunks = get_chunks()
    if not chunks:
        return
    embeddings = embed(chunks)
    ids        = [f"{USER_ID}_chunk_{i}" for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings)

# ─── GRACEFUL EXIT HANDLER ────────────────────────────────
def graceful_exit(sig=None, frame=None):
    print("\n[Memory] Intercepted shutdown — running emergency distillation...")
    if turn_buffer:
        run_distillation(turn_buffer, f"{total_turns - len(turn_buffer)//2 + 1}-{total_turns}", reason="emergency")
    else:
        save_buffer([])
    print("[Memory] Soul file safely committed. Goodbye!")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

# ─── MAIN CHAT LOOP ───────────────────────────────────────
turn_buffer = load_buffer()   # resume unsaved buffer from last session
total_turns = 0
sync_vector_store()

print(f"Chat started | Soul file: {SOUL_FILE}")
print("Type 'memory' to see soul file | 'quit' to exit\n")
print("-" * 50)

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        graceful_exit()
    if user_input.lower() == "memory":
        print("\n" + read_soul())
        continue

    total_turns += 1

    # ── STEP 1: topic drift check → early distillation if needed
    if detect_topic_drift(turn_buffer, user_input):
        run_distillation(
            turn_buffer,
            f"{total_turns - len(turn_buffer)//2}-{total_turns - 1}",
            reason="topic-drift"
        )
        turn_buffer = []

    # ── STEP 2: hybrid retrieve from soul file
    context = hybrid_retrieve(user_input)

    # ── STEP 3: build prompt + get LLM response
    messages = [{"role": "system", "content": f"""You are a helpful AI assistant talking with {USER_ID}.
You have a personal memory about this user built over time.
Rules:
- Use memory ONLY when it is directly relevant to the question
- If memory does not clearly answer something, say you don't know — NEVER guess or make up facts
- Do NOT force memory into every reply
- Do NOT repeat memory facts unless directly asked
- Answer off-topic questions (math, history, science) normally without injecting personal info
- Be concise and natural like a friend"""}]

    if context:
        messages.append({"role": "system", "content": f"[What you know about {USER_ID} — only use if relevant]\n{context}"})
    messages.append({"role": "user", "content": user_input})

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, max_tokens=1024, temperature=0.7
    )
    reply = response.choices[0].message.content.strip()
    print(f"\nBot: {reply}")

    # ── STEP 4: append to buffer + persist
    turn_buffer.append({"role": "user",      "content": user_input})
    turn_buffer.append({"role": "assistant", "content": reply})
    save_buffer(turn_buffer)   # persist after every turn

    # ── STEP 5: scheduled distillation every N turns
    if len(turn_buffer) >= DISTILL_EVERY * 2:
        run_distillation(
            turn_buffer,
            f"{total_turns - DISTILL_EVERY + 1}-{total_turns}",
            reason="scheduled"
        )
        turn_buffer = []