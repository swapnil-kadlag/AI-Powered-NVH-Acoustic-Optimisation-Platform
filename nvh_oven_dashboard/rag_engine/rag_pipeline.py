"""
rag_engine/rag_pipeline.py — Phase 5: NVH Knowledge RAG Engine
===============================================================
Hybrid retrieval-augmented generation pipeline:
  1. Chunk knowledge base documents (RecursiveCharacterTextSplitter)
  2. Embed with sentence-transformers (all-MiniLM-L6-v2)
  3. Store in FAISS local vector DB
  4. BM25 sparse retrieval (rank_bm25) for keyword fallback
  5. Hybrid RRF fusion (Reciprocal Rank Fusion)
  6. Query routing by intent

Offline / no API key required: 
  Uses local sentence-transformer embeddings + simple extractive QA.
  Set USE_LLM=True and provide ANTHROPIC_API_KEY to get full answers.
"""

import re
import json
import math
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE    = Path(__file__).resolve().parent.parent
KB_DIR  = BASE / "rag_engine" / "knowledge_base"
INDEX_DIR = BASE / "rag_engine" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────
CHUNK_SIZE    = 400    # chars (~100 tokens)
CHUNK_OVERLAP = 80
EMBED_MODEL   = "all-MiniLM-L6-v2"
TOP_K_DENSE   = 5
TOP_K_SPARSE  = 5
TOP_K_FINAL   = 4

# ── Document routing map ────────────────────────────────────────
ROUTE_MAP = {
    "mitigation":  "Mitigation_library.txt",
    "fix":         "Mitigation_library.txt",
    "reduce":      "Mitigation_library.txt",
    "standard":    "NVH_standards.txt",
    "iec":         "NVH_standards.txt",
    "iso":         "NVH_standards.txt",
    "limit":       "NVH_standards.txt",
    "tpa":         "SPR_methodology.txt",
    "spr":         "SPR_methodology.txt",
    "path":        "SPR_methodology.txt",
    "ntf":         "SPR_methodology.txt",
    "ods":         "SPR_methodology.txt",
}


# ══════════════════════════════════════════════════════════════
# A — Document chunking
# ══════════════════════════════════════════════════════════════

def recursive_split(text: str, chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP) -> list[str]:
    """
    Recursive character splitter: tries to break on double-newline,
    then single newline, then space.
    """
    separators = ["\n\n", "\n", " ", ""]
    chunks = []

    def split_rec(txt, seps):
        if not seps or len(txt) <= chunk_size:
            if txt.strip():
                chunks.append(txt.strip())
            return
        sep = seps[0]
        parts = txt.split(sep)
        current = ""
        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current += (sep if current else "") + part
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = part
        if current.strip():
            chunks.append(current.strip())

    split_rec(text, separators)

    # Apply overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i-1][-overlap:]
            overlapped.append(tail + " " + chunks[i])
        return overlapped
    return chunks


def load_and_chunk_kb() -> tuple[list[str], list[dict]]:
    """
    Load all .txt files from knowledge_base/, chunk them.
    Returns (chunks, metadata_list).
    """
    all_chunks = []
    all_meta   = []

    for fpath in sorted(KB_DIR.glob("*.txt")):
        text = fpath.read_text(encoding="utf-8")
        doc_chunks = recursive_split(text)
        for i, chunk in enumerate(doc_chunks):
            all_chunks.append(chunk)
            all_meta.append({
                "source":    fpath.name,
                "chunk_idx": i,
                "char_len":  len(chunk),
            })

    return all_chunks, all_meta


# ══════════════════════════════════════════════════════════════
# B — FAISS dense index
# ══════════════════════════════════════════════════════════════

def build_faiss_index(chunks: list[str],
                      model_name=EMBED_MODEL) -> tuple:
    """Embed chunks and build FAISS flat L2 index."""
    print(f"    Loading embedding model: {model_name} …")
    model = SentenceTransformer(model_name)
    print(f"    Embedding {len(chunks)} chunks …")
    embeddings = model.encode(chunks, show_progress_bar=False,
                               batch_size=32, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine similarity (normalised)
    index.add(embeddings.astype(np.float32))
    return index, model, embeddings


# ══════════════════════════════════════════════════════════════
# C — BM25 sparse index
# ══════════════════════════════════════════════════════════════

def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    """Tokenise chunks and build BM25Okapi index."""
    tokenised = [re.findall(r"\w+", c.lower()) for c in chunks]
    return BM25Okapi(tokenised)


# ══════════════════════════════════════════════════════════════
# D — Hybrid RRF retrieval
# ══════════════════════════════════════════════════════════════

def rrf_fusion(dense_ranks: list[int], sparse_ranks: list[int],
               k: int = 60) -> dict[int, float]:
    """
    Reciprocal Rank Fusion:
    score(d) = Σ 1 / (k + rank_i(d))
    """
    scores: dict[int, float] = {}
    for rank, doc_id in enumerate(dense_ranks):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ranks):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def detect_intent(query: str) -> Optional[str]:
    """Route query to specific document if keyword matches."""
    q = query.lower()
    for kw, doc in ROUTE_MAP.items():
        if kw in q:
            return doc
    return None  # search all


def retrieve(query: str, chunks: list[str], meta: list[dict],
             faiss_index, bm25: BM25Okapi, model: SentenceTransformer,
             top_k=TOP_K_FINAL, route_doc: Optional[str] = None
             ) -> list[dict]:
    """
    Hybrid retrieval:
      1. FAISS top-K dense
      2. BM25 top-K sparse
      3. RRF fusion
      4. Optional: filter to route_doc if intent detected
    """
    # Optional pre-filter by document source
    if route_doc:
        candidate_idx = [i for i, m in enumerate(meta) if m["source"] == route_doc]
    else:
        candidate_idx = list(range(len(chunks)))

    if not candidate_idx:
        candidate_idx = list(range(len(chunks)))

    # Dense retrieval
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores_d, idxs_d = faiss_index.search(q_emb, min(TOP_K_DENSE * 3, len(chunks)))
    dense_hits = [i for i in idxs_d[0] if i in candidate_idx][:TOP_K_DENSE]

    # Sparse retrieval (BM25 over candidate subset)
    subset_chunks = [chunks[i] for i in candidate_idx]
    bm25_sub = BM25Okapi([re.findall(r"\w+", c.lower()) for c in subset_chunks])
    bm25_scores = bm25_sub.get_scores(re.findall(r"\w+", query.lower()))
    sparse_local = np.argsort(bm25_scores)[::-1][:TOP_K_SPARSE]
    sparse_hits  = [candidate_idx[i] for i in sparse_local]

    # RRF
    rrf_scores = rrf_fusion(dense_hits, sparse_hits)
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        results.append({
            "chunk":  chunks[doc_id],
            "source": meta[doc_id]["source"],
            "score":  round(score, 4),
        })
    return results


# ══════════════════════════════════════════════════════════════
# E — Simple extractive answer (offline / no LLM)
# ══════════════════════════════════════════════════════════════

def extractive_answer(query: str, retrieved: list[dict]) -> str:
    """
    Fallback: return most relevant retrieved chunk as answer with citation.
    For production: replace with LLM call via Anthropic / Ollama.
    """
    if not retrieved:
        return "No relevant information found in the NVH knowledge base."

    best = retrieved[0]
    context_lines = [r["chunk"] for r in retrieved[:2]]
    context = "\n---\n".join(context_lines)

    answer = (
        f"**Query:** {query}\n\n"
        f"**Retrieved Context (top-{len(retrieved[:2])} chunks):**\n\n"
        f"{context}\n\n"
        f"**Sources:** {', '.join(set(r['source'] for r in retrieved))}"
    )
    return answer


# ══════════════════════════════════════════════════════════════
# F — Full RAG pipeline class
# ══════════════════════════════════════════════════════════════

class NVHRagEngine:
    """Stateful RAG engine — load once, query many times."""

    def __init__(self):
        self.chunks = None
        self.meta   = None
        self.faiss_index = None
        self.bm25   = None
        self.model  = None
        self.is_built = False

    def build(self):
        print("[RAG] Loading & chunking knowledge base …")
        self.chunks, self.meta = load_and_chunk_kb()
        print(f"      {len(self.chunks)} chunks from {len(set(m['source'] for m in self.meta))} documents")

        print("[RAG] Building FAISS dense index …")
        self.faiss_index, self.model, _ = build_faiss_index(self.chunks)

        print("[RAG] Building BM25 sparse index …")
        self.bm25 = build_bm25_index(self.chunks)

        self.is_built = True
        print("[RAG] ✅ Engine ready\n")

    def query(self, question: str, top_k=TOP_K_FINAL) -> dict:
        if not self.is_built:
            self.build()

        intent = detect_intent(question)
        retrieved = retrieve(question, self.chunks, self.meta,
                             self.faiss_index, self.bm25, self.model,
                             top_k=top_k, route_doc=intent)
        answer = extractive_answer(question, retrieved)

        return {
            "question":   question,
            "intent":     intent or "general",
            "answer":     answer,
            "sources":    list(set(r["source"] for r in retrieved)),
            "retrieved":  retrieved,
        }


# ══════════════════════════════════════════════════════════════
# MAIN — Demo queries
# ══════════════════════════════════════════════════════════════
def run_rag_demo():
    print("="*62)
    print("  MHC Oven — RAG Engine Demo  (Phase 5)")
    print("="*62 + "\n")

    engine = NVHRagEngine()
    engine.build()

    demo_queries = [
        "What does IEC 60704-1 require for domestic microwave ovens?",
        "How do I reduce fan blade-pass frequency noise?",
        "What is the SPR methodology and how is NTF measured?",
        "What mitigation options exist for magnetron hum at 100 Hz?",
        "Recommend a fix to meet the 52 dBA target within $10 budget",
    ]

    for q in demo_queries:
        print(f"{'─'*60}")
        print(f"Q: {q}")
        result = engine.query(q)
        print(f"Intent routed to: {result['intent']}")
        print(f"Sources: {result['sources']}")
        # Show snippet of answer
        snippet = result["answer"][:500].replace("\n", " ")
        print(f"Answer (snippet): {snippet}…\n")

    # Save index metadata
    meta_out = {"chunks": len(engine.chunks),
                "documents": list(set(m["source"] for m in engine.meta)),
                "embed_model": EMBED_MODEL}
    (BASE / "rag_engine" / "index_meta.json").write_text(
        json.dumps(meta_out, indent=2))
    print(f"✅  Phase-5 RAG engine ready | {meta_out}")
    return engine


if __name__ == "__main__":
    engine = run_rag_demo()
