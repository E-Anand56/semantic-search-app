"""
Semantic Similarity Search — FastAPI backend
==============================================

Wraps the Word2Vec semantic search pipeline (originally prototyped in
Semantic_Similarity_Search.ipynb) behind a small FastAPI service, and serves
a single-page web UI for it.

Run with:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000
"""

import os
import re
import pickle
import hashlib
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("semantic-search")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "semantic_search_5000_documents.csv")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
STATIC_DIR = os.path.join(BASE_DIR, "static")

WORD2VEC_PARAMS = dict(vector_size=100, window=5, min_count=1, workers=4, sg=1)

# In-memory store populated at startup (see `load_or_train` below)
STATE = {
    "df": None,
    "model": None,
    "document_vectors": None,
    "ready": False,
    "error": None,
}


# --------------------------------------------------------------------------
# Pipeline — mirrors the notebook's preprocessing / training / search steps
# --------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_document_vector(text: str, model: Word2Vec) -> np.ndarray:
    words = text.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def _cache_key(csv_path: str) -> str:
    """Cache is keyed on the CSV's mtime + size, so edits invalidate it."""
    stat = os.stat(csv_path)
    raw = f"{csv_path}:{stat.st_mtime}:{stat.st_size}:{WORD2VEC_PARAMS}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_or_train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find '{os.path.basename(DATA_PATH)}' next to main.py.\n"
            f"Expected it at: {DATA_PATH}\n"
            f"Copy the CSV you used in the notebook into this folder and restart."
        )

    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(DATA_PATH)
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")

    if os.path.exists(cache_path):
        logger.info("Loading cached model + document vectors ...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return cached["df"], cached["model"], cached["document_vectors"]

    logger.info("No cache found — training Word2Vec from scratch (first run only) ...")
    df = pd.read_csv(DATA_PATH)

    required_cols = {"id", "title", "text", "keywords", "category", "intent"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected column(s): {missing}")

    df = df.dropna(subset=["text"]).drop_duplicates().reset_index(drop=True)
    df["clean_text"] = df["text"].apply(preprocess_text)

    tokenized_documents = df["clean_text"].apply(str.split).tolist()
    model = Word2Vec(sentences=tokenized_documents, **WORD2VEC_PARAMS)

    document_vectors = np.array(
        [get_document_vector(t, model) for t in df["clean_text"]]
    )

    with open(cache_path, "wb") as f:
        pickle.dump(
            {"df": df, "model": model, "document_vectors": document_vectors}, f
        )
    logger.info("Training complete. Cached to %s", cache_path)

    return df, model, document_vectors


def search_word2vec(query: str, top_k: int = 5, category: Optional[str] = None):
    df, model, document_vectors = STATE["df"], STATE["model"], STATE["document_vectors"]

    clean_query = preprocess_text(query)
    query_vector = get_document_vector(clean_query, model)

    if not np.any(query_vector):
        return [], []

    similarity_scores = cosine_similarity([query_vector], document_vectors)[0]
    ranked_indexes = similarity_scores.argsort()[::-1]

    results = []
    for idx in ranked_indexes:
        row = df.iloc[idx]
        if category and category != "All" and row["category"] != category:
            continue
        results.append(
            {
                "id": int(row["id"]),
                "title": row["title"],
                "text": row["text"],
                "category": row["category"],
                "intent": row["intent"],
                "score": round(float(similarity_scores[idx]), 4),
            }
        )
        if len(results) >= top_k:
            break

    # A small extra: show semantically related terms for query words that
    # exist in the trained vocabulary, so the UI can demonstrate *why*
    # results matched beyond plain keyword overlap.
    related_terms = []
    seen = set()
    for word in clean_query.split():
        if word in model.wv and word not in seen:
            seen.add(word)
            for similar_word, _ in model.wv.most_similar(word, topn=3):
                if similar_word not in seen:
                    related_terms.append(similar_word)
                    seen.add(similar_word)

    return results, related_terms[:8]


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        df, model, document_vectors = load_or_train()
        STATE.update(df=df, model=model, document_vectors=document_vectors, ready=True)
        logger.info("Ready — %d documents indexed.", len(df))
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = str(exc)
        logger.error("Startup failed: %s", exc)
    yield


app = FastAPI(title="Semantic Similarity Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    id: int
    title: str
    text: str
    category: str
    intent: str
    score: float


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[SearchResult]
    related_terms: List[str]


class StatsResponse(BaseModel):
    ready: bool
    error: Optional[str] = None
    total_documents: Optional[int] = None
    vector_size: Optional[int] = None
    vocabulary_size: Optional[int] = None
    categories: Optional[List[str]] = None


@app.get("/api/stats", response_model=StatsResponse)
def stats():
    if not STATE["ready"]:
        return StatsResponse(ready=False, error=STATE["error"])
    df, model = STATE["df"], STATE["model"]
    return StatsResponse(
        ready=True,
        total_documents=len(df),
        vector_size=model.vector_size,
        vocabulary_size=len(model.wv),
        categories=sorted(df["category"].unique().tolist()),
    )


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Free-text search query"),
    top_k: int = Query(5, ge=1, le=50),
    category: Optional[str] = Query(None),
):
    if not STATE["ready"]:
        raise HTTPException(
            status_code=503,
            detail=STATE["error"] or "Model is still loading, try again shortly.",
        )

    results, related_terms = search_word2vec(q, top_k=top_k, category=category)
    return SearchResponse(query=q, count=len(results), results=results, related_terms=related_terms)


# Serve the frontend (index.html + assets) from /static, and mount it at "/"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
