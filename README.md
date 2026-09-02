# Semantic Similarity Search — Web App

FastAPI backend + single-page frontend for the Word2Vec semantic search
pipeline from `Semantic_Similarity_Search.ipynb`.

## Project structure

```
semantic-search-app/
├── main.py                              # FastAPI backend (loads CSV, trains/caches Word2Vec, serves API + UI)
├── requirements.txt
├── static/
│   └── index.html                       # Frontend (HTML/CSS/JS, no build step)
└── semantic_search_5000_documents.csv   # ← you add this (see below)
```

## Setup

1. **Copy your dataset into this folder.** The app expects the same CSV you
   used in the notebook, named exactly:
   ```
   semantic_search_5000_documents.csv
   ```
   placed directly next to `main.py`. It needs the columns:
   `id, title, text, keywords, category, intent`.

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS/Linux

   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   uvicorn main:app --reload
   ```

4. Open **http://127.0.0.1:8000** in your browser.

## Notes

- On the **first run**, the app trains the Word2Vec model from the CSV
  (same parameters as the notebook: `vector_size=100, window=5, min_count=1,
  sg=1`) and caches the trained model + document vectors to a `.cache/`
  folder. Subsequent restarts load instantly from cache. If you edit the
  CSV, the cache key changes automatically and it retrains.
- `GET /api/search?q=...&top_k=8&category=HR` — search endpoint.
- `GET /api/stats` — dataset/model stats used by the header of the UI.
- Interactive API docs are available at **http://127.0.0.1:8000/docs**.
