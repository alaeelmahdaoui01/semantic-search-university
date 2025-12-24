

from fastapi import FastAPI, Query
#from search import semantic_search

from backend.search import semantic_search


app = FastAPI(
    title="University Semantic Search API",
    description="Semantic search over university documents using vector embeddings",
    version="1.0"
)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.get("/search")
def search(
    query: str = Query(..., description="Natural language search query"),
    top_k: int = Query(5, ge=1, le=10, description="Number of results to return")
):
    results = semantic_search(query, top_k)
    return {
        "query": query,
        "results": results
    }



# Query(...):  Documents parameters, Validates input

# top_k: Prevents abuse, Shows backend thinking

# JSON response: Frontend-ready, Easy to extend




# We already have:   semantic_search(query) → results
# Now we expose it as:  GET /search?query=...




# Run the API
# From the project root:  uvicorn backend.api:app --reload
# You should see:  Uvicorn running on http://127.0.0.1:8000


# I exposed the semantic search pipeline through a FastAPI service, 
# providing a documented REST endpoint that allows clients 
# to retrieve semantically relevant document chunks using vector similarity.