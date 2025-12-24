import faiss
import numpy as np
from backend.embeddings import load_embedding_model

INDEX_PATH = "data/processed/index.faiss"
CHUNKS_PATH = "data/processed/chunks.txt"


def load_index():
    return faiss.read_index(INDEX_PATH)


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def semantic_search(query: str, top_k: int = 5):
    # Load resources
    model = load_embedding_model()
    index = load_index()
    chunks = load_chunks()

    # Embed the query
    query_embedding = model.encode([query])

    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "text": chunks[idx],
            "score": float(dist)
        })

    return results


# FAISS returns distance, not similarity
# Smaller distance = more similar


if __name__ == "__main__":
    query = "how can i write the adjacency matrix representation of a graph?"
    results = semantic_search(query)

    for i, res in enumerate(results, 1):
        print(f"\nResult {i}")
        print(f"Score: {res['score']:.4f}")
        print(res["text"][:300], "...")


# User queries are embedded into the same vector space as document chunks. 
# I then perform nearest-neighbor search using FAISS to retrieve the most semantically relevant content, 
# even when keywords differ.