

from fastapi import FastAPI, Query, UploadFile, File
import shutil
import os
from backend.indexer import process_single_pdf, update_index_with_new_chunks
#from search import semantic_search
from fastapi.middleware.cors import CORSMiddleware

from backend.search import semantic_search


app = FastAPI(
    title="University Semantic Search API",
    description="Semantic search over university documents using vector embeddings",
    version="1.0"
)

# from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

# http://127.0.0.1:8000/search?query=YOURQUERY


# I exposed the semantic search pipeline through a FastAPI service, 
# providing a documented REST endpoint that allows clients 
# to retrieve semantically relevant document chunks using vector similarity.


UPLOAD_DIR = "data/raw"  
FRONTEND_PDFS_DIR = "frontend/pdfs" 


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed"}

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save to backend storage
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Also copy to frontend folder for viewer
    frontend_path = os.path.join(FRONTEND_PDFS_DIR, file.filename)
    shutil.copyfile(file_path, frontend_path)

    # Process the new PDF and update FAISS + metadata
    new_chunks, new_embeddings = process_single_pdf(file_path)
    update_index_with_new_chunks(new_chunks, new_embeddings)

    return {"message": f"{file.filename} uploaded, indexed, and added to frontend/pdfs successfully"}

@app.get("/list-pdfs")
def list_pdfs():
    files = os.listdir("frontend/pdfs")
    files = [f for f in files if f.endswith(".pdf")]
    return {"files": files}
