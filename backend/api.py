from fastapi import FastAPI

app = FastAPI(title="Semantic Search API")

@app.get("/")
def health_check():
    return {"status": "ok"}
