import numpy as np
from embeddings import load_embedding_model

import faiss

import os
from utils.text_processing import extract_text_from_pdf, clean_text, chunk_text

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

#This function:
#Reads all PDFs
#Extracts text
#Cleans it
#Chunks it
#Keeps metadata (VERY important later)

def process_documents():
    all_chunks = []

    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(RAW_DATA_DIR, filename)

            raw_text = extract_text_from_pdf(path)
            clean = clean_text(raw_text)
            chunks = chunk_text(clean)

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": filename
                })

    return all_chunks

def embed_chunks(chunks):
    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings




# dimension = size of embedding vector (e.g. 384)
# IndexFlatL2 = exact similarity search
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# Reload index later, Map results back to text
def save_index(index, chunks):
    faiss.write_index(index, "data/processed/index.faiss")

    with open("data/processed/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk["text"].replace("\n", " ") + "\n")



# if __name__ == "__main__":
#     chunks = process_documents()
#     print(f"Total chunks: {len(chunks)}")
#     print(chunks[0])


# FULL PIPELINE TEST 
if __name__ == "__main__":
    chunks = process_documents()
    print(f"Chunks: {len(chunks)}")

    embeddings = embed_chunks(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    save_index(index, chunks)

    print("FAISS index built and saved successfully.")

# I preprocess documents into chunks, 
# convert them into vector embeddings using a sentence transformer model, 
# and index them with FAISS to enable efficient semantic similarity search based on vector distance.


# index.faiss → vector index
# chunks.txt → text chunks
# Embedding model