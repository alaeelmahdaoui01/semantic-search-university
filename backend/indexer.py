import numpy as np
from backend.embeddings import load_embedding_model

import faiss
import json

import os
from backend.utils.text_processing import extract_text_from_pdf, clean_text, chunk_text

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

#This function:
#Reads all PDFs
#Extracts text
#Cleans it
#Chunks it
#Keeps metadata (VERY important later)

# def process_documents():
#     all_chunks = []

#     for filename in os.listdir(RAW_DATA_DIR):
#         if filename.endswith(".pdf"):
#             path = os.path.join(RAW_DATA_DIR, filename)

#             raw_text = extract_text_from_pdf(path)
#             clean = clean_text(raw_text)
#             chunks = chunk_text(clean)

#             for chunk in chunks:
#                 all_chunks.append({
#                     "text": chunk,
#                     "source": filename
#                 })


#     return all_chunks


def process_documents():
    all_chunks = []

    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(RAW_DATA_DIR, filename)

            pages = extract_text_from_pdf(path)

            for page_data in pages:
                page_text = page_data["text"]
                page_number = page_data["page"]

                chunks = chunk_text(
                    page_text,
                    source=filename,
                    page=page_number
                )

                for chunk in chunks:
                    all_chunks.append(chunk)

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

    # with open("data/processed/chunks.txt", "w", encoding="utf-8") as f:
    #     for chunk in chunks:
    #         f.write(chunk["text"].replace("\n", " ") + "\n")

    with open("data/processed/metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)



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



# This allows incremental indexing without rebuilding everything when uploading new pdfs 
def process_single_pdf(path):
    
    pages = extract_text_from_pdf(path)
    all_chunks = []

    for page_data in pages:
        chunks = chunk_text(
            page_data["text"],
            source=os.path.basename(path),
            page=page_data["page"]
        )
        all_chunks.extend(chunks)

    embeddings = load_embedding_model().encode([c["text"] for c in all_chunks])
    print(f"Processed {len(all_chunks)} chunks from {path}")
    return all_chunks, embeddings

#Update FAISS index with new chunks
def update_index_with_new_chunks(new_chunks, new_embeddings):

    # Load existing index
    index = faiss.read_index("data/processed/index.faiss")

    # Add new embeddings
    index.add(new_embeddings)

    # Save updated index
    faiss.write_index(index, "data/processed/index.faiss")

    # Load existing metadata
    with open("data/processed/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Append new chunks
    metadata.extend(new_chunks)

    # Save metadata
    with open("data/processed/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)



# when you upload a new PDF

# File is saved (both in data/raw and frontend/pdfs/)

# Backend calls process_single_pdf(file_path) →
# Extracts pages
# Chunks each page
# Adds metadata (text, source, page)
# Creates embeddings

# Backend calls update_index_with_new_chunks(new_chunks, new_embeddings) →
# Loads existing FAISS index
# Adds new embeddings
# Loads existing metadata.json
# Appends new_chunks to the metadata list
# Saves updated metadata.json


