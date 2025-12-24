from pypdf import PdfReader
import re

# def extract_text_from_pdf(pdf_path: str) -> str:
#     reader = PdfReader(pdf_path)
#     text = ""

#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + "\n"

#     return text



def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
# Removes weird spacing and Makes chunking easier


# Reads each page, Extracts readable text, Ignores empty pages
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "text": clean_text(text),
                "page": i + 1
            })

    return pages


# def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
#     words = text.split()
#     chunks = []

#     start = 0
#     while start < len(words):
#         end = start + chunk_size
#         chunk = words[start:end]
#         chunks.append(" ".join(chunk))
#         start = end - overlap

#     return chunks


def chunk_text(text, chunk_size=400, overlap=50, source=None, page=None):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        chunks.append({
            "text": " ".join(chunk),
            "source": source,
            "page": page
        })

        start = end - overlap

    return chunks


# print(chunk_text("kdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" \
# "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" \
# "ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" \
# "ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" \
# "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"))

# Keeps context, Avoids cutting ideas abruptly, Overlap improves recall