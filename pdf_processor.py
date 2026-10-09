
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

VECTORSTORE = Path("vectorstore")
CHUNKS_FILE = VECTORSTORE / "chunks.pkl"
EMBEDDINGS_FILE = VECTORSTORE / "embeddings.npy"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages_text.append(
                f"[Page {page_number}]\n{text.strip()}"
            )

    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise ValueError(
            "No extractable text found. Scanned PDFs may require OCR."
        )

    return full_text


def split_text_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def process_pdf(uploaded_file):
    full_text = extract_pdf_text(uploaded_file)
    chunks = split_text_into_chunks(full_text)

    if not chunks:
        raise ValueError("No usable text chunks were created.")

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    VECTORSTORE.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_FILE, "wb") as file:
        pickle.dump(chunks, file)

    np.save(EMBEDDINGS_FILE, embeddings)

    return {
        "pages": len(PdfReader(uploaded_file).pages),
        "chunks": len(chunks),
        "embedding_shape": embeddings.shape
    }
