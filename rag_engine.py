import pickle
import numpy as np
import requests
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load the existing vector store
@st.cache_resource
def load_rag_resources():
    with open("vectorstore/chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    embeddings = np.load("vectorstore/embeddings.npy")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return chunks, embeddings, model


# Answer a question using RAG and Ollama
def answer_question(question):
    chunks, embeddings, model = load_rag_resources()

    # Convert question into an embedding
    question_embedding = model.encode([question])

    # Find the most relevant chunks
    similarities = cosine_similarity(
        question_embedding, embeddings
    )[0]

    top_indices = similarities.argsort()[-3:][::-1]

    # Build context
    context = "\n\n".join(chunks[index] for index in top_indices)

    # Create the prompt
    prompt = f"""

You are a PDF question-answering assistant.

Rules:
1. Answer the question using only the provided context.
2. Do not use outside knowledge or make up facts.
3. If the context does not contain enough information,
reply exactly:
"The answer is not available in the PDF."
4. Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:

"""

    # Send the prompt to Ollama
    url = "http://localhost:11434/api/generate"

    data = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data, timeout=180)
    response.raise_for_status()

    result = response.json()

    sources = [
        {
            "similarity": float(similarities[index]),
            "text": chunks[index]
        }
        for index in top_indices
    ]

    return result["response"], sources
