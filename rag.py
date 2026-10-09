import pickle
import numpy as np
import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# LOAD VECTOR STORE
# ==========================================

with open("vectorstore/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)

embeddings = np.load("vectorstore/embeddings.npy")

print("\n==============================")
print("PDF AI ASSISTANT - RAG")
print("==============================")

print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# ASK QUESTION
# ==========================================

question = input("\nAsk a question about the PDF: ")


# ==========================================
# CREATE QUESTION EMBEDDING
# ==========================================

question_embedding = model.encode([question])


# ==========================================
# SEARCH RELEVANT CHUNKS
# ==========================================

similarities = cosine_similarity(
    question_embedding,
    embeddings
)[0]

top_indices = similarities.argsort()[-3:][::-1]


# ==========================================
# BUILD CONTEXT
# ==========================================

context = ""

for index in top_indices:

    context += chunks[index]
    context += "\n\n"


# ==========================================
# CREATE RAG PROMPT
# ==========================================

prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say exactly:

"The answer is not available in the PDF."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


# ==========================================
# SEND REQUEST TO OLLAMA
# ==========================================

url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.2:3b",
    "prompt": prompt,
    "stream": False
}

response = requests.post(url, json=data)

result = response.json()


# ==========================================
# DISPLAY AI ANSWER
# ==========================================

print("\n==============================")
print("AI ANSWER")
print("==============================")

print(result["response"])


# ==========================================
# DISPLAY SOURCES
# ==========================================

print("\n==============================")
print("RETRIEVED SOURCES")
print("==============================")


for rank, index in enumerate(top_indices, start=1):

    print(f"\n--- Source {rank} ---")

    print(
        "Similarity:",
        round(similarities[index], 4)
    )

    print(chunks[index][:300])

''' import streamlit as st

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF AI Assistant")
st.write("Ask questions about your PDF using AI.")

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"PDF uploaded successfully: {uploaded_file.name}")

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if st.button("Get Answer"):
        if question.strip():
            st.info("Your question: " + question)
            st.write("Next, we will connect this interface to your RAG pipeline and Ollama.")
        else:
            st.warning("Please enter a question first.")'''