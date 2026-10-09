'''from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from chunker import split_text
from pdf_processor import extract_text


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Read PDF
pdf_path = "documents/sample.pdf"

text = extract_text(pdf_path)

# Split PDF into chunks
chunks = split_text(text)

# Create embeddings for chunks
chunk_embeddings = model.encode(chunks)


# Ask a question
question = input("\nAsk a question about the PDF: ")

# Create embedding for question
question_embedding = model.encode([question])


# Calculate similarity
similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]


# Get top 3 results
top_indices = similarities.argsort()[-3:][::-1]


print("\n==============================")
print("SEARCH RESULTS")
print("==============================")


for rank, index in enumerate(top_indices, start=1):

    print(f"\n--- Result {rank} ---")
    print("Similarity:", round(similarities[index], 4))
    print(chunks[index][:500])'''
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# LOAD VECTOR STORE
# ==========================================

with open("vectorstore/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)

embeddings = np.load("vectorstore/embeddings.npy")


print("\n==============================")
print("VECTOR STORE LOADED")
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
# CALCULATE SIMILARITY
# ==========================================

similarities = cosine_similarity(
    question_embedding,
    embeddings
)[0]


# ==========================================
# GET TOP 3 RESULTS
# ==========================================

top_indices = similarities.argsort()[-3:][::-1]


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==============================")
print("SEARCH RESULTS")
print("==============================")


for rank, index in enumerate(top_indices, start=1):

    print(f"\n--- Result {rank} ---")

    print(
        "Similarity:",
        round(similarities[index], 4)
    )

    print(chunks[index][:500])