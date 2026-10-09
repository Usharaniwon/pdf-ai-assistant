import os
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from chunker import split_text
from pdf_processor import extract_text


# ==========================================
# LOAD PDF
# ==========================================

pdf_path = "documents/sample.pdf"

text = extract_text(pdf_path)

print("\nPDF text extracted successfully.")


# ==========================================
# SPLIT TEXT INTO CHUNKS
# ==========================================

chunks = split_text(text)

print("Total chunks:", len(chunks))


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

embeddings = model.encode(chunks)

print("Embeddings created.")
print("Embedding shape:", embeddings.shape)


# ==========================================
# CREATE VECTORSTORE FOLDER
# ==========================================

os.makedirs("vectorstore", exist_ok=True)


# ==========================================
# SAVE CHUNKS
# ==========================================

with open("vectorstore/chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)


# ==========================================
# SAVE EMBEDDINGS
# ==========================================

np.save("vectorstore/embeddings.npy", embeddings)


print("\n==============================")
print("VECTOR STORE CREATED")
print("==============================")

print("Saved:")
print("vectorstore/chunks.pkl")
print("vectorstore/embeddings.npy")