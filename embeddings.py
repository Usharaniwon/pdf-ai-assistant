from sentence_transformers import SentenceTransformer
from chunker import split_text
from pdf_processor import extract_text


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


if __name__ == "__main__":

    pdf_path = "documents/sample.pdf"

    # Extract PDF text
    text = extract_text(pdf_path)

    # Split text into chunks
    chunks = split_text(text)

    # Create embeddings
    embeddings = model.encode(chunks)

    print("\n==============================")
    print("EMBEDDING GENERATION")
    print("==============================")

    print("Total chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)

    print("\nFirst chunk:")
    print(chunks[0][:300])

    print("\nFirst embedding:")
    print(embeddings[0][:10])