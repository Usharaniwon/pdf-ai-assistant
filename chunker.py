def split_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


if __name__ == "__main__":

    from pdf_processor import extract_text

    pdf_path = "documents/sample.pdf"

    text = extract_text(pdf_path)

    chunks = split_text(text)

    print("\n==============================")
    print("TEXT CHUNKING")
    print("==============================")

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:5], start=1):

        print(f"\n--- Chunk {i} ---")
        print(chunk[:300])