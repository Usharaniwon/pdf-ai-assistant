
import streamlit as st
import requests

from pdf_processor import process_pdf
from rag_engine import answer_question

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide"
)
# Initialize chat history before using it
if "messages" not in st.session_state:
    st.session_state["messages"] = []
st.title("📚 PDF AI Assistant")
st.write("Upload a PDF and ask questions using local AI with RAG.")

# ==========================================
# PDF UPLOAD
# ==========================================

st.subheader("1. Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Process PDF", type="primary"):

        try:
            with st.spinner(
                "Extracting text and creating embeddings..."
            ):
                result = process_pdf(uploaded_file)

            st.session_state["pdf_processed"] = True
            st.session_state["pdf_name"] = uploaded_file.name

            st.success("PDF processed successfully!")

            st.write("**File:**", uploaded_file.name)
            st.write("**Pages:**", result["pages"])
            st.write("**Chunks:**", result["chunks"])
            st.write("**Embedding shape:**", result["embedding_shape"])

        except Exception as exc:
            st.error(f"PDF processing failed: {exc}")


# ==========================================
# ASK QUESTIONS
# ==========================================

st.divider()
# Initialize chat history
if st.button("Clear Chat"):
    st.session_state["messages"] = []
    st.rerun()
st.subheader("2. Chat with your PDF")

# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept a new question
question = st.chat_input(
    "Ask a question about your PDF...",
    key="pdf_chat_input"
)

if question is not None and question.strip():
    if not st.session_state.get("pdf_processed", False):
        st.warning("Please upload and process a PDF first.")
    else:
        # Continue with storing the question
        # and calling answer_question(question.strip())
        pass
        st.session_state["messages"].append({
            "role": "user",
            "content": question
        })

        try:
            with st.spinner("Searching the PDF and generating an answer..."):
                answer, sources = answer_question(question)

            # Display and store the AI response
            with st.chat_message("assistant"):
                st.write(answer)

                with st.expander("View retrieved sources"):
                    for rank, source in enumerate(sources, start=1):
                        st.markdown(f"**Source {rank}**")
                        st.caption(
                            f"Similarity: {source['similarity']:.4f}"
                        )
                        st.write(source["text"])

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer
            })

        except requests.RequestException:
            st.error(
                "Could not connect to Ollama. "
                "Check that Ollama is running and the model is available."
            )
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")

if st.button("Get Answer"):

    if question and question.strip():
        st.warning("Please enter a question.")

    elif not st.session_state.get("pdf_processed", False):
        st.warning("Please upload and process a PDF first.")

    else:
        try:
            with st.spinner("Searching PDF and generating answer..."):
                answer, sources = answer_question(question.strip())

            st.subheader("AI Answer")
            st.write(answer)

            with st.expander("View retrieved sources"):
                for rank, source in enumerate(sources, start=1):
                    st.markdown(f"**Source {rank}**")
                    st.caption(
                        f"Similarity: {source['similarity']:.4f}"
                    )
                    st.write(source["text"])

        except requests.RequestException:
            st.error(
                "Could not connect to Ollama. "
                "Check that Ollama is running and the model is available."
            )

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
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
'''import streamlit as st

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