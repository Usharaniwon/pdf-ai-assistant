
import streamlit as st
import requests

from pdf_processor import process_pdf
from rag_engine import answer_question

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False

if "pdf_name" not in st.session_state:
    st.session_state["pdf_name"] = ""

st.title("📚 PDF AI Assistant")
st.write("Upload a PDF and ask questions using local AI with RAG.")

# PDF upload and processing
st.subheader("1. Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Process PDF", type="primary"):
        try:
            with st.spinner("Extracting text and creating embeddings..."):
                result = process_pdf(uploaded_file)

            st.session_state["pdf_processed"] = True
            st.session_state["pdf_name"] = uploaded_file.name
            st.session_state["messages"] = []

            st.success("PDF processed successfully!")
            st.write("**File:**", uploaded_file.name)
            st.write("**Pages:**", result["pages"])
            st.write("**Chunks:**", result["chunks"])
            st.write("**Embedding shape:**", result["embedding_shape"])

        except Exception as exc:
            st.session_state["pdf_processed"] = False
            st.error(f"PDF processing failed: {exc}")

# Chat interface
st.divider()
st.subheader("2. Chat with your PDF")

if st.session_state["pdf_processed"]:
    st.caption(f"Current PDF: {st.session_state['pdf_name']}")
else:
    st.info("Upload and process a PDF before asking questions.")

if st.button("Clear Chat"):
    st.session_state["messages"] = []
    st.rerun()

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Keep exactly one chat input
question = st.chat_input(
    "Ask a question about your PDF...",
    key="pdf_chat_input",
    disabled=not st.session_state["pdf_processed"]
)

if question and question.strip():
    question = question.strip()

    st.session_state["messages"].append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    try:
        with st.spinner("Searching the PDF and generating an answer..."):
            answer, sources = answer_question(question)

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
