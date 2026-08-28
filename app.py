import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="My RAG App", page_icon="📄")
st.title("📄 Ask Questions About Your PDF")

# Load the embedding model once (cached so it doesn't reload every time)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def chunk_text(text, chunk_size=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and "chunks" not in st.session_state:
    with st.spinner("Reading and processing PDF..."):
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        chunks = chunk_text(full_text)
        chunk_embeddings = embedder.encode(chunks)

        st.session_state.chunks = chunks
        st.session_state.chunk_embeddings = chunk_embeddings

    st.success(f"PDF processed! ({len(st.session_state.chunks)} chunks created)")

# Ask a question
if "chunks" in st.session_state:
    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Finding answer..."):
            question_embedding = embedder.encode(question)
            scores = cos_sim(question_embedding, st.session_state.chunk_embeddings)[0]

            # Get top 3 chunks instead of just 1
            top_indices = scores.argsort(descending=True)[:3]
            top_chunks = [st.session_state.chunks[i] for i in top_indices]
            combined_context = "\n\n---\n\n".join(top_chunks)

            client = genai.Client(api_key=api_key)
            prompt = f"""Answer the question based only on the context below.

Context:
{combined_context}

Question: {question}
"""
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

        st.markdown("### Answer:")
        st.write(response.text)

        with st.expander("See retrieved context"):
            st.write(combined_context)