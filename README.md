# 📄 PDF Q&A — A RAG-Powered Document Assistant

A simple yet powerful **Retrieval-Augmented Generation (RAG)** application that lets you upload any PDF and ask natural language questions about its content. Built from scratch to understand and demonstrate how modern AI systems retrieve relevant information and generate grounded, accurate answers.

🔗 **Live Demo:** https://rag-project-krnqrtbchasg5fiqdjfc8s.streamlit.app/

---

## 🎯 What This Project Does

Instead of relying on an AI's general knowledge (which has limits and cutoffs), this app grounds every answer in **your own document**. Upload a PDF, ask a question, and get an answer generated *only* from the relevant parts of that file — with the retrieved context shown transparently.

---

## 🧠 How It Works (The RAG Pipeline)

1. **Text Extraction** — The uploaded PDF is parsed and its full text is extracted using `pypdf`.
2. **Chunking** — The extracted text is split into smaller, manageable chunks.
3. **Embedding** — Each chunk is converted into a numerical vector representation using the `sentence-transformers` model (`all-MiniLM-L6-v2`), capturing semantic meaning rather than just keywords.
4. **Retrieval** — When a question is asked, it's embedded the same way, and **cosine similarity** is used to find the most relevant chunks from the document.
5. **Generation** — The retrieved chunks, along with the original question, are passed to **Google's Gemini API**, which generates a natural, accurate answer grounded strictly in the provided context.
6. **Interface** — Everything is wrapped in an interactive web UI built with **Streamlit**.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Web Interface | Streamlit |
| PDF Parsing | pypdf |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Similarity | Cosine similarity |
| LLM (Generation) | Google Gemini API |
| Secrets Management | python-dotenv |

---

## 🚀 Getting Started (Run Locally)

### Prerequisites
- Python 3.10+
- A free [Google Gemini API key](https://aistudio.google.com/apikey)

### Installation

```bash
# Clone the repository
git clone https://github.com/Fatima-art12/rag-project.git
cd rag-project

# Install dependencies
pip install -r requirements.txt
```

### Set Up Your API Key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

> ⚠️ Never commit your `.env` file — it's already excluded via `.gitignore`.

### Run the App

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📸 How to Use

1. Upload any PDF document.
2. Wait for it to be processed (text extraction + embedding).
3. Type a question about the document's content.
4. Get an AI-generated answer, along with the exact retrieved context it was based on.

---

## 💡 Key Learnings From Building This

- How embeddings capture semantic meaning beyond exact keyword matching
- How retrieval narrows down large documents to only the most relevant information
- Why RAG is essential for large-scale or private data that can't fit into a model's context window
- Secure handling of API keys using environment variables and `.gitignore`
- Deploying a Python + AI application as a live web app

---

## 🔮 Possible Improvements

- Support for multiple PDFs at once
- Smarter chunking (sentence-aware instead of fixed character count)
- Persistent vector storage using a dedicated vector database (e.g. ChromaDB)
- Support for other file types (Word, TXT, web pages)

---

## 👩‍💻 Author

**Fatima Aboul**
Computer Science Student | DHA Suffa University

---

*This project was built as a hands-on learning exercise to understand Retrieval-Augmented Generation (RAG) from the ground up — from raw PDF text to a deployed AI-powered web app.*
