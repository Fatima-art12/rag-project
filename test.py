import os
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Extract text from the resume PDF
reader = PdfReader("resume.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

# Step 3: Split text into chunks
def chunk_text(text, chunk_size=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

chunks = chunk_text(full_text)

# Step 4: Create embeddings for all chunks
chunk_embeddings = model.encode(chunks)

# Step 5: Ask a question and find the most similar chunk
question = "What programming languages does this person know?"
question_embedding = model.encode(question)

scores = cos_sim(question_embedding, chunk_embeddings)[0]
best_chunk_index = scores.argmax().item()
best_chunk = chunks[best_chunk_index]

print("Question:", question)
print("Retrieved chunk:", best_chunk)

# Step 6: Send the question + retrieved chunk to Gemini AI to generate an answer
client = genai.Client(api_key=api_key)

prompt = f"""Answer the question based only on the context below.

Context:
{best_chunk}

Question: {question}
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("\n--- AI Answer ---")
print(response.text)