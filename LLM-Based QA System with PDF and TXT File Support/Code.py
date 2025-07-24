# Uncomment and run the line below if you're not using Kaggle or already installed
# !pip install sentence-transformers pymupdf faiss-cpu transformers

# -----------------------------------------------
# Step 1: Import necessary libraries
# -----------------------------------------------

from huggingface_hub import login
import fitz                         # PyMuPDF for PDF text extraction
import numpy as np
import faiss                        # Facebook AI Similarity Search (vector index)
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------------------------
# Step 2: Hugging Face authentication (for gated models if needed)
# -----------------------------------------------

login()  # You can pass your token here like: login(token="your_hf_token")

# -----------------------------------------------
# Step 3: Initialize embedding & generation models
# -----------------------------------------------

# Lightweight sentence embedding model (384-dim)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Powerful instruction-following LLM (Mistral 7B)
llm = pipeline(
    task="text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",             # auto-select GPU/CPU
    torch_dtype="auto",
    max_new_tokens=300             # max tokens for generated answer
)

# -----------------------------------------------
# Step 4: Text extraction utilities
# -----------------------------------------------

def extract_text_from_pdf(pdf_path):
    """Extract raw text from all pages of a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_txt(txt_path):
    """Read all text from a plain text file."""
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

# -----------------------------------------------
# Step 5: Preprocessing - Split text into chunks
# -----------------------------------------------

def chunk_text(text, max_words=200):
    """Split the full text into chunks of fixed word size for retrieval."""
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# -----------------------------------------------
# Step 6: Create FAISS index of embedded chunks
# -----------------------------------------------

def create_vector_store(chunks):
    """Embed each chunk and index them using FAISS for similarity search."""
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)     # L2 distance metric
    index.add(np.array(embeddings))    # Add embeddings to index
    return index, chunks

# -----------------------------------------------
# Step 7: QA - Retrieve + generate answer
# -----------------------------------------------

def answer_question(question, index, chunks, top_k=3):
    """Answer the user’s question using top-k relevant chunks + LLM."""
    q_embedding = embedder.encode([question])
    _, retrieved_indices = index.search(np.array(q_embedding), top_k)

    # Concatenate top-k retrieved chunks as context
    context = "\n\n".join(chunks[i] for i in retrieved_indices[0])

    # Construct the prompt for the LLM
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Generate the answer from the LLM
    result = llm(prompt)[0]["generated_text"]
    answer = result.split("Answer:")[-1].strip()

    # Post-processing: Handle unknown answers
    low = answer.lower()
    if any(phrase in low for phrase in ["i don't know", "not mentioned", "can't answer"]):
        return "Sorry, I can't answer that based on the provided document."
    
    return answer

# -----------------------------------------------
# Step 8: Load your document and process it
# -----------------------------------------------

# You can change this to a .txt path if needed
pdf_path = "/kaggle/input/llms-interviews-questions/LLMs interviews questions.pdf"

# 1. Extract full text from PDF
text = extract_text_from_pdf(pdf_path)

# 2. Chunk text into smaller blocks for retrieval
chunks = chunk_text(text)

# 3. Create vector index for retrieval
index, chunk_list = create_vector_store(chunks)
print(f"Document processed into {len(chunks)} chunks.")

# -----------------------------------------------
# Step 9: Ask your question!
# -----------------------------------------------

question = "Who is Trumb?"  # Example question
print("Question:", question)
print("Answer:", answer_question(question, index, chunk_list))
