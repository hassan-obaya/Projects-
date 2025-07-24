# 📄 LLM-Powered Question Answering from Documents

A powerful Python project that allows you to **upload a PDF or TXT file**, ask any question related to its content, and get answers from a **state-of-the-art Language Model (LLM)**. If the answer is not present in the document, the system will politely let you know.

---

##  Features

- ✅ Supports **PDF** and **TXT** documents
- ✅ Uses **Sentence Transformers** for semantic chunking
- ✅ Uses **FAISS** for fast vector-based retrieval
- ✅ Powered by **Mistral-7B-Instruct** from Hugging Face
- ✅ Avoids hallucinations: detects out-of-context questions
- ✅ Clean, modular, and easy-to-extend Python code

---

##  Project Structure

```bash
.
├── Code.py     # Main QA script
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation


---

##  How It Works

1. **Document Ingestion**

   * Extracts text from a PDF or TXT file.
2. **Chunking**

   * Splits text into manageable chunks (\~200 words each).
3. **Embedding + Indexing**

   * Converts chunks into dense vectors using `sentence-transformers`.
   * Indexes them using `faiss` for efficient retrieval.
4. **Question Answering**

   * When a user asks a question:

     * Retrieves top relevant chunks.
     * Sends them + the question as a prompt to the LLM.
     * Returns the most relevant answer.
   * If the answer is not in the context, a fallback response is triggered.

---

##  Getting Started

### 1. Install Dependencies

```bash
pip install sentence-transformers pymupdf faiss-cpu transformers
```

Or use the `!pip install ...` cell if you're on **Kaggle Notebooks**.

---

### 2. Authenticate with Hugging Face (if needed)

```python
from huggingface_hub import login
login()  # Or login(token="your_hf_token")
```

---

### 3. Run the Script

Update the file path and run:

```python
pdf_path = "/kaggle/input/.../your_file.pdf"
text = extract_text_from_pdf(pdf_path)
```

Then ask:

```python
question = "What are the main topics discussed?"
print(answer_question(question, index, chunk_list))
```

---

##  Dependencies

* `sentence-transformers`
* `transformers`
* `faiss-cpu`
* `PyMuPDF` (imported as `fitz`)
* `huggingface_hub`

---

Out-of-context question?

```text
Sorry, I can't answer that based on the provided document.
```

---

## 📄 License

This project is open-source and free to use for educational or research purposes. Please cite Hugging Face models used if you publish work based on this.

---

## 🙋‍♂️ Author

By Hassan Obaia

