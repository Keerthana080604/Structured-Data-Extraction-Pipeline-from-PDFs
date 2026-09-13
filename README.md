# 📄 Structured Data Extraction from PDFs using RAG

Extract clean, validated, structured data from unstructured PDF documents powered by a Retrieval-Augmented Generation (RAG) pipeline, LLM-based schema extraction, and a custom-built UI. Fully containerized and deployable with Docker.


---

## 🚀 Overview

Traditional LLM prompting on long, unstructured PDFs is prone to hallucination and inconsistent output formats. This project solves that by combining **retrieval-grounded generation** with **schema-validated extraction**, so the system pulls information *from* the source document rather than guessing and returns it in a predictable, structured format instead of free text.

Built end-to-end: ingestion → chunking → embedding → retrieval → schema-constrained extraction → custom UI → Docker deployment.

---

## ✨ Key Features

- **Grounded extraction, not hallucination** - Answers and extracted fields are tied back to actual chunks retrieved from the source document.
- **Schema-validated output** - Extracted data is validated against a defined schema (e.g., Pydantic models) instead of relying on raw LLM text, so downstream systems get consistent, structured data (e.g., tabular / JSON) every time.
- **Vector-based retrieval** - Documents are chunked and embedded into a vector store for fast, relevant context retrieval at query time.
- **Custom UI** -  Built a UI from scratch (beyond any reference implementation) for uploading PDFs, triggering extraction, and reviewing structured results in a clean, usable interface.
- **Containerized deployment** - Packaged with Docker for reproducible setup and one-command deployment, independent of local environment quirks.

---

## 🏗️ Architecture

```
PDF Upload
   │
   ▼
Text Extraction & Chunking
   │
   ▼
Embedding Generation ──► Vector Store (Chroma)
   │
   ▼
Retrieval (top-k relevant chunks)
   │
   ▼
LLM Extraction (schema-constrained)
   │
   ▼
Validated Structured Output ──► UI Display / Export
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| LLM / Embeddings | Gemini API |
| RAG Framework | LangChain |
| Vector Store | Chroma |
| Schema Validation | Pydantic |
| UI | Streamlit |
| Deployment | Docker |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Docker (for containerized run)
- A Gemini API key



### Run Locally
```bash
python app/main.py
```

### Run with Docker
```bash
docker build -t pdf-rag-extractor .
docker run -p 8501:8501 --env-file .env pdf-rag-extractor
```

---

## 🎯 Usage

1. Launch the app (locally or via Docker).
2. Upload a PDF document through the UI.
3. The pipeline chunks, embeds, and indexes the document automatically.
4. Extracted structured data is displayed in the UI and can be exported (e.g., as JSON/CSV).

---

## 📊 Example Output

<!-- Add a before/after or sample output screenshot/table here -->
```json
{
  "field_1": "extracted value",
  "field_2": "extracted value",
  "source_chunk": "reference to originating text"
}
```

---


## 🔮 Future Improvements

- [ ] Support for batch PDF processing
- [ ] Add authentication for multi-user deployment
- [ ] Expand schema library for different document types
- [ ] CI/CD pipeline for automated Docker builds

---


## 🙋 About

Built by **Keerthana Bammidi** - Computer Science undergraduate exploring applied GenAI, RAG systems, and full-stack development.

[Email](mailto:bammidikeerthana8@gmail.com)
