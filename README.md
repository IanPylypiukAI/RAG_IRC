# Tax Chatbot AI — Ingestion Script

This repository contains scripts for building a **Retrieval-Augmented Generation (RAG)** system for U.S. tax law using **LLaMA3** and **BGE-M3** embeddings.

Currently, the repository includes:

- `ingestion.py`: A script to download, process, and embed the U.S. Internal Revenue Code PDF into a Qdrant vector store.

---

## Features of `ingestion.py`

1. **Download and extract the US Internal Revenue Code**  
   Downloads the PDF ZIP from the official source and extracts it.

2. **Read and parse PDF documents**  
   Uses `PyMuPDFReader` to read PDFs and split the content into nodes for embeddings.

3. **HuggingFace and Groq integration**  
   Authenticates with Hugging Face for embedding models and Groq for LLM usage.

4. **Create embeddings with BGE-M3**  
   Generates vector embeddings for each text chunk.

5. **Store embeddings in Qdrant**  
   Pushes all embeddings to a Qdrant collection for fast retrieval.

---

## Requirements

- Python 3.10+  
- GPU recommended (CUDA or Apple MPS) for faster embedding generation  
- Secrets for Colab or environment variables:
  - `HUGGINGFACE_TAX_APP_SECRET`
  - `QDRANT_CLIENT_API_KEY`
  - `QDRANT_CLIENT_URL`
