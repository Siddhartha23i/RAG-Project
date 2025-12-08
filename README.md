# 🏢 TechTitan RAG: IT Giants Knowledge Base

**TechTitan RAG** is a Retrieval-Augmented Generation (RAG) system designed to provide accurate, context-aware answers about the world's top 5 Information Technology companies (Google, Microsoft, Apple, Amazon, Meta). 

This project implements a full end-to-end pipeline, including a **Data Injection/Ingestion System** to process raw company profiles and a **Retrieval System** to answer natural language queries based on that structured data.

-----

## 🚀 Project Overview

Standard LLMs often hallucinate specifics or lack the most up-to-date corporate structures. This system solves that by grounding answers in a curated "knowledge base" of structured text files.

### Key Features
* **Knowledge Base:** Specialized detailed profiles for Google, Microsoft, Apple, Amazon, and Meta.
* **Injection Pipeline:** Automates the loading, chunking, and vectorization of text data.
* **Semantic Search:** Uses vector embeddings to find the most relevant paragraphs for a user's question.
* **Contextual Q&A:** Generates precise answers citing facts from the injected documents.

-----

## 🛠️ Architecture

The project consists of two main workflows:

### 1. The Injection Pipeline (Ingestion)
1.  **Load:** Reads structured `.txt` files (e.g., `GOOGLE_KNOWLEDGE_BASE.txt`) from the `data/` directory.
2.  **Chunk:** Splits long documents into manageable segments (e.g., 500-1000 characters) with overlap to preserve context.
3.  **Embed:** Converts text chunks into vector embeddings using an Embedding Model (e.g., OpenAI, HuggingFace).
4.  **Store:** Saves vectors into a Vector Database (e.g., ChromaDB, FAISS, Pinecone).

### 2. The Retrieval System (Inference)
1.  **Query:** User asks a question (e.g., *"What companies did Google acquire in 2014?"*).
2.  **Retrieve:** System finds the top $k$ most similar chunks from the Vector DB.
3.  **Generate:** An LLM receives the question + relevant chunks and synthesizes an answer.

-----

## 📂 Directory Structure

```bash
TechTitan-RAG/
├── data/                      # Place your company .txt files here
│   ├── google_profile.txt
│   ├── microsoft_profile.txt
│   └── ...
├── src/
│   ├── ingestion.py           # Script to chunk and load data to Vector DB
│   ├── retrieval.py           # RAG logic (Search + LLM generation)
│   └── vector_store.py        # DB connection/setup logic
├── main.py                    # Entry point for the CLI/App
├── requirements.txt           # Python dependencies
├── .env                       # API Keys (OpenAI, HuggingFace, etc.)
└── README.md                  # This file
