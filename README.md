# RAG (Retrieval-Augmented Generation) Project with Mistral and Chroma

This repository implements a **Retrieval-Augmented Generation (RAG)** system using:

- **LangChain** for vector store and document handling  
- **Chroma** for storing embeddings  
- **Qwen3 Embeddings** from HuggingFace for text embeddings  
- **Mistral API** for LLM-based query answering and evaluation  

The system allows you to load documents (PDFs), embed them, store the embeddings in Chroma, and then query them interactively with runtime retrieval and generation.

---

## Repository Structure

| File | Description |
|------|-------------|
| `get_embeddings.py` | Returns the HuggingFace embeddings function (Qwen3). |
| `populate_db.py` | Loads PDF documents, splits them into chunks, generates embeddings, and saves them to Chroma. |
| `query_data.py` | Implements the RAG query pipeline using stored embeddings and Mistral API. |
| `rag.py` | Interactive runtime for querying the RAG system and optionally evaluating answers. |

---

## Features

1. **Document Embedding & Storage**: PDFs are split into chunks and stored as embeddings in Chroma.  
2. **Retrieval-Augmented Generation**: Uses Chroma embeddings to retrieve relevant context, then queries Mistral for generation.  
3. **Interactive Queries**: Users can type questions at runtime and get RAG-based answers.  
4. **Answer Evaluation**: Optionally evaluate answers against expected responses using Mistral.  
5. **Optimized Embeddings**: Embeddings are created once and reused at runtime for faster query performance.  

---

## Installation

1. **Clone the repository**:

```bash
git clone <repo_url>
cd <repo_folder>

2. **Create a Virtual Environment**
python -m venv vsvenv
source vsvenv/Scripts/activate  # Windows
# or
source vsvenv/bin/activate      # Linux/Mac

3. **Install Dependencies**
pip install -r requirements.txt

4. **Create a .env file for API key**
MODEL_API_KEY = <model_api_key>

5. **Run the code and prompt the LLM**
python rag.py
