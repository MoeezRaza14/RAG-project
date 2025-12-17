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
| `get_embeddings.py` | Returns the HuggingFace embeddings function (Qwen3 0.6B embedding model). |
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
```
2. **Create a Virtual Environment**
```bash
python -m venv vsvenv
source vsvenv/Scripts/activate # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

5. **Create a .env file for API key**
```bash
MODEL_API_KEY = <model_api_key>
```

6. **Run the code and prompt the LLM**
```bash
python rag.py #run this command in the terminal to run the rag.py
```

## **You will get this type of output to ask the question from LLM. You can also use Streamlit or Chainlit for interactive UI for conversation.**
Interactive RAG (type 'exit' to quit)

Enter your question: **If a player uses all 7 tiles (1 blank tile and 6 others) in a single turn then how many points will he get?** #this is an example question. 

Running RAG for your query…………………………………!

Response: Based on the provided context:

1. The player scores the **regular points** for the word formed by the 7 tiles (including the blank tile, which has no score value but contributes to the word).   
2. If the word covers any **premium squares** (Double/Triple Letter or Word), those bonuses are applied **before** the 50-point bonus.
3. After calculating the regular score (including premium squares), the player receives a **50-point bonus** for using all 7 tiles in a single turn.

### Example Calculation:
- Suppose the player plays a 7-letter word (e.g., "EXAMPLE") with:
  - 1 blank tile (0 points) and 6 other tiles (e.g., E1, X8, A1, M3, P3, L1, E1).
  - The word covers a **Double Word Score** square.
- **Regular score**:
  - Sum of letter values: 1 (E) + 8 (X) + 1 (A) + 3 (M) + 3 (P) + 1 (L) + 1 (E) = **18 points**.
  - Double Word Score applies: 18 × 2 = **36 points**.
- **50-point bonus**: Added after premiums → **36 + 50 = 86 points total**.

### Final Answer:
The player will receive:
- The **regular score** for the word (including premium squares if applicable) **plus 50 points**.
- The exact total depends on the word's letter values and premium squares, but the **50-point bonus is always added last**.

### RAG Answer:
Based on the provided context:

1. The player scores the **regular points** for the word formed by the 7 tiles (including the blank tile, which has no score value but contributes to the word).   
2. If the word covers any **premium squares** (Double/Triple Letter or Word), those bonuses are applied **before** the 50-point bonus.
3. After calculating the regular score (including premium squares), the player receives a **50-point bonus** for using all 7 tiles in a single turn.

### Example Calculation:
- Suppose the player plays a 7-letter word (e.g., "EXAMPLE") with:
  - 1 blank tile (0 points) and 6 other tiles (e.g., E1, X8, A1, M3, P3, L1, E1).
  - The word covers a **Double Word Score** square.
- **Regular score**:
  - Sum of letter values: 1 (E) + 8 (X) + 1 (A) + 3 (M) + 3 (P) + 1 (L) + 1 (E) = **18 points**.
  - Double Word Score applies: 18 × 2 = **36 points**.
- **50-point bonus**: Added after premiums → **36 + 50 = 86 points total**.

### Final Answer:
The player will receive:
- The **regular score** for the word (including premium squares if applicable) **plus 50 points**.
- The exact total depends on the word's letter values and premium squares, but the **50-point bonus is always added last**.

Do you want to evaluate this answer? (y/n):
