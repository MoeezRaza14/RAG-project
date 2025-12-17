from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from get_embeddings import get_embedding_function
from dotenv import load_dotenv
import os
from mistralai import Mistral

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHROMA_PATH = "chroma"
EMBEDDING_FN = get_embedding_function()
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=EMBEDDING_FN,
    )
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = response.choices[0].message.content
    sources = [doc.metadata.get("id") for doc, _ in results]
    print(f"Response: {response_text}\nSources: {sources}")
    return response_text
