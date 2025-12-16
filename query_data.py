from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from get_embeddings import get_embedding_function
from dotenv import load_dotenv
from mistralai import Mistral
import os
CHROMA_PATH = "chroma"

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Load the vector database
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve nearest chunks
    results = db.similarity_search_with_score(query_text, k=5)

    # Build context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Format prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Call the Mistral API
    # Ensure you set your MISTRAL_API_KEY environment variable
    client = Mistral(api_key=MISTRAL_API_KEY)

    # Chat completion call
    response = client.chat.complete(
        model="mistral-large-latest",  # or any available Mistral chat model
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract text from the API response
    # (Mistral API returns choices/messages similar to other LLM APIs)
    response_text = response.choices[0].message.content

    # Collect sources
    sources = [doc.metadata.get("id", None) for doc, _ in results]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text