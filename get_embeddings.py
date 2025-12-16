from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    embeddings = HuggingFaceEmbeddings
    model_name = "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs= {"device": "cpu"}
    return embeddings