from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embeddings import get_embedding_function
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "dataset" #folder path where the datasets are placed like C:\...\dataset

def main(reset_db: bool = False):
    if reset_db:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # 🔹 Use embedding function instance
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing = db.get(include=[])
    existing_ids = set(existing["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [
        c for c in chunks_with_ids if c.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        ids = [c.metadata["id"] for c in new_chunks]
        db.add_documents(new_chunks, ids=ids)
        # persistence happens automatically
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    print("Database clear not implemented (manual deletion recommended).")
    print("Manually delete the 'chroma' folder to reset the DB.")

if __name__ == "__main__":

    main(reset_db=False)
