from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embeddings import get_embedding_function
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "C:\\F\\ms_vs_code_files\\Dataset"

def main(reset_db: bool = False):
    """
    Call this function directly with reset_db=True
    if you want to clear the existing database programmatically.
    """
    if reset_db:
        print("Clearing Database")
        clear_database()

    # Load & split documents
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    """Load all PDFs from the data directory."""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    """Split long docs into short chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    """Add only new documents (by ID) into the Chroma DB."""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [
        chunk for chunk in chunks_with_ids
        if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    """
    Generate stable chunk IDs like "file.pdf:page:chunk_index".
    """
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
    """
    If you want to clear the DB, manually implement logic here
    (e.g., delete Chroma collections via Chroma API instead of os/shutil).
    """
    # NOTE: You can delete via Chroma API instead of file system if supported:
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    # db.delete(ids=existing_ids_list)   # Example: remove specific docs
    # db.persist()
    print("Database clear not implemented (no os/shutil).")
    # Optionally print instructions for manual clear:
    print("Manually delete the 'chroma' folder to reset DB.")

if __name__ == "__main__":
    # Example calls:
    # main(reset_db=True)  # clear then populate
    main(reset_db=False)