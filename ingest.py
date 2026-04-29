import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

# Configuration
PDF_FOLDER = "data/"
QDRANT_PATH =  "qdrant_storage"
COLLECTION_NAME = "banking_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COHERE_MODEL = "embed-english-v3.0"

def check_pdfs_exist() :
    if not os.path.exists(PDF_FOLDER) :
        print(f"Folder '{PDF_FOLDER}' does not exist.")
        return False

    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if not pdfs :
        print(f"No PDF files found in '{PDF_FOLDER}/'")
        return False

    print(f"Found {len(pdfs)} banking documents")
    for pdf in pdfs :
        print(f"  {pdf}")
    return  True

def load_documents() :
    print("\n Loading PDFs....")
    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    docs = loader.load()
    print(f"  Loaded {len(docs)} pages in total.")
    return docs

def split_into_chunks(docs) :
    print("\n Splitting into chunks....")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE ,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Add rich metadata
    for i, chunk in enumerate(chunks):
        raw_source = chunk.metadata.get("source", "unknown")
        filename = os.path.basename(raw_source)
        chunk.metadata["filename"] = filename
        chunk.metadata["chunk_index"] = i
        chunk.metadata["source"] = raw_source
        print(f"  Created {len(chunks)} chunks.")
        return chunks

def create_qdrant_vectorstore(chunks):
    # Delete old storage if exists
    if os.path.exists(QDRANT_PATH):
        print("Removing old Qdrant storage....")
        shutil.rmtree(QDRANT_PATH)

    # Check Cohere API key
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError(
            "COHERE_API_KEY not found. "
            "Add it to your .env file."
        )

    print("Initialising Cohere embeddings....")
    embeddings = CohereEmbeddings(
        model=COHERE_MODEL,
        cohere_api_key=cohere_key,
    )

    print("\nEmbedding chunks and storing in Qdrant...")
    print("  This may take 2-4 minutes....")

    # QdrantVectorStore.from_documents handles everything
    # collection creation + embedding + storing
    # No need to manually create client or collection
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True,    # recreates collection if exists
    )

    print(f"  Saved {len(chunks)} chunks to Qdrant.")
    return vectorstore


def main() :
    print("Finsight AI  - Banking Document Ingestion ")

    if not check_pdfs_exist():
        return

    docs = load_documents()
    chunks = split_into_chunks(docs)
    create_qdrant_vectorstore(chunks)

    print("Ingestion complete!")

if __name__ == "__main__" :
    main()