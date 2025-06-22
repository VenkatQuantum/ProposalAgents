import os
import json
from uuid import uuid4
from glob import glob
from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Load environment
load_dotenv()
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL    = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
CHROMA_PERSIST = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectordb   = Chroma(persist_directory=CHROMA_PERSIST, embedding_function=embeddings)


def clean_metadata(meta):
    """Ensure metadata values are JSON-serializable."""
    result = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = json.dumps(value)
    return result


def ingest_pdfs(folder: str, doc_type: str):
    """
    Ingest all PDFs in a folder into the vector store.
    doc_type should be 'call' or 'proposal'.
    """
    if not os.path.isdir(folder):
        print(f"Warning: {folder} not found.")
        return
    pdf_files = glob(os.path.join(folder, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {folder}.")
        return

    for path in pdf_files:
        filename = os.path.basename(path)
        print(f"Processing {doc_type}: {filename}...")
        reader = PdfReader(path)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

        # Split into chunks
        from langchain_text_splitters import CharacterTextSplitter
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        # Embed
        print(f"Embedding {len(chunks)} chunks...")
        ids = [str(uuid4()) for _ in chunks]
        for chunk, cid in tqdm(zip(chunks, ids), total=len(chunks), desc="Embedding chunks"):
            vectordb.add_texts(
                texts=[chunk],
                metadatas=[clean_metadata({
                    "source": filename,
                    "type": doc_type
                })],
                ids=[cid]
            )


def persist_store():
    client = getattr(vectordb, "_client", None) or getattr(vectordb, "client", None)
    if client and hasattr(client, "persist"):
        client.persist()
        print("Vector store persisted.")
    else:
        print("Warning: persistence not available; data may be in-memory only.")


if __name__ == "__main__":
    os.makedirs(CHROMA_PERSIST, exist_ok=True)
    # Ingest government calls for proposals
    ingest_pdfs("grant_docs", doc_type="call")
    # Ingest company-written proposals
    ingest_pdfs("company_proposals", doc_type="proposal")
    persist_store()
    print("Embedding complete.")