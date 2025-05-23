import os
import json
from uuid import uuid4
from glob import glob
from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL    = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
CHROMA_PERSIST = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectordb   = Chroma(persist_directory=CHROMA_PERSIST, embedding_function=embeddings)

def clean_metadata(meta):
    result = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = json.dumps(value)
    return result

def ingest_company_profile(path):
    print("Loading company profile...")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
    try:
        with open(path, encoding="utf-8") as f:
            company = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return

    pid = "COMPANY_PROFILE"
    resp = vectordb.get(ids=[pid], include=["metadatas"])
    metas = resp.get("metadatas", [])
    if metas and metas[0]:
        print("Company profile already indexed.")
        return

    print("Indexing company profile...")
    vectordb.add_texts(
        texts=[company["description"]],
        metadatas=[clean_metadata(company)],
        ids=[pid]
    )

def ingest_pdf(path):
    filename = os.path.basename(path)
    print(f"Processing {filename}...")
    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    print(f"Embedding {len(chunks)} chunks...")
    ids = [str(uuid4()) for _ in chunks]
    for chunk, cid in tqdm(zip(chunks, ids), total=len(chunks), desc="Embedding"):
        vectordb.add_texts(
            texts=[chunk],
            metadatas=[{"source": filename}],
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
    ingest_company_profile("company_info.json")

    pdf_files = glob("grant_docs/*.pdf")
    if not pdf_files:
        print("No PDFs found in grant_docs.")
    else:
        for pdf in pdf_files:
            ingest_pdf(pdf)

    persist_store()
    print("Embedding complete.")
