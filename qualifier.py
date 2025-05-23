import os
import json
from glob import glob
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

load_dotenv()
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL    = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
CHAT_MODEL     = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
CHROMA_PERSIST = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectordb   = Chroma(persist_directory=CHROMA_PERSIST, embedding_function=embeddings)
llm        = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL)

def load_company_profile():
    resp = vectordb.get(ids=["COMPANY_PROFILE"], include=["documents"])
    docs = resp.get("documents", [])
    if not docs or not docs[0]:
        raise ValueError("Company profile not found in Chroma store.")
    return docs[0]

def evaluate_each_proposal():
    company_text = load_company_profile()
    results = {}

    for pdf_path in glob("grant_docs/*.pdf"):
        filename = os.path.basename(pdf_path)
        chunks = vectordb.similarity_search(
            "company profile and grant proposals",
            k=5,
            filter={"source": filename}
        )
        if not chunks:
            results[filename] = {
                "qualifies": "no",
                "reason": "No content indexed for this proposal."
            }
            continue

        proposal_text = "\n\n".join(c.page_content for c in chunks)
        prompt = (
            "SYSTEM:\n"
            "You are an expert grant-qualification assistant. Given a company profile and exactly one grant proposal, "
            "for the proposal answer: Does the company qualify? Output JSON with keys "
            "\"qualifies\":\"yes\"/\"no\",\"reason\":\"...\".\n\n"
            f"CONTEXT:\nCompany Profile:\n{company_text}\n\n"
            f"Proposal ({filename}):\n{proposal_text}\n\n"
            "ASSISTANT (JSON only):"
        )

        output = llm.invoke(prompt)

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = {
                "qualifies": None,
                "reason": output.strip()
            }

        results[filename] = parsed

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    evaluate_each_proposal()
