import os
import json
from glob import glob
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from tqdm import tqdm

# Load environment variables from .env
load_dotenv()
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL    = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
CHAT_MODEL     = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
CHROMA_PERSIST = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")

# Initialize the embeddings, vector store and language model
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectordb   = Chroma(persist_directory=CHROMA_PERSIST, embedding_function=embeddings)
llm        = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL)

# Load company profile from the vector store
def load_company_profile():
    print("Loading company profile...")
    resp = vectordb.get(ids=["COMPANY_PROFILE"], include=["documents"])
    docs = resp.get("documents", [])
    if not docs or not docs[0]:
        raise ValueError("Company profile not found in Chroma store.")
    return docs[0]

# Evaluate proposals
def evaluate_each_proposal():
    # Load the company profile
    company_text = load_company_profile()
    results = {}

    # Loop through each PDF file in the grant_docs folder
    for pdf_path in glob("grant_docs/*.pdf"):
        filename = os.path.basename(pdf_path)
        print(f"Processing {filename}...")
        
        # Retrieve matching chunks from the vector store (based on the company profile)
        chunks = vectordb.similarity_search(
            query="company profile and grant proposals",
            k=5,
            filter={"source": filename}
        )
        
        # If no matching chunks are found, mark the proposal as not qualifying
        if not chunks:
            results[filename] = {
                "qualifies": "no",
                "reason": "No content indexed for this proposal."
            }
            continue
        
        # Combine the matched chunks of text from the grant proposal
        proposal_text = "\n\n".join(c.page_content for c in chunks)
        
        # Create a better prompt for the LLM to evaluate the proposal
        prompt = (
            "SYSTEM:\n"
            "You are an expert grant-qualification assistant. Given a company profile and exactly one grant proposal, "
            "evaluate the proposal based on its alignment with the company's profile. For each section of the proposal, "
            "assign a score from 0 to 10 based on how well it matches the company profile, and provide reasoning for the score.\n\n"
            "Output the results in the following format:\n"
            "[\n"
            "  {\"section\": \"section_name\", \"score\": score, \"reason\": \"reasoning for score\"},\n"
            "  ...\n"
            "]\n\n"
            f"CONTEXT:\nCompany Profile:\n{company_text}\n\n"
            f"Proposal ({filename}):\n{proposal_text}\n\n"
            "ASSISTANT (JSON only):"
        )
        
        # Call the LLM to get the response
        output = llm.invoke(prompt)

        # Try to parse the LLM's output as JSON
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            parsed = {
                "qualifies": None,
                "reason": output.strip()
            }

        # Store the result in the dictionary
        results[filename] = parsed

    # Print the results as a formatted JSON output
    print(json.dumps(results, indent=2))

# Main function to start the evaluation process
if __name__ == "__main__":
    evaluate_each_proposal()
