import os
import json
from glob import glob
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from tqdm import tqdm
<<<<<<< HEAD
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

=======

# Load environment variables from .env
>>>>>>> a4a3b5fd5877032cbb813ebcc743909885264ef3
load_dotenv()
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL    = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
CHAT_MODEL     = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
CHROMA_PERSIST = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")

<<<<<<< HEAD
# Initialize embeddings, vector store, and LLM
=======
# Initialize the embeddings, vector store and language model
>>>>>>> a4a3b5fd5877032cbb813ebcc743909885264ef3
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectordb   = Chroma(persist_directory=CHROMA_PERSIST, embedding_function=embeddings)
llm        = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL)

<<<<<<< HEAD

def load_company_profile():
    """
    Load the company profile document from Chroma by ID.
    """
=======
# Load company profile from the vector store
def load_company_profile():
    print("Loading company profile...")
>>>>>>> a4a3b5fd5877032cbb813ebcc743909885264ef3
    resp = vectordb.get(ids=["COMPANY_PROFILE"], include=["documents"])
    docs = resp.get("documents", [])
    if not docs or not docs[0]:
        raise ValueError("Company profile not found in Chroma store.")
    return docs[0]

<<<<<<< HEAD

def chunk_proposal_text(pdf_path: str):
    """
    Load a PDF and split its text into overlapping chunks.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    full_text = "\n".join(doc.page_content for doc in documents)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(full_text)


def evaluate_each_proposal():
    """
    For each PDF, split into chunks, compare each to the company profile,
    and return per-chunk scores and reasons.
    """
    company_text = load_company_profile()
    results = {}

    for pdf_path in tqdm(glob("grant_docs/*.pdf"), desc="Evaluating proposals"):
        filename = os.path.basename(pdf_path)
        chunks = chunk_proposal_text(pdf_path)

        if not chunks:
            results[filename] = [{"section": "Overall", "score": 0, "reason": "No text found in proposal."}]
            continue

        section_evaluations = []
        for idx, chunk in enumerate(chunks, start=1):
            prompt = (
                "SYSTEM:\n"
                "You are a grant assessment assistant. Compare the company's profile to this proposal section. "
                "Assess the company's ability to execute these section requirements based on its strengths. "
                "Assign a score from 0 to 10 and provide reasoning. Return a JSON array of objects with sections 'section', 'score', and 'reason'.\n\n"
                f"Company Profile:\n{company_text}\n\n"
                f"Proposal Section (Chunk {idx}):\n{chunk}\n\n"
                "ASSISTANT (JSON only):"
            )
            output = llm.invoke(prompt)
            try:
                sec_results = json.loads(output)
            except json.JSONDecodeError:
                sec_results = [{"section": f"Chunk {idx}", "score": 0, "reason": output.strip()}]

            for item in sec_results:
                if 'section' not in item:
                    item['section'] = f"Chunk {idx}"
                section_evaluations.append(item)

        results[filename] = section_evaluations

    return results


def generate_pdf(results: dict, output_path: str = "./grant_evaluation_report.pdf"):
    """
    Generate a structured PDF report with readable text formatting and colored scores.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=60, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['BodyText']
    normal_style.fontSize = 10
    normal_style.leading = 14

    elements = []
    # Title
    elements.append(Paragraph("Grant Proposals Evaluation Report", title_style))
    elements.append(Spacer(1, 24))

    # Overall Summary
    elements.append(Paragraph("Overall Summary", heading_style))
    elements.append(Spacer(1, 12))
    for filename, data in results.items():
        avg_score = sum(item.get('score', 0) for item in data) / max(len(data), 1)
        if avg_score >= 8:
            summary_text = "Strong ability to execute; excellent fit."
        elif avg_score >= 5:
            summary_text = "Moderate ability; some capability gaps."
        else:
            summary_text = "Limited ability; major gaps."
        elements.append(Paragraph(
            f"<b>{filename}</b>: Average Score: <b>{avg_score:.1f}</b> - {summary_text}",
            normal_style
        ))
        elements.append(Spacer(1, 6))
    elements.append(PageBreak())

    # Detailed Sections per Proposal
    for filename, data in results.items():
        elements.append(Paragraph(f"Proposal: {filename}", heading_style))
        elements.append(Spacer(1, 12))
        for item in data:
            section = item.get('section', '')
            score = item.get('score', 0)
            reason = item.get('reason', '').replace('\n', '<br/>')
            if score < 5:
                color = "#FF0000"
            elif score < 8:
                color = "#FFA500"
            else:
                color = "#008000"
            elements.append(Paragraph(f"<b>Section:</b> {section}", subheading_style))
            elements.append(Paragraph(f"<b>Score:</b> <font color='{color}'>{score}</font>", normal_style))
            elements.append(Paragraph(f"<b>Reason:</b> {reason}", normal_style))
            elements.append(Spacer(1, 12))
        elements.append(PageBreak())

    doc.build(elements)
    print(f"PDF report generated at: {output_path}")


if __name__ == "__main__":
    results = evaluate_each_proposal()
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results.json")
    generate_pdf(results, output_path="./grant_evaluation_report.pdf")
=======
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
>>>>>>> a4a3b5fd5877032cbb813ebcc743909885264ef3
