import os
import json
from glob import glob
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")

# Initialize LLM
llm = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL)


def extract_text(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    reader = PdfReader(pdf_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_sections(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split the call document into overlapping text sections."""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def rank_from_score(score: float) -> str:
    if score <= 4:
        return "Poor"
    elif score <= 7:
        return "Good"
    else:
        return "Excellent"


def evaluate_proposals() -> dict:
    """Evaluate each company proposal against its corresponding government call doc, by section."""
    call_docs = glob("grant_docs/*.pdf")
    proposals = glob("company_proposals/*.pdf")
    results = {}

    for prop_path in proposals:
        prop_name = os.path.basename(prop_path)
        prop_base = os.path.splitext(prop_name)[0].replace("_proposal", "")
        # Find matching call document by prefix
        matching = [c for c in call_docs if prop_base in os.path.basename(c)]
        if not matching:
            print(f"No matching call doc for proposal {prop_name}, skipping.")
            continue
        call_path = matching[0]
        print(f"Evaluating {prop_name} against {os.path.basename(call_path)}...")

        # Extract texts
        call_text = extract_text(call_path)
        prop_text = extract_text(prop_path)

        # Split call into sections
        sections = chunk_sections(call_text)
        section_results = []

        # Evaluate each section
        for idx, sec in enumerate(sections, start=1):
            section_label = f"Section {idx}"
            prompt = (
                "SYSTEM:\n"
                "You are an expert grant qualification assistant. "
                "Given this specific section of the government call for proposals and the full company proposal, "
                "evaluate how well the proposal addresses the section objectives. "
                "Provide a score from 0 to 10, concise reasoning, and actionable suggestions for improvement.\n\n"
                f"CALL SECTION ({section_label}):\n{sec}\n\n"
                f"COMPANY PROPOSAL:\n{prop_text}\n\n"
                "ASSISTANT (JSON only, format: {\"section\": str, \"score\": float, \"reason\": str, \"suggestions\": str}):"
            )
            output = llm.invoke(prompt)
            try:
                res = json.loads(output)
                score = float(res.get("score", 0))
                reason = res.get("reason", "")
                suggestions = res.get("suggestions", "")
            except json.JSONDecodeError:
                score = 0
                reason = output.strip()
                suggestions = ""

            section_results.append({
                "section": section_label,
                "score": score,
                "reason": reason,
                "suggestions": suggestions
            })

        # Aggregate overall
        avg_score = sum(item["score"] for item in section_results) / max(len(section_results), 1)
        overall_rank = rank_from_score(avg_score)

        results[prop_name] = {
            "overall_score": avg_score,
            "overall_rank": overall_rank,
            "sections": section_results
        }

    return results


def generate_pdf(results: dict, output_path: str = "./grant_evaluation_report.pdf"):
    """Generate a PDF report with summary and detailed per-section feedback."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=60,
        bottomMargin=40
    )
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Grant Proposals Evaluation Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Summary Table
    elements.append(Paragraph("Summary of Proposal Scores", styles["Heading2"]))
    elements.append(Spacer(1, 12))
    summary_data = [["Proposal", "Score", "Rank"]]
    for name, res in results.items():
        summary_data.append([name, f"{res['overall_score']:.1f}", res['overall_rank']])

    table = Table(summary_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    elements.append(table)
    elements.append(PageBreak())

    # Detailed Per-Section Feedback
    for name, res in results.items():
        elements.append(Paragraph(f"Proposal: {name}", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"<b>Overall Score:</b> {res['overall_score']:.1f}", styles["BodyText"]))
        elements.append(Paragraph(f"<b>Overall Rank:</b> {res['overall_rank']}", styles["BodyText"]))
        elements.append(Spacer(1, 12))

        for sec in res['sections']:
            elements.append(Paragraph(f"{sec['section']}", styles["Heading3"]))
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(f"<b>Score:</b> {sec['score']:.1f}", styles["BodyText"]))
            elements.append(Paragraph(f"<b>Reason:</b> {sec['reason']}", styles["BodyText"]))
            elements.append(Paragraph(f"<b>Suggestions:</b> {sec['suggestions']}", styles["BodyText"]))
            elements.append(Spacer(1, 12))

        elements.append(PageBreak())

    doc.build(elements)
    print(f"PDF report generated at: {output_path}")


if __name__ == "__main__":
    # Ensure input directories exist
    os.makedirs("grant_docs", exist_ok=True)
    os.makedirs("company_proposals", exist_ok=True)

    # Evaluate proposals and build report
    results = evaluate_proposals()
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    generate_pdf(results)