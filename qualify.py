import os
import json
from glob import glob
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from langchain_ollama import OllamaLLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Load environment variables
load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
MAX_WORKERS = int(os.getenv("MAX_SECTION_WORKERS", 4))  # adjust concurrency
MAX_SECTIONS = int(os.getenv("MAX_SECTIONS", 20))  # limit sections evaluated per call

# Initialize LLM
llm = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_URL)


def extract_text(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    with PdfReader(pdf_path) as reader:
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_sections(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping sections using recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    sections = splitter.split_text(text)
    return sections[:MAX_SECTIONS]  # limit number of sections


def rank_from_score(score: float) -> str:
    if score <= 4:
        return "Poor"
    elif score <= 7:
        return "Good"
    else:
        return "Excellent"


def evaluate_section(call_section: str, prop_text: str, label: str) -> dict:
    """Invoke LLM to evaluate a single section."""
    prompt = (
        "SYSTEM:\n"
        "You are an expert grant qualification assistant. "
        "Evaluate how well the proposal addresses this section of the call. "
        "Return JSON: {\"section\": label, \"score\": float, \"reason\": str, \"suggestions\": str}.\n\n"
        f"CALL SECTION ({label}):\n{call_section}\n\n"
        f"COMPANY PROPOSAL:\n{prop_text}\n\n"
        "ASSISTANT (JSON only):"
    )
    output = llm.invoke(prompt)
    try:
        res = json.loads(output)
        res['score'] = float(res.get('score', 0))
        res.setdefault('section', label)
    except json.JSONDecodeError:
        res = {'section': label, 'score': 0.0, 'reason': output.strip(), 'suggestions': ''}
    return res


def evaluate_proposals() -> dict:
    """Evaluate proposals against matching call docs with concurrent section scoring."""
    call_docs = glob("grant_docs/*.pdf")
    proposals = glob("company_proposals/*.pdf")
    # Pre-extract and chunk call docs once
    calls = {}
    for call_path in call_docs:
        calls[call_path] = chunk_sections(extract_text(call_path))

    results = {}
    for prop_path in proposals:
        prop_name = os.path.basename(prop_path)
        prop_base = os.path.splitext(prop_name)[0].replace("_proposal", "")
        matches = [p for p in calls if prop_base in os.path.basename(p)]
        if not matches:
            print(f"No matching call doc for {prop_name}, skipping.")
            continue
        call_path = matches[0]
        print(f"Evaluating {prop_name}... (call: {os.path.basename(call_path)})")

        prop_text = extract_text(prop_path)
        sections = calls[call_path]

        section_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(evaluate_section, sec, prop_text, f"Section {i+1}"): i for i, sec in enumerate(sections)}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scoring {prop_name}"):
                section_results.append(future.result())

        # Aggregate
        avg_score = sum(item['score'] for item in section_results) / max(len(section_results), 1)
        overall_rank = rank_from_score(avg_score)
        results[prop_name] = {
            'overall_score': avg_score,
            'overall_rank': overall_rank,
            'sections': section_results
        }
    return results


def generate_pdf(results: dict, output_path: str = "./grant_evaluation_report.pdf"):
    """Generate a PDF report with summary and detailed per-section feedback."""
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    elems = []
    # Title & Summary
    elems.append(Paragraph("Grant Proposals Evaluation Report", styles['Title']))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("Summary of Proposal Scores", styles['Heading2']))
    data = [["Proposal", "Score", "Rank"]]
    for name, res in results.items():
        data.append([name, f"{res['overall_score']:.1f}", res['overall_rank']])
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')]))
    elems.extend([table, PageBreak()])

    # Details
    for name, res in results.items():
        elems.append(Paragraph(f"Proposal: {name}", styles['Heading2']))
        elems.append(Spacer(1,6))
        elems.append(Paragraph(f"<b>Overall Score:</b> {res['overall_score']:.1f}", styles['BodyText']))
        elems.append(Paragraph(f"<b>Overall Rank:</b> {res['overall_rank']}", styles['BodyText']))
        elems.append(Spacer(1,12))
        for sec in res['sections']:
            elems.append(Paragraph(sec['section'], styles['Heading3']))
            elems.append(Spacer(1,4))
            elems.append(Paragraph(f"<b>Score:</b> {sec['score']:.1f}", styles['BodyText']))
            elems.append(Paragraph(f"<b>Reason:</b> {sec.get('reason','')}", styles['BodyText']))
            elems.append(Paragraph(f"<b>Suggestions:</b> {sec.get('suggestions','')}", styles['BodyText']))
            elems.append(Spacer(1,12))
        elems.append(PageBreak())
    doc.build(elems)
    print(f"PDF generated at: {output_path}")

if __name__ == '__main__':
    os.makedirs('grant_docs', exist_ok=True)
    os.makedirs('company_proposals', exist_ok=True)
    results = evaluate_proposals()
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    generate_pdf(results)

