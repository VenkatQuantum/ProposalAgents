# PROPOSAL AGENTS

## 📂 Example Directory Structure

```
.
├── chroma_db/           # Persisted Chroma vector store
├── grant_docs/          # Place grant proposal PDFs here
├── company_info.json    # JSON file containing company profile
├── embed.py             # Script to ingest and embed data into Chroma
├── qualifier.py         # Script to evaluate proposals against the profile
├── requirements.txt     # Python dependencies
└── .env                 # Environment variable definitions (ignored by Git)
```

## 🛠 Prerequisites

* **Python**
* **Ollama** server running locally (default: `http://localhost:11434`)
* Download instructions: https://github.com/ollama/ollama
* Pull required models
   ```bash
  ollama pull mxbai-embed-large
  ollama pull llama3.2:3b
   ```

  * Ensure the embed model (e.g., `mxbai-embed-large`) and chat model (e.g., `llama3.2:3b`) are available.
* **pip** and **venv**, or use your preferred virtual environment tool.

## ⚙️ Setup

1. **Clone the repository**

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   * Copy `.env` (already in repo but untracked) or create your own with the following:

     ```ini
     OLLAMA_URL=http://localhost:11434
     OLLAMA_EMBED_MODEL=mxbai-embed-large
     OLLAMA_CHAT_MODEL=llama3.2:3b
     CHROMA_PERSIST_PATH=./chroma_db
     ```

## 🚀 Usage

### 1. Ingest & Embed Data

Run the `embed.py` script to load and index:

```bash
python embed.py
```

* Indexes the **company profile** as a single vector.
* Splits each PDF in `grant_docs/` into chunks, embeds them, and adds them to Chroma.
* Persists the vector store to `chroma_db/`.

### 2. Evaluate Proposals

Run the `qualifier.py` script to assess each proposal:

```bash
python qualifier.py
```


