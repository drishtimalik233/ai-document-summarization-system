# 📄 AI Document Summarization System

## 1. Project Description
A locally running web application that lets a user upload a PDF, extracts its text,
and generates a structured summary using the **llama3.2:1b** model served by **Ollama**.
No paid API, subscription or internet connection is needed for AI inference.
If the AI is unavailable, a document-based fallback summary is generated automatically.

## 2. Features
- Upload any text-based PDF; page count detection
- Page-by-page text extraction with page boundaries preserved
- Viewable extracted text
- Structured summary (document info, exam structure, topics, questions, instructions)
- Download summary as a TXT file
- Safe handling of large PDFs (head + tail truncation)
- Automatic fallback summary when Ollama fails, times out or returns nothing
- Friendly error messages (invalid, empty, encrypted, scanned PDFs; Ollama offline)
- The AI is instructed never to solve or answer exam questions
- Live Ollama status indicator in the sidebar

## 3. Technologies Used
Python, Streamlit, Ollama (llama3.2:1b), pypdf, requests

## 4. System Requirements
- Windows 11 (also works on macOS/Linux), PowerShell, VS Code
- Python 3.10 – 3.14
- 8 GB RAM recommended (4 GB minimum), ~2 GB free disk for the model

## 5. Installation
```powershell
cd "Document Summarization System"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 6. Ollama Installation
Download from https://ollama.com/download (Windows installer). Ollama then runs in the background
on `http://localhost:11434`. Verify with `ollama --version`.

## 7. Pulling the Model
```powershell
ollama pull llama3.2:1b
```

## 8. Running the Project
```powershell
streamlit run app.py
```
Open http://localhost:8501 if the browser does not open automatically.

## 9. Project Structure
```
Document Summarization System/
├── app.py              # Streamlit UI
├── summarizer.py       # Ollama call, prompt, truncation, fallback
├── pdf_extractor.py    # pypdf text extraction
├── requirements.txt
├── README.md
├── Dockerfile          # optional, for deployment
├── .gitignore
├── .streamlit/config.toml
└── sample/
    ├── README.txt
    └── sample_exam.pdf # test PDF
```

## 10. How the System Works
1. The user uploads a PDF; `pdf_extractor.py` reads it with pypdf, page by page.
2. Text and page count are shown in the interface.
3. On "Generate Summary", `summarizer.py` truncates very large text (keeps beginning and end),
   builds a strict prompt and calls `POST /api/generate` (`stream=false`, `temperature=0`, 150 s timeout).
4. The response is post-processed: the heading is enforced and *Document Name* / *Total Pages*
   are overwritten with the true values.
5. On any failure, `build_fallback_summary()` produces a summary from the extracted text (no Ollama needed).
6. The summary is displayed and can be downloaded as `document_summary.txt`.

## 11. Limitations
- Scanned/image-only PDFs are not supported (no OCR is implemented).
- Math symbols may be extracted imperfectly by pypdf.
- llama3.2:1b is small; output may occasionally be imperfect. Very large PDFs are truncated.
- The exam-oriented format shows "Not detected" for non-exam documents.

## 12. Future Scope
OCR support (Tesseract), chunked map-reduce summarization, DOCX/TXT input, larger models,
multi-language support, PDF/DOCX export of summaries, user authentication.

## Deployment
Because AI inference needs an Ollama server, choose one option:

**A. Share from your own PC (easiest demo).**
Run `streamlit run app.py --server.address 0.0.0.0`, then use your LAN URL
(`http://<your-ip>:8501`), or a free tunnel such as Cloudflare Tunnel:
`cloudflared tunnel --url http://localhost:8501`.

**B. Docker.**
```powershell
docker build -t doc-summarizer .
docker run -p 8501:8501 -e OLLAMA_HOST=http://host.docker.internal:11434 doc-summarizer
```

**C. VPS / cloud VM (full deployment).**
Install Ollama and this app on the same Linux VM (4+ GB RAM), run `ollama pull llama3.2:1b`,
then run Streamlit behind Nginx or via Docker.

**D. Streamlit Community Cloud.**
Deploys the UI from GitHub for free, but it cannot run Ollama, so the app will use the
fallback summary unless `OLLAMA_HOST` (in app Secrets/env) points to a publicly reachable Ollama server.

## Common Errors
| Problem | Fix |
|---|---|
| `streamlit` not recognized | Activate venv, re-run `pip install -r requirements.txt`, or use `python -m streamlit run app.py` |
| Could not connect to Ollama | Start Ollama (open the app / `ollama serve`) |
| Model not found | `ollama pull llama3.2:1b` |
| Timeout | Close other heavy apps, try a smaller PDF, retry (first call loads the model) |
| Scanned PDF message | The PDF is an image; OCR is required |
| Activate.ps1 blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
