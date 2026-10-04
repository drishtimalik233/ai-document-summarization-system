"""
summarizer.py
-------------
Local AI summarization through Ollama (llama3.2:1b) with an automatic,
Ollama-independent fallback summary.

Main entry point:
    result = generate_summary(document_name, total_pages, text)
    result.summary        -> str  (Markdown)
    result.used_fallback  -> bool
    result.warning        -> str  (empty if AI succeeded)

generate_summary() never raises: any failure produces a fallback summary.
"""

import os
import re
from collections import Counter
from dataclasses import dataclass

import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
MODEL_NAME = "llama3.2:1b"

REQUEST_TIMEOUT = 180      # seconds per Ollama call (spec: 120-180)
MAX_INPUT_CHARS = 9000     # ~2,300 tokens: safe for a 1B model with num_ctx=4096
NUM_CTX = 6144
NUM_PREDICT = 900          # cap on generated tokens (general summaries)
EXAM_NUM_PREDICT = 1300    # exam summaries are longer
FALLBACK_PREVIEW_CHARS = 4000

# Chunked (map-reduce) summarization for large, non-exam documents
CHUNK_CHARS = 4000         # size of each part read by the model
MAX_CHUNKS = 5             # max parts read (evenly spread across the document)
CHUNK_NUM_PREDICT = 200    # short notes per part
MAX_NOTES_CHARS = 8000     # cap on combined notes sent to the final call

SUMMARY_HEADING = "# FINAL DOCUMENT SUMMARY"
ENGINE_VERSION = "2.3-exam-rebuild"


@dataclass
class SummaryResult:
    summary: str
    used_fallback: bool
    warning: str = ""
    info: str = ""


class OllamaError(Exception):
    """Raised for any Ollama failure. Message is safe to show to the user."""


# --------------------------------------------------------------------------
# Text preparation (safe truncation for large PDFs)
# --------------------------------------------------------------------------
def prepare_text(text: str, max_chars: int = MAX_INPUT_CHARS):
    """
    Limit text size for the small model.
    Keeps the beginning (60%) and the end (40%), since headers, instructions
    and structure usually appear there.

    Returns (prepared_text, was_truncated).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text, False

    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len
    omitted = len(text) - head_len - tail_len
    prepared = (
        text[:head_len].rstrip()
        + f"\n\n[... {omitted} characters omitted because the document is very large ...]\n\n"
        + text[-tail_len:].lstrip()
    )
    return prepared, True


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
def build_exam_prompt(document_name: str, total_pages: int, text: str) -> str:
    return f"""You are a document summarization assistant. Your ONLY task is to summarize the uploaded document below.

STRICT RULES:
- Summarize ONLY the uploaded document.
- Do not invent information. Do not hallucinate.
- Do not solve mathematical questions. Do not calculate answers.
- Do not generate answers to examination questions.
- Do not change numbers, dates, names or question numbers.
- Do not invent questions. Do not duplicate questions.
- Preserve the original order of questions and sections.
- Only mention topics actually found in the document.
- Only include instructions actually found in the document.
- If information is unavailable, write "Not detected".
- If the document is not an exam paper, write "Not detected" for exam-specific fields and describe the real content under IMPORTANT TOPICS.

Known facts: Document Name = {document_name}; Total Pages = {total_pages}.

Write the summary in EXACTLY this Markdown format:

# FINAL DOCUMENT SUMMARY

## DOCUMENT INFORMATION

- Document Name:
- Subject:
- Subject Code:
- Date:
- Time:
- Maximum Marks:
- Total Pages:
- Total Questions:

## EXAM STRUCTURE

- Section-A:
- Section-B:
- Section-C:

## IMPORTANT TOPICS

- Topic 1
- Topic 2
- Topic 3
- Topic 4
- Topic 5

## QUESTIONS INCLUDED

### Section-A

(List questions in original order, or "Not detected")

### Section-B

(List questions in original order, or "Not detected")

### Section-C

(List questions in original order, or "Not detected")

## IMPORTANT INSTRUCTIONS

(List only instructions actually present in the document, or "Not detected")

===== DOCUMENT START =====
{text}
===== DOCUMENT END =====

Now write the summary. Do NOT answer or solve any question. Only summarize."""


# --------------------------------------------------------------------------
# Document type detection and chunking
# --------------------------------------------------------------------------
EXAM_PATTERNS = [
    r"maximum\s+marks", r"max\.?\s*marks", r"time\s+allowed", r"section\s*[-\u2013]?\s*[abc]\b",
    r"attempt\s+(all|any)", r"subject\s+code", r"roll\s+no", r"\bq\.?\s*\d+[.)]",
    r"end[\s-]*term", r"mid[\s-]*term", r"question\s+paper",
]


def is_exam_document(text: str) -> bool:
    """Heuristic: at least 3 different exam-paper indicators near the start."""
    head = text[:8000]
    hits = sum(1 for pattern in EXAM_PATTERNS if re.search(pattern, head, re.IGNORECASE))
    return hits >= 3


def split_into_chunks(text: str, size: int = CHUNK_CHARS):
    """Pack paragraphs into chunks of at most `size` characters."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        while len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:size])
            para = para[size:]
        if not para:
            continue
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def pick_chunks(chunks, max_chunks: int = MAX_CHUNKS):
    """Return (selected_chunks, was_sampled). Selection is evenly spread, incl. first and last."""
    if len(chunks) <= max_chunks:
        return chunks, False
    indices = sorted({round(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)})
    return [chunks[i] for i in indices], True


def build_chunk_prompt(chunk: str) -> str:
    return f"""Summarize the following part of a document in 4 to 6 short bullet points.
Use ONLY facts stated in the text. Do not invent anything. Do not solve or answer any questions.
Do not add an introduction or conclusion. Output only the bullet points.

TEXT:
{chunk}

BULLET POINTS:"""


def build_general_prompt(document_name: str, total_pages: int, material: str) -> str:
    return f"""You are a document summarization assistant. Write a summary of the document using ONLY the notes/text below.

STRICT RULES:
- Do not invent information. Do not hallucinate.
- Do not invent sections, questions, exam details, dates or names.
- Do not change numbers, dates or names that appear in the text.
- Do not answer or solve any questions found in the text.
- If information is unavailable, write "Not detected".

Known facts: Document Name = {document_name}; Total Pages = {total_pages}.

Write the summary in EXACTLY this Markdown format:

# FINAL DOCUMENT SUMMARY

## DOCUMENT INFORMATION

- Document Name:
- Document Type: (for example research paper, report, notes, article)
- Main Subject:
- Total Pages:

## OVERVIEW

(3 to 5 sentences describing what the document is about)

## KEY POINTS

- (5 to 8 bullet points with the most important findings or ideas)

## IMPORTANT TERMS AND TOPICS

- (list of terms or topics actually found in the text)

## CONCLUSION

(The document's conclusion if present, otherwise "Not detected")

===== DOCUMENT NOTES START =====
{material}
===== DOCUMENT NOTES END =====

Now write ONLY the summary in the exact format above. Do not repeat the notes. Stop after the CONCLUSION section."""


# --------------------------------------------------------------------------
# Ollama communication
# --------------------------------------------------------------------------
def check_ollama_status():
    """
    Returns (is_running: bool, model_available: bool, message: str).
    Never raises.
    """
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
        available = any(name == MODEL_NAME or name.startswith(MODEL_NAME) for name in models)
        if available:
            return True, True, "Ollama is running and the model is available."
        return True, False, f"Ollama is running, but model '{MODEL_NAME}' is not installed."
    except Exception:
        return False, False, "Ollama is not reachable."


def call_ollama(prompt: str, num_predict: int = NUM_PREDICT) -> str:
    """Send the prompt to the local Ollama server and return the response text."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",   # keep the model loaded between calls (faster)
        "options": {
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": num_predict,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise OllamaError(
            "Could not connect to Ollama at localhost:11434. "
            "Make sure Ollama is installed and running."
        )
    except requests.exceptions.Timeout:
        raise OllamaError(
            f"The AI model did not respond within {REQUEST_TIMEOUT} seconds."
        )
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"A network error occurred while contacting Ollama ({type(exc).__name__}).")

    if response.status_code == 404:
        raise OllamaError(
            f"Model '{MODEL_NAME}' was not found. Run: ollama pull {MODEL_NAME}"
        )
    if response.status_code != 200:
        raise OllamaError(f"Ollama returned an error (HTTP {response.status_code}).")

    try:
        data = response.json()
    except ValueError:
        raise OllamaError("Ollama returned an unreadable (non-JSON) response.")

    if not isinstance(data, dict):
        raise OllamaError("Ollama returned an unexpected response format.")
    if data.get("error"):
        raise OllamaError(f"Ollama reported an error: {data['error']}")

    answer = data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise OllamaError("Ollama returned an empty response.")

    return answer.strip()


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------
_ALL_HEADING_NAMES = [
    "DOCUMENT INFORMATION", "EXAM STRUCTURE", "IMPORTANT TOPICS", "QUESTIONS INCLUDED",
    "IMPORTANT INSTRUCTIONS", "OVERVIEW", "KEY POINTS", "IMPORTANT TERMS AND TOPICS", "CONCLUSION",
]


def _normalize_headings(summary: str) -> str:
    """Turn '**NAME**', '# NAME', 'NAME:' etc. into a clean '## NAME' heading."""
    summary = re.sub(r"(?im)^[ \t#*]*FINAL DOCUMENT SUMMARY[ \t*:]*$", SUMMARY_HEADING, summary)
    for name in _ALL_HEADING_NAMES:
        summary = re.sub(rf"(?im)^[ \t#*]*{re.escape(name)}[ \t*:]*$", f"## {name}", summary)
    return summary


def _postprocess(summary: str, document_name: str, total_pages: int) -> str:
    """Enforce heading and force known facts (name, pages) so they are never wrong."""
    summary = summary.strip()
    summary = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", summary).strip()
    summary = _normalize_headings(summary)

    # 1) Cut off anything after the model echoes the input markers / notes.
    for pattern in (
        r"={3,}\s*DOCUMENT\s+(?:NOTES|START|END)",
        r"\n\s*Part\s+\d+\s*:",
        r"\n[^\n]*\(CONT'?D\)",
        r"\n\s*#\s*FINAL DOCUMENT SUMMARY",   # a repeated heading
    ):
        match = re.search(pattern, summary[1:], re.IGNORECASE)
        if match:
            summary = summary[: match.start() + 1].rstrip()

    # 2) Replace copied placeholder text with "Not detected" / remove placeholder-only lines.
    placeholder = r"\((?:for example|list |3 to|5 to|the document's conclusion)[^)]*\)"
    summary = re.sub(
        rf"(?im)^(\s*[-*]\s*[^:\n]+:)\s*{placeholder}\s*$", r"\1 Not detected", summary
    )
    summary = re.sub(rf"(?im)^\s*[-*]?\s*{placeholder}\s*$\n?", "", summary)
    summary = summary.strip()

    if not summary.upper().startswith(SUMMARY_HEADING):
        summary = f"{SUMMARY_HEADING}\n\n{summary}"

    summary = re.sub(
        r"(?im)^(\s*[-*]\s*Document Name:).*$",
        lambda m: f"{m.group(1)} {document_name}",
        summary,
    )
    summary = re.sub(
        r"(?im)^(\s*[-*]\s*Total Pages:).*$",
        lambda m: f"{m.group(1)} {total_pages}",
        summary,
    )

    # If the model skipped these lines entirely, insert them under DOCUMENT INFORMATION.
    missing = []
    if not re.search(r"(?im)^\s*[-*]\s*Document Name:", summary):
        missing.append(f"- Document Name: {document_name}")
    if not re.search(r"(?im)^\s*[-*]\s*Total Pages:", summary):
        missing.append(f"- Total Pages: {total_pages}")
    if missing:
        block = "\n".join(missing)
        heading = re.search(r"(?im)^##\s*DOCUMENT INFORMATION\s*$", summary)
        if heading:
            summary = summary[: heading.end()] + "\n\n" + block + summary[heading.end():]
        else:
            summary = summary.replace(SUMMARY_HEADING, f"{SUMMARY_HEADING}\n\n## DOCUMENT INFORMATION\n\n{block}", 1)
    return summary


def _looks_valid(summary: str) -> bool:
    return len(summary) >= 80 and "DOCUMENT INFORMATION" in summary.upper()


# --------------------------------------------------------------------------
# Exam post-processing: copy facts from the PDF text instead of trusting the model
# --------------------------------------------------------------------------
_STOPWORDS = set("""
about above after again also another because before being below between could does doing during each
every first found given hence here into otherould shall should since some such than that their them then
there these they this those three through under until using value values very what when where whether which
while will with within without would write find prove define following question questions section marks
mark page pages attempt answer answers solve state show explain obtain determine consider calculate
numbers number equal compare discuss subject university examination maximum paper code time date roll
""".split())

_SECTION_RE = re.compile(r"(?im)^[ \t]*section\s*[-\u2013\u2014:]?\s*([A-Ca-c])\b[^\n]*$")
_INSTRUCTION_RE = re.compile(
    r"(?i)\b(note|notes|instruction|instructions|attempt|candidates?|calculator|assume|compulsory|"
    r"allowed|answer all|any one|any two|any three|any five)\b"
)


def extract_keywords(text: str, n: int = 5):
    words = re.findall(r"[A-Za-z]{5,}", text.lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    return [word.capitalize() for word, _ in counts.most_common(n)]


def _clean_exam_text(text: str) -> str:
    text = re.sub(r"(?m)^--- Page \d+ ---\s*$", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_questions_section(text: str):
    """Return (questions_markdown, instructions_markdown). Either may be ''."""
    clean = _clean_exam_text(text)
    matches = list(_SECTION_RE.finditer(clean))
    if len(matches) < 2:
        # Looser fallback: first occurrence of Section A, B, C in increasing order.
        matches, last = [], 0
        for m in re.finditer(r"(?i)section\s*[-\u2013\u2014:]?\s*([A-C])\b", clean):
            if ord(m.group(1).upper()) > last:
                matches.append(m)
                last = ord(m.group(1).upper())
    if len(matches) < 2:
        return "", ""

    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        body = clean[match.start():end].strip()
        letter = match.group(1).upper()
        sections[letter] = (sections[letter] + "\n\n" + body) if letter in sections else body

    parts = []
    for letter in sorted(sections):
        parts.append(f"### Section-{letter}\n\n{sections[letter]}")
    questions_md = "\n\n".join(parts)

    head = clean[: matches[0].start()]
    lines = [ln.strip() for ln in head.splitlines() if _INSTRUCTION_RE.search(ln)]
    instructions_md = "\n".join(f"- {ln}" for ln in lines[:10])
    return questions_md, instructions_md


_SECTION_NAMES = [
    "DOCUMENT INFORMATION", "EXAM STRUCTURE", "IMPORTANT TOPICS",
    "QUESTIONS INCLUDED", "IMPORTANT INSTRUCTIONS",
]


def _get_section(summary: str, name: str) -> str:
    """Body of a section, tolerant of heading style (##, #, **bold**, trailing colon)."""
    others = "|".join(re.escape(n) for n in _SECTION_NAMES)
    pattern = re.compile(
        rf"(?ims)^[ \t#*]*{re.escape(name)}[ \t*:]*$\n?(.*?)(?=^[ \t#*]*(?:{others})[ \t*:]*$|\Z)"
    )
    match = pattern.search(summary)
    return match.group(1).strip() if match else ""


def _finalize_exam(summary: str, text: str, document_name: str, total_pages: int):
    """
    Rebuild the exam summary: keep the model's information/structure text, but take
    topics, questions and instructions from the PDF text so nothing is cut off or invented.
    Returns (summary, info_message).
    """
    information = _get_section(summary, "DOCUMENT INFORMATION")
    structure = _get_section(summary, "EXAM STRUCTURE") or "- Not detected"

    topics = _get_section(summary, "IMPORTANT TOPICS")
    if len(topics) < 5 or re.search(r"(?i)\bTopic\s*\d\b", topics):
        keywords = extract_keywords(text)
        topics = "\n".join(f"- {k}" for k in keywords) if keywords else "- Not detected"

    questions_md, instructions_md = extract_questions_section(text)
    from_pdf = bool(questions_md)
    if not questions_md:
        questions_md = _get_section(summary, "QUESTIONS INCLUDED") or "Not detected"
    if not instructions_md:
        instructions_md = _get_section(summary, "IMPORTANT INSTRUCTIONS") or "- Not detected"

    rebuilt = (
        f"{SUMMARY_HEADING}\n\n"
        f"## DOCUMENT INFORMATION\n\n{information or '- Not detected'}\n\n"
        f"## EXAM STRUCTURE\n\n{structure}\n\n"
        f"## IMPORTANT TOPICS\n\n{topics}\n\n"
        f"## QUESTIONS INCLUDED\n\n{questions_md}\n\n"
        f"## IMPORTANT INSTRUCTIONS\n\n{instructions_md}\n"
    )
    rebuilt = _postprocess(rebuilt, document_name, total_pages)
    info = "Exam paper detected. Questions " + (
        "were copied directly from the PDF text." if from_pdf else "come from the AI model (section headings not found in the PDF)."
    )
    return rebuilt, info


# --------------------------------------------------------------------------
# Fallback (works without Ollama)
# --------------------------------------------------------------------------
def build_fallback_summary(document_name: str, total_pages: int, text: str) -> str:
    """Document-based summary generated locally; never solves questions."""
    body = re.sub(r"[ \t]+", " ", text).strip()
    truncated = len(body) > FALLBACK_PREVIEW_CHARS
    preview = body[:FALLBACK_PREVIEW_CHARS].rstrip()
    if truncated:
        preview += "\n\n[... preview truncated ...]"

    word_count = len(text.split())

    return (
        f"{SUMMARY_HEADING}\n\n"
        "## DOCUMENT INFORMATION\n\n"
        f"- Document Name: {document_name}\n"
        f"- Total Pages: {total_pages}\n"
        f"- Approximate Word Count: {word_count}\n\n"
        "## DOCUMENT CONTENT\n\n"
        "The following is a clean preview of the extracted document text "
        "(automatic fallback, no AI processing):\n\n"
        f"{preview}\n"
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def build_partial_summary(document_name: str, total_pages: int, notes) -> str:
    """Used when some parts were summarized but the final AI call failed."""
    body = "\n\n".join(n.replace("Part ", "**Part ", 1).replace(":\n", "**\n", 1) for n in notes)
    return (
        f"{SUMMARY_HEADING}\n\n"
        "## DOCUMENT INFORMATION\n\n"
        f"- Document Name: {document_name}\n"
        f"- Total Pages: {total_pages}\n\n"
        "## KEY POINTS (from the parts the AI could read)\n\n"
        f"{body}\n"
    )


def generate_summary(document_name: str, total_pages: int, text: str, progress=None) -> SummaryResult:
    """
    Try Ollama first; on ANY failure return a fallback summary. Never raises.
    `progress` is an optional callable taking a status string (used by the UI).
    """
    try:
        warning = ""
        info = "General document (not an exam paper)."
        exam_mode = False
        if is_exam_document(text):
            prepared, truncated = prepare_text(text)
            raw = call_ollama(build_exam_prompt(document_name, total_pages, prepared), EXAM_NUM_PREDICT)
            exam_mode = True
            if truncated:
                warning = (
                    "The document is very large, so only its beginning and end were "
                    "sent to the AI model. The summary may not cover the middle pages."
                )
        else:
            chunks, sampled = pick_chunks(split_into_chunks(text))
            notes, skipped, failures_in_row, last_error = [], 0, 0, None
            if len(chunks) <= 1:
                material = chunks[0] if chunks else text
            else:
                for number, chunk in enumerate(chunks, start=1):
                    if progress:
                        progress(f"🤖 Reading part {number} of {len(chunks)}...")
                    try:
                        notes.append(f"Part {number}:\n" + call_ollama(build_chunk_prompt(chunk), CHUNK_NUM_PREDICT))
                        failures_in_row = 0
                    except OllamaError as exc:
                        last_error, skipped = exc, skipped + 1
                        failures_in_row += 1
                        if failures_in_row >= 2:   # model is clearly not responding; stop waiting
                            break
                if not notes:
                    raise last_error or OllamaError("The AI model could not read the document.")
                material = "\n\n".join(notes)[:MAX_NOTES_CHARS]
            if progress:
                progress("🤖 Writing the final summary...")
            try:
                raw = call_ollama(build_general_prompt(document_name, total_pages, material))
            except OllamaError as exc:
                if not notes:
                    raise
                partial = build_partial_summary(document_name, total_pages, notes)
                return SummaryResult(
                    summary=partial, used_fallback=False,
                    warning=f"The final AI step failed ({exc}). Showing key points from the parts the AI could read.",
                    info="General document: partial AI summary.",
                )
            if sampled:
                warning = (
                    "The document is very large, so the AI read evenly spaced parts of it "
                    "rather than every page. Some details may be missing."
                )
            if skipped:
                warning = (warning + " " if warning else "") + f"{skipped} part(s) could not be read by the AI (timeout)."

        summary = _postprocess(raw, document_name, total_pages)
        if exam_mode:
            summary, info = _finalize_exam(summary, text, document_name, total_pages)
        if not _looks_valid(summary):
            raise OllamaError("The AI model returned an incomplete summary.")
        return SummaryResult(summary=summary, used_fallback=False, warning=warning, info=info)

    except OllamaError as exc:
        reason = str(exc)
    except Exception as exc:  # Safety net: never crash the app.
        reason = f"An unexpected error occurred ({type(exc).__name__})."

    fallback = build_fallback_summary(document_name, total_pages, text)
    return SummaryResult(
        summary=fallback,
        used_fallback=True,
        warning=f"AI summary could not be generated. Reason: {reason}",
    )
