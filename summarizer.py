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
from dataclasses import dataclass

import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
MODEL_NAME = "llama3.2:1b"

REQUEST_TIMEOUT = 150      # seconds (spec: 120-180)
MAX_INPUT_CHARS = 9000     # ~2,300 tokens: safe for a 1B model with num_ctx=4096
NUM_CTX = 4096
NUM_PREDICT = 900          # cap on generated tokens
FALLBACK_PREVIEW_CHARS = 4000

SUMMARY_HEADING = "# FINAL DOCUMENT SUMMARY"


@dataclass
class SummaryResult:
    summary: str
    used_fallback: bool
    warning: str = ""


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
def build_prompt(document_name: str, total_pages: int, text: str) -> str:
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


def call_ollama(prompt: str) -> str:
    """Send the prompt to the local Ollama server and return the response text."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
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
def _postprocess(summary: str, document_name: str, total_pages: int) -> str:
    """Enforce heading and force known facts (name, pages) so they are never wrong."""
    summary = summary.strip()
    summary = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", summary).strip()

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
    return summary


def _looks_valid(summary: str) -> bool:
    return len(summary) >= 80 and "DOCUMENT INFORMATION" in summary.upper()


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
def generate_summary(document_name: str, total_pages: int, text: str) -> SummaryResult:
    """Try Ollama first; on ANY failure return a fallback summary. Never raises."""
    try:
        prepared, truncated = prepare_text(text)
        prompt = build_prompt(document_name, total_pages, prepared)
        raw = call_ollama(prompt)
        summary = _postprocess(raw, document_name, total_pages)

        if not _looks_valid(summary):
            raise OllamaError("The AI model returned an incomplete summary.")

        warning = ""
        if truncated:
            warning = (
                "The document is very large, so only its beginning and end were "
                "sent to the AI model. The summary may not cover the middle pages."
            )
        return SummaryResult(summary=summary, used_fallback=False, warning=warning)

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
