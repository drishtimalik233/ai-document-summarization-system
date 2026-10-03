"""
pdf_extractor.py
----------------
Reusable PDF text extraction built on pypdf.

Main entry point:
    text, total_pages = extract_text_from_pdf(uploaded_file)

Raises PDFExtractionError with a user-friendly message for invalid,
encrypted, empty or scanned (image-only) PDFs.
"""

import io
import re

from pypdf import PdfReader

# Minimum number of extracted characters for a PDF to count as "text-based".
MIN_TEXT_CHARS = 20

SCANNED_PDF_MESSAGE = (
    "This PDF appears to be scanned or image-based. "
    "Text extraction was not possible. OCR would be required."
)


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be read. The message is safe to show to users."""


def _read_bytes(uploaded_file) -> bytes:
    """Return raw bytes from a Streamlit UploadedFile, file object, or bytes."""
    if isinstance(uploaded_file, (bytes, bytearray)):
        return bytes(uploaded_file)
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return uploaded_file.read()
    raise PDFExtractionError("No valid file was provided.")


def _clean_page_text(text: str) -> str:
    """Remove null bytes and excessive blank lines while keeping line structure."""
    text = text.replace("\x00", "")
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_text_from_pdf(uploaded_file):
    """
    Extract text page by page, preserving page boundaries.

    Returns:
        (text, total_pages)
    """
    data = _read_bytes(uploaded_file)
    if not data:
        raise PDFExtractionError("The uploaded file is empty.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:
        raise PDFExtractionError(
            "The file could not be opened. It may be corrupted or not a valid PDF."
        )

    # Handle password-protected PDFs (empty user password only).
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise PDFExtractionError(
                    "This PDF is password-protected. Please upload an unprotected PDF."
                )
        except PDFExtractionError:
            raise
        except Exception:
            raise PDFExtractionError(
                "This PDF is encrypted and could not be opened. "
                "Please upload an unprotected PDF."
            )

    try:
        total_pages = len(reader.pages)
    except Exception:
        raise PDFExtractionError("The PDF structure is damaged; pages could not be counted.")

    if total_pages == 0:
        raise PDFExtractionError("The PDF contains no pages.")

    page_blocks = []
    total_chars = 0

    for page_number in range(total_pages):
        try:
            raw = reader.pages[page_number].extract_text() or ""
        except Exception:
            raw = ""  # Skip a problematic page instead of failing everything.

        page_text = _clean_page_text(raw)
        total_chars += len(page_text)

        if not page_text:
            page_text = "[No text could be extracted from this page]"

        page_blocks.append(f"--- Page {page_number + 1} ---\n{page_text}")

    if total_chars < MIN_TEXT_CHARS:
        raise PDFExtractionError(SCANNED_PDF_MESSAGE)

    return "\n\n".join(page_blocks), total_pages
