"""Safe text extraction for Chat Lab document uploads.

Images are rejected here as well as in the browser. PDF and Word parsing is
imported lazily so the rest of the Python stack still works without those extras.
"""

from __future__ import annotations

import io
import re
from typing import Tuple

MAX_BYTES = 5 * 1024 * 1024
MAX_CHARS = 200_000

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json"}
ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".avif",
    ".heic", ".heif", ".tif", ".tiff", ".ico",
}


class DocumentError(ValueError):
    """Raised for unsupported or unreadable uploaded documents."""


def extension_of(name: str) -> str:
    value = str(name or "")
    dot = value.rfind(".")
    return value[dot:].lower() if dot >= 0 else ""


def validate_document(name: str, content_type: str = "") -> str:
    """Return the normalized extension or raise a readable error."""
    extension = extension_of(name)
    mime = str(content_type or "").lower()
    if extension in IMAGE_EXTENSIONS or mime.startswith("image/"):
        raise DocumentError(
            "Images are not supported. Load a TXT, MD, CSV, JSON, PDF or Word (.docx) document."
        )
    if extension == ".doc":
        raise DocumentError(
            "Legacy .doc files are not supported. Save the file as .docx and load it again."
        )
    if extension not in ALLOWED_EXTENSIONS:
        raise DocumentError(
            "Load a TXT, MD, CSV, JSON, PDF or Word (.docx) document. Images are not supported."
        )
    return extension


def normalize_text(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def extract_text(name: str, data: bytes, content_type: str = "") -> Tuple[str, bool]:
    """Extract readable text from an uploaded document.

    Returns (text, truncated). Only text is kept; no images are ever processed.
    """
    extension = validate_document(name, content_type)
    if not data:
        raise DocumentError("That document is empty.")
    if len(data) > MAX_BYTES:
        raise DocumentError("That document is too large. The maximum size is 5 MB.")

    if extension in TEXT_EXTENSIONS:
        text = data.decode("utf-8", errors="replace")
    elif extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DocumentError("PDF support is not installed in the Python service.") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise DocumentError("I could not read that PDF.") from exc
    else:
        try:
            from docx import Document
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DocumentError("Word support is not installed in the Python service.") from exc
        try:
            document = Document(io.BytesIO(data))
            parts = [paragraph.text for paragraph in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    for cell in row.cells:
                        parts.append(cell.text)
            text = "\n".join(parts)
        except Exception as exc:
            raise DocumentError("I could not read that Word document.") from exc

    cleaned = normalize_text(text)
    if not cleaned:
        raise DocumentError("I could not find any readable text in that document.")
    truncated = len(cleaned) > MAX_CHARS
    return cleaned[:MAX_CHARS], truncated
