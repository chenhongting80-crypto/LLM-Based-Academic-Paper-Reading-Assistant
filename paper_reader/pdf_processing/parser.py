"""PDF parsing with page and chunk provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4

import fitz


@dataclass(frozen=True)
class ParsedPDF:
    paper_id: str
    sha256: str
    file_name: str
    page_count: int
    parse_status: str
    page_texts: list[dict[str, object]]
    chunks: list[dict[str, object]]
    warning: str | None = None


def compute_sha256(pdf_bytes: bytes) -> str:
    return sha256(pdf_bytes).hexdigest()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_page_text(text: str, page_number: int, chunk_size: int = 1200, overlap: int = 180) -> list[dict[str, object]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")
    clean = clean_text(text)
    if not clean:
        return []

    chunks: list[dict[str, object]] = []
    start = 0
    chunk_index = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunk_text = clean[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                }
            )
        if end == len(clean):
            break
        start = max(0, end - overlap)
        chunk_index += 1
    return chunks


def parse_pdf_bytes(file_name: str, pdf_bytes: bytes) -> ParsedPDF:
    digest = compute_sha256(pdf_bytes)
    paper_id = str(uuid4())
    page_texts: list[dict[str, object]] = []
    chunks: list[dict[str, object]] = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = doc.page_count
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    page_texts.append({"page_number": page_index, "text": text})
                    chunks.extend(chunk_page_text(text, page_index))
    except Exception as exc:
        return ParsedPDF(
            paper_id=paper_id,
            sha256=digest,
            file_name=file_name,
            page_count=0,
            parse_status="failed",
            page_texts=[],
            chunks=[],
            warning=f"{file_name}: failed to parse PDF. The file may be damaged. Detail: {exc}",
        )

    if not page_texts:
        return ParsedPDF(
            paper_id=paper_id,
            sha256=digest,
            file_name=file_name,
            page_count=page_count,
            parse_status="no_text",
            page_texts=[],
            chunks=[],
            warning=f"{file_name}: no selectable text found. It may be scanned or image-only.",
        )

    return ParsedPDF(
        paper_id=paper_id,
        sha256=digest,
        file_name=file_name,
        page_count=page_count,
        parse_status="completed",
        page_texts=page_texts,
        chunks=chunks,
    )
