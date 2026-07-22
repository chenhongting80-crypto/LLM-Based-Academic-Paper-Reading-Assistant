"""Paper ingestion and library service."""

from __future__ import annotations

from typing import Any

from paper_reader.database.repository import DuplicatePaperError, PaperRepository, RepositoryError
from paper_reader.pdf_processing.parser import ParsedPDF, parse_pdf_bytes


def ingest_pdf(file_name: str, pdf_bytes: bytes, repository: PaperRepository | None) -> tuple[ParsedPDF, str]:
    if repository is None:
        raise RepositoryError("Database connection is unavailable. Configure and connect to MySQL before processing PDFs.")
    parsed = parse_pdf_bytes(file_name, pdf_bytes)

    try:
        repository.save_paper_with_chunks(
            paper_id=parsed.paper_id,
            file_name=parsed.file_name,
            sha256_value=parsed.sha256,
            page_count=parsed.page_count,
            parse_status=parsed.parse_status,
            chunks=parsed.chunks,
        )
        if parsed.warning:
            return parsed, parsed.warning
        return parsed, f"Saved {file_name} to MySQL."
    except DuplicatePaperError as exc:
        parsed = ParsedPDF(
            paper_id=exc.paper_id,
            sha256=parsed.sha256,
            file_name=exc.file_name,
            page_count=parsed.page_count,
            parse_status="duplicate",
            page_texts=parsed.page_texts,
            chunks=parsed.chunks,
            warning=f"Duplicate upload detected. Existing paper: {exc.file_name}",
        )
        return parsed, parsed.warning or "Duplicate upload detected."
    except RepositoryError as exc:
        return parsed, str(exc)


def paper_chunks_for_retrieval(repository: PaperRepository, paper_id: str, file_name: str) -> list[dict[str, Any]]:
    chunks = repository.get_chunks(paper_id)
    return [
        {
            "paper_id": item["paper_id"],
            "file_name": file_name,
            "page_number": item["page_number"],
            "chunk_index": item["chunk_index"],
            "chunk_text": item["chunk_text"],
        }
        for item in chunks
    ]
