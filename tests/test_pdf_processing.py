from __future__ import annotations

import unittest
import uuid

from paper_reader.database.repository import RepositoryError
from paper_reader.pdf_processing.parser import chunk_page_text, compute_sha256, parse_pdf_bytes
from paper_reader.services.papers import ingest_pdf
from tests.helpers import make_pdf_bytes


class PdfProcessingTests(unittest.TestCase):
    def test_ingestion_requires_database_repository(self) -> None:
        with self.assertRaisesRegex(RepositoryError, "Database connection is unavailable"):
            ingest_pdf("paper.pdf", b"not parsed", repository=None)

    def test_chunk_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            chunk_page_text("text", page_number=1, chunk_size=0, overlap=0)

    def test_overlap_must_be_smaller_than_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_page_text("text", page_number=1, chunk_size=10, overlap=10)

    def test_paper_id_is_independent_from_sha256(self) -> None:
        pdf_bytes = make_pdf_bytes(["PFAS removal in groundwater"])
        self.assertEqual(compute_sha256(pdf_bytes), compute_sha256(pdf_bytes))
        parsed = parse_pdf_bytes("paper.pdf", pdf_bytes)
        self.assertEqual(str(uuid.UUID(parsed.paper_id)), parsed.paper_id)
        self.assertNotEqual(parsed.paper_id, parsed.sha256)

    def test_chunking_keeps_page_numbers(self) -> None:
        chunks = chunk_page_text("alpha beta gamma " * 200, page_number=3, chunk_size=80, overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["page_number"] == 3 for chunk in chunks))

    def test_empty_pdf_gets_no_text_status(self) -> None:
        pdf_bytes = make_pdf_bytes([""])
        parsed = parse_pdf_bytes("scan.pdf", pdf_bytes)
        self.assertEqual(parsed.parse_status, "no_text")
        self.assertEqual(parsed.chunks, [])


if __name__ == "__main__":
    unittest.main()
