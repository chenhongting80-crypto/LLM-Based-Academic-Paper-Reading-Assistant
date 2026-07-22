from __future__ import annotations

import unittest

from paper_reader.retrieval.tfidf import TfidfRetriever


class RetrievalTests(unittest.TestCase):
    def test_tfidf_retrieves_relevant_chunk(self) -> None:
        chunks = [
            {
                "paper_id": "a",
                "file_name": "air.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "chunk_text": "PM2.5 monitoring uses optical sensors for air quality.",
            },
            {
                "paper_id": "a",
                "file_name": "air.pdf",
                "page_number": 2,
                "chunk_index": 0,
                "chunk_text": "Anaerobic digestion treats food waste.",
            },
        ]
        retriever = TfidfRetriever(chunks)
        retriever.build()
        results = retriever.search("air quality optical PM2.5", top_k=1)
        self.assertEqual(results[0].page_number, 1)
        self.assertGreater(results[0].score, 0)

    def test_empty_chunks_do_not_shift_result_metadata(self) -> None:
        chunks = [
            {
                "paper_id": "empty",
                "file_name": "empty.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "chunk_text": "   ",
            },
            {
                "paper_id": "target",
                "file_name": "target.pdf",
                "page_number": 9,
                "chunk_index": 3,
                "chunk_text": "Activated carbon adsorption removes contaminants from water.",
            },
        ]
        retriever = TfidfRetriever(chunks)
        retriever.build()

        result = retriever.search("activated carbon adsorption", top_k=1)[0]

        self.assertEqual(result.paper_id, "target")
        self.assertEqual(result.page_number, 9)
        self.assertEqual(result.chunk_index, 3)


if __name__ == "__main__":
    unittest.main()
