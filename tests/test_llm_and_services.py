from __future__ import annotations

import unittest
from unittest.mock import patch

from paper_reader.llm.client import _safe_llm_error
from paper_reader.llm.parsing import parse_json_model
from paper_reader.models.schemas import ReadingCard
from paper_reader.services.qa import INSUFFICIENT_EVIDENCE_MESSAGE, answer_from_retrieved_chunks
from paper_reader.services.reading_cards import generate_reading_card_from_chunks


class LlmAndServiceTests(unittest.TestCase):
    def test_reading_card_parses_standard_json(self) -> None:
        raw = '{"research_question":"Q","method_data":"M","key_findings":"K","limitations":"L","relevance_takeaway":"R","keywords":["PFAS"]}'
        parsed, error = parse_json_model(raw, ReadingCard)
        self.assertIsNone(error)
        self.assertEqual(parsed.research_question, "Q")
        self.assertEqual(parsed.keywords, ["PFAS"])

    def test_structured_output_parsing_with_fallback(self) -> None:
        raw = 'Here is JSON:\n{"research_question":"Q","method_data":"M","key_findings":"K","limitations":"L","relevance_takeaway":"R","keywords":["PFAS"]}'
        parsed, error = parse_json_model(raw, ReadingCard)
        self.assertIsNone(error)
        self.assertEqual(parsed.keywords, ["PFAS"])

    def test_reading_card_parses_json_code_fence(self) -> None:
        raw = '```json\n{"research_question":"Q","method_data":"M","key_findings":"K","limitations":"L","relevance_takeaway":"R","keywords":["water"]}\n```'
        parsed, error = parse_json_model(raw, ReadingCard)
        self.assertIsNone(error)
        self.assertEqual(parsed.method_data, "M")

    def test_reading_card_joins_list_text_fields(self) -> None:
        raw = '{"research_question":["Question one","Question two"],"keywords":[]}'
        parsed, error = parse_json_model(raw, ReadingCard)
        self.assertIsNone(error)
        self.assertEqual(parsed.research_question, "Question one; Question two")
        self.assertEqual(parsed.method_data, ReadingCard().method_data)

    def test_reading_card_flattens_dict_text_fields(self) -> None:
        raw = '{"method_data":{"method":"batch tests","data":"water samples"},"keywords":[]}'
        parsed, error = parse_json_model(raw, ReadingCard)
        self.assertIsNone(error)
        self.assertEqual(parsed.method_data, "method: batch tests; data: water samples")

    def test_reading_card_splits_comma_separated_keywords(self) -> None:
        raw = '{"keywords":"PFAS, adsorption, water treatment"}'
        parsed, error = parse_json_model(raw, ReadingCard)
        self.assertIsNone(error)
        self.assertEqual(parsed.keywords, ["PFAS", "adsorption", "water treatment"])

    def test_reading_card_rejects_non_json_text(self) -> None:
        parsed, error = parse_json_model("This is not JSON.", ReadingCard)
        self.assertIsNone(parsed)
        self.assertEqual(error, "LLM output could not be parsed as the required JSON object.")

    def test_reading_card_generation_repairs_output_once(self) -> None:
        repaired = '{"research_question":"Q","method_data":"M","key_findings":"K","limitations":"L","relevance_takeaway":"R","keywords":["water"]}'
        with (
            patch("paper_reader.services.reading_cards.local_chunk_summaries", return_value=(["summary"], None)),
            patch(
                "paper_reader.services.reading_cards.invoke_text",
                side_effect=[('{"research_question": 5}', None), (repaired, None)],
            ) as invoke,
        ):
            card, error, summaries = generate_reading_card_from_chunks("paper.pdf", [], object())

        self.assertIsNone(error)
        self.assertEqual(card.research_question, "Q")
        self.assertEqual(summaries, ["summary"])
        self.assertEqual(invoke.call_count, 2)

    def test_llm_error_redacts_api_key_fragments(self) -> None:
        error = _safe_llm_error(RuntimeError("Incorrect API key provided: sk-abc123********xyz."))
        self.assertNotIn("sk-abc123", error)
        self.assertIn("[REDACTED]", error)

    def test_qa_no_evidence_does_not_invent_citation(self) -> None:
        result, error = answer_from_retrieved_chunks("What is the removal rate?", [], llm=None)
        self.assertIsNone(error)
        self.assertEqual(result.answer, INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertEqual(result.citations, [])

    def test_qa_retrieval_augments_author_and_conclusion_questions(self) -> None:
        chunks = [
            {
                "paper_id": "p1",
                "file_name": "paper.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "chunk_text": "Jane Smith, Alex Chen, and Priya Patel. A study of photocatalysis.",
            },
            {
                "paper_id": "p1",
                "file_name": "paper.pdf",
                "page_number": 8,
                "chunk_index": 0,
                "chunk_text": "The conclusion is that the catalyst improves degradation under visible light.",
            },
        ]

        author_result, _ = answer_from_retrieved_chunks("List three authors of the paper.", chunks, llm=None)
        conclusion_result, _ = answer_from_retrieved_chunks("What is the conclusion of this paper?", chunks, llm=None)

        self.assertTrue(any(item.page_number == 1 for item in author_result.citations))
        self.assertTrue(any("conclusion" in item.chunk_text.lower() for item in conclusion_result.citations))


if __name__ == "__main__":
    unittest.main()
