from __future__ import annotations

import unittest
import uuid

from paper_reader.database.repository import PaperRepository, RepositoryError
from paper_reader.database.session import session_factory
from paper_reader.exporting.exporters import report_to_markdown
from paper_reader.models.schemas import ReadingCard, RetrievedChunk
from paper_reader.services.comparison import build_detailed_comparison
from paper_reader.workspace import normalize_workspace_id, resolve_workspace_id
from tests.helpers import DatabaseTestCase


class WorkspaceResolutionTests(unittest.TestCase):
    def test_valid_query_workspace_is_restored(self) -> None:
        workspace_id = str(uuid.uuid4())
        self.assertEqual(resolve_workspace_id(workspace_id, str(uuid.uuid4())), workspace_id)

    def test_invalid_query_workspace_is_replaced(self) -> None:
        generated = uuid.UUID("11111111-1111-1111-1111-111111111111")
        self.assertEqual(resolve_workspace_id("not-a-uuid", id_factory=lambda: generated), str(generated))
        self.assertIsNone(normalize_workspace_id("not-a-uuid"))


class WorkspaceRepositoryIsolationTests(DatabaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.workspace_a = str(uuid.uuid4())
        self.workspace_b = str(uuid.uuid4())
        factory = session_factory(self.engine)
        self.repo_a = PaperRepository(factory, self.workspace_a)
        self.repo_b = PaperRepository(factory, self.workspace_b)
        self.sha = "a" * 64
        self.paper_a = str(uuid.uuid4())
        self.paper_b = str(uuid.uuid4())
        self.repo_a.save_paper_with_chunks(
            self.paper_a,
            "a.pdf",
            self.sha,
            1,
            "completed",
            [{"page_number": 1, "chunk_index": 0, "chunk_text": "workspace A"}],
        )

    def _save_b_same_pdf(self) -> None:
        self.repo_b.save_paper_with_chunks(
            self.paper_b,
            "b.pdf",
            self.sha,
            1,
            "completed",
            [{"page_number": 1, "chunk_index": 0, "chunk_text": "workspace B"}],
        )

    def test_same_sha_is_allowed_and_papers_are_private(self) -> None:
        self._save_b_same_pdf()
        self.assertEqual([row["paper_id"] for row in self.repo_a.list_papers()], [self.paper_a])
        self.assertEqual([row["paper_id"] for row in self.repo_b.list_papers()], [self.paper_b])
        self.assertIsNone(self.repo_b.get_paper(self.paper_a))
        self.assertEqual(self.repo_b.get_chunks(self.paper_a), [])
        self.repo_b.update_parse_status(self.paper_a, "failed")
        self.assertEqual(self.repo_a.get_paper(self.paper_a)["parse_status"], "completed")
        with self.assertRaisesRegex(RepositoryError, "Paper not found"):
            self.repo_b.save_reading_card(self.paper_a, ReadingCard(), "model")
        with self.assertRaisesRegex(RepositoryError, "Paper not found"):
            self.repo_b.save_qa_history(self.paper_a, "x", "y", [], "model")

    def test_delete_only_affects_owner(self) -> None:
        self._save_b_same_pdf()
        self.assertEqual(self.repo_a.delete_papers([self.paper_b]), 0)
        self.assertEqual(self.repo_a.delete_papers([self.paper_a]), 1)
        self.assertIsNotNone(self.repo_b.get_paper(self.paper_b))

    def test_cards_compare_and_export_are_private(self) -> None:
        self._save_b_same_pdf()
        card_a = ReadingCard(research_question="A question", keywords=["A"])
        card_b = ReadingCard(research_question="B question", keywords=["B"])
        self.repo_a.save_reading_card(self.paper_a, card_a, "model")
        self.repo_b.save_reading_card(self.paper_b, card_b, "model")

        self.assertIsNone(self.repo_b.latest_reading_card(self.paper_a))
        self.assertEqual(self.repo_b.delete_reading_card_for_paper(self.paper_a), 0)
        cards_a = self.repo_a.list_saved_reading_cards()
        detail = build_detailed_comparison(cards_a)
        exported = report_to_markdown(cards_a, self.repo_a.list_qa_history())
        self.assertTrue(all(row["paper_id"] == self.paper_a for rows in detail.values() for row in rows))
        self.assertIn("a.pdf", exported)
        self.assertNotIn("b.pdf", exported)

    def test_qa_conversations_and_messages_are_private(self) -> None:
        citation = RetrievedChunk(
            paper_id=self.paper_a,
            file_name="a.pdf",
            page_number=1,
            chunk_index=0,
            chunk_text="evidence",
            score=1.0,
        )
        self.repo_a.save_qa_history(self.paper_a, "question", "answer", [citation], "model")
        conversation = self.repo_a.create_conversation(self.paper_a)
        self.repo_a.add_conversation_turn(conversation["conversation_id"], "question", "answer", [citation])

        self.assertEqual(self.repo_b.list_qa_history(self.paper_a), [])
        self.assertIsNone(self.repo_b.get_conversation(conversation["conversation_id"]))
        self.assertEqual(self.repo_b.list_conversations(self.paper_a), [])
        with self.assertRaisesRegex(RepositoryError, "Chat not found"):
            self.repo_b.update_conversation_title(conversation["conversation_id"], "stolen")
        self.assertEqual(self.repo_b.list_conversation_messages(conversation["conversation_id"]), [])
        self.assertEqual(self.repo_b.delete_conversation(conversation["conversation_id"]), 0)
        self.assertEqual(self.repo_b.delete_conversation_turn(conversation["conversation_id"], 1), 0)
        with self.assertRaisesRegex(RepositoryError, "Chat not found"):
            self.repo_b.add_conversation_turn(conversation["conversation_id"], "x", "y", [])
        self.assertEqual(len(self.repo_a.list_conversation_messages(conversation["conversation_id"])), 2)


if __name__ == "__main__":
    unittest.main()
