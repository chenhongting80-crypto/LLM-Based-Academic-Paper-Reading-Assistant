from __future__ import annotations

import unittest

from sqlalchemy import event, select
from sqlalchemy.orm import Session

from paper_reader.database.models import Conversation, ConversationMessage, Paper, PaperChunk, QAHistory, ReadingCardRow
from paper_reader.database.repository import DuplicatePaperError, RepositoryError
from paper_reader.models.schemas import ReadingCard, RetrievedChunk
from paper_reader.services.deletion import apply_successful_deletion_state, delete_selected_papers, selected_paper_names
from tests.helpers import DatabaseTestCase


class DatabaseRepositoryTests(DatabaseTestCase):
    def _save_paper_bundle(self, paper_id: str, file_name: str) -> None:
        self.repository.save_paper_with_chunks(
            paper_id=paper_id,
            file_name=file_name,
            sha256_value=paper_id,
            page_count=1,
            parse_status="completed",
            chunks=[{"page_number": 1, "chunk_index": 0, "chunk_text": f"Chunk for {file_name}"}],
        )
        self.repository.save_reading_card(
            paper_id,
            ReadingCard(
                research_question=f"Question for {file_name}",
                method_data="Method",
                key_findings="Finding",
                limitations="Limitation",
                relevance_takeaway="Relevance",
                keywords=["test"],
            ),
            model_name="mock-model",
            overwrite=True,
        )
        self.repository.save_qa_history(
            paper_id,
            f"What about {file_name}?",
            "Answer.",
            [
                RetrievedChunk(
                    paper_id=paper_id,
                    file_name=file_name,
                    page_number=1,
                    chunk_index=0,
                    chunk_text="Citation",
                    score=0.5,
                )
            ],
            "mock-model",
        )

    def test_mysql_crud_and_duplicate_detection(self) -> None:
        self.repository.save_paper_with_chunks(
            paper_id="a" * 64,
            file_name="paper.pdf",
            sha256_value="a" * 64,
            page_count=2,
            parse_status="completed",
            chunks=[
                {"page_number": 1, "chunk_index": 0, "chunk_text": "PFAS adsorption evidence"},
                {"page_number": 2, "chunk_index": 0, "chunk_text": "LC-MS/MS measurement"},
            ],
        )
        papers = self.repository.list_papers()
        self.assertEqual(len(papers), 1)
        self.assertEqual(len(self.repository.get_chunks("a" * 64)), 2)
        with self.assertRaises(DuplicatePaperError):
            self.repository.save_paper_with_chunks(
                paper_id="b" * 64,
                file_name="copy.pdf",
                sha256_value="a" * 64,
                page_count=2,
                parse_status="completed",
                chunks=[],
            )

    def test_unique_constraint_rolls_back_transaction(self) -> None:
        with self.assertRaises(RepositoryError):
            self.repository.save_paper_with_chunks(
                paper_id="c" * 64,
                file_name="bad.pdf",
                sha256_value="c" * 64,
                page_count=1,
                parse_status="completed",
                chunks=[
                    {"page_number": 1, "chunk_index": 0, "chunk_text": "one"},
                    {"page_number": 1, "chunk_index": 0, "chunk_text": "duplicate position"},
                ],
            )
        self.assertEqual(self.repository.list_papers(), [])

    def test_reading_card_save_and_read(self) -> None:
        self.repository.save_paper_with_chunks("d" * 64, "card.pdf", "d" * 64, 1, "completed", [])
        card = ReadingCard(
            research_question="How is PFAS removed?",
            method_data="Batch adsorption",
            key_findings="High removal",
            limitations="Small scale",
            relevance_takeaway="Useful for groundwater treatment",
            keywords=["PFAS", "adsorption"],
        )
        self.repository.save_reading_card("d" * 64, card, model_name="mock-model")
        saved = self.repository.latest_reading_card("d" * 64)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["card"]["research_question"], "How is PFAS removed?")

    def test_reading_card_skip_duplicate_without_overwrite(self) -> None:
        paper_id = "9" * 64
        self.repository.save_paper_with_chunks(paper_id, "single-card.pdf", paper_id, 1, "completed", [])
        first = ReadingCard(research_question="First")
        second = ReadingCard(research_question="Second")

        self.repository.save_reading_card(paper_id, first, model_name="mock-model", overwrite=False)
        self.repository.save_reading_card(paper_id, second, model_name="mock-model", overwrite=False)

        saved = self.repository.latest_reading_card(paper_id)
        cards = self.repository.list_saved_reading_cards()
        self.assertEqual(saved["card"]["research_question"], "First")
        self.assertEqual(len(cards), 1)
        with self.repository.factory() as session:
            rows = session.execute(select(ReadingCardRow).where(ReadingCardRow.paper_id == paper_id)).all()
        self.assertEqual(len(rows), 1)

    def test_reading_card_overwrite_updates_existing_record(self) -> None:
        paper_id = "0" * 64
        self.repository.save_paper_with_chunks(paper_id, "overwrite-card.pdf", paper_id, 1, "completed", [])
        first = ReadingCard(research_question="First")
        second = ReadingCard(research_question="Second")

        self.repository.save_reading_card(paper_id, first, model_name="old-model", overwrite=True)
        self.repository.save_reading_card(paper_id, second, model_name="new-model", overwrite=True)

        saved = self.repository.latest_reading_card(paper_id)
        cards = self.repository.list_saved_reading_cards()
        self.assertEqual(saved["card"]["research_question"], "Second")
        self.assertEqual(saved["model_name"], "new-model")
        self.assertEqual(len(cards), 1)
        self.assertIn("created_at", cards[0])
        self.assertIn("updated_at", cards[0])
        with self.repository.factory() as session:
            rows = session.execute(select(ReadingCardRow).where(ReadingCardRow.paper_id == paper_id)).all()
        self.assertEqual(len(rows), 1)

    def test_delete_reading_card_does_not_delete_paper(self) -> None:
        paper_id = "b" * 64
        self.repository.save_paper_with_chunks(paper_id, "delete-card-only.pdf", paper_id, 1, "completed", [])
        self.repository.save_reading_card(paper_id, ReadingCard(research_question="Keep paper"), "mock-model")

        deleted_count = self.repository.delete_reading_card_for_paper(paper_id)

        self.assertEqual(deleted_count, 1)
        self.assertIsNotNone(self.repository.get_paper(paper_id))
        self.assertIsNone(self.repository.latest_reading_card(paper_id))

    def test_qa_history_and_citations_saved(self) -> None:
        self.repository.save_paper_with_chunks("e" * 64, "qa.pdf", "e" * 64, 1, "completed", [])
        citation = RetrievedChunk(
            paper_id="e" * 64,
            file_name="qa.pdf",
            page_number=4,
            chunk_index=2,
            chunk_text="Evidence snippet",
            score=0.5,
        )
        self.repository.save_qa_history("e" * 64, "Question?", "Answer.", [citation], "mock-model")
        rows = self.repository.list_qa_history("e" * 64)
        self.assertEqual(rows[0]["citation_pages"], [4])
        self.assertEqual(rows[0]["citation_snippets"][0]["snippet"], "Evidence snippet")

    def test_conversation_messages_are_isolated_by_chat(self) -> None:
        paper_id = "a1" * 32
        self.repository.save_paper_with_chunks(paper_id, "project.pdf", paper_id, 1, "completed", [])
        first = self.repository.create_conversation(paper_id)
        second = self.repository.create_conversation(paper_id)
        citation = RetrievedChunk(
            paper_id=paper_id,
            file_name="project.pdf",
            page_number=2,
            chunk_index=0,
            chunk_text="Project-specific evidence",
            score=0.7,
        )

        self.repository.add_conversation_turn(first["conversation_id"], "What methods were used?", "Methods answer.", [citation])
        self.repository.add_conversation_turn(second["conversation_id"], "What are the findings?", "Findings answer.", [])

        first_messages = self.repository.list_conversation_messages(first["conversation_id"])
        second_messages = self.repository.list_conversation_messages(second["conversation_id"])
        conversations = self.repository.list_conversations(paper_id)

        self.assertEqual(len(first_messages), 2)
        self.assertEqual(len(second_messages), 2)
        self.assertEqual(first_messages[0]["content"], "What methods were used?")
        self.assertEqual(second_messages[0]["content"], "What are the findings?")
        self.assertEqual(first_messages[1]["sources"][0]["page_number"], 2)
        self.assertEqual(len(conversations), 2)

    def test_delete_conversation_removes_only_that_chat(self) -> None:
        paper_id = "a2" * 32
        self.repository.save_paper_with_chunks(paper_id, "delete-chat.pdf", paper_id, 1, "completed", [])
        first = self.repository.create_conversation(paper_id)
        second = self.repository.create_conversation(paper_id)
        self.repository.add_conversation_turn(first["conversation_id"], "First?", "First answer.", [])
        self.repository.add_conversation_turn(second["conversation_id"], "Second?", "Second answer.", [])

        deleted_count = self.repository.delete_conversation(first["conversation_id"])

        self.assertEqual(deleted_count, 1)
        self.assertEqual(self.repository.list_conversation_messages(first["conversation_id"]), [])
        self.assertEqual(len(self.repository.list_conversation_messages(second["conversation_id"])), 2)
        self.assertIsNotNone(self.repository.get_paper(paper_id))

    def test_delete_conversation_turn_removes_question_and_answer_only(self) -> None:
        paper_id = "a3" * 32
        self.repository.save_paper_with_chunks(paper_id, "delete-turn.pdf", paper_id, 1, "completed", [])
        conversation = self.repository.create_conversation(paper_id)
        self.repository.add_conversation_turn(conversation["conversation_id"], "First?", "First answer.", [])
        self.repository.add_conversation_turn(conversation["conversation_id"], "Second?", "Second answer.", [])

        deleted_count = self.repository.delete_conversation_turn(conversation["conversation_id"], 1)
        messages = self.repository.list_conversation_messages(conversation["conversation_id"])

        self.assertEqual(deleted_count, 2)
        self.assertEqual(len(messages), 2)
        self.assertEqual({message["turn_id"] for message in messages}, {2})
        self.assertEqual(messages[0]["content"], "Second?")

    def test_delete_paper_removes_conversations_and_messages(self) -> None:
        paper_id = "a4" * 32
        self.repository.save_paper_with_chunks(paper_id, "delete-project.pdf", paper_id, 1, "completed", [])
        conversation = self.repository.create_conversation(paper_id)
        self.repository.add_conversation_turn(conversation["conversation_id"], "Question?", "Answer.", [])

        self.repository.delete_papers([paper_id])

        self.assertIsNone(self.repository.get_paper(paper_id))
        with self.repository.factory() as session:
            self.assertIsNone(session.get(Conversation, conversation["conversation_id"]))
            self.assertIsNone(
                session.execute(
                    select(ConversationMessage).where(
                        ConversationMessage.conversation_id == conversation["conversation_id"]
                    )
                ).first()
            )

    def test_delete_one_paper_removes_related_records(self) -> None:
        paper_id = "f" * 64
        self._save_paper_bundle(paper_id, "single-delete.pdf")
        deleted_count = self.repository.delete_papers([paper_id])
        self.assertEqual(deleted_count, 1)
        self.assertIsNone(self.repository.get_paper(paper_id))
        self.assertEqual(self.repository.get_chunks(paper_id), [])
        self.assertIsNone(self.repository.latest_reading_card(paper_id))
        self.assertEqual(self.repository.list_qa_history(paper_id), [])

    def test_delete_multiple_papers_keeps_unselected_unchanged(self) -> None:
        first_id = "1" * 64
        second_id = "2" * 64
        keep_id = "3" * 64
        self._save_paper_bundle(first_id, "multi-delete-a.pdf")
        self._save_paper_bundle(second_id, "multi-delete-b.pdf")
        self._save_paper_bundle(keep_id, "keep.pdf")

        deleted_count = self.repository.delete_papers([first_id, second_id])

        self.assertEqual(deleted_count, 2)
        self.assertIsNone(self.repository.get_paper(first_id))
        self.assertIsNone(self.repository.get_paper(second_id))
        self.assertIsNotNone(self.repository.get_paper(keep_id))
        self.assertEqual(len(self.repository.get_chunks(keep_id)), 1)
        self.assertIsNotNone(self.repository.latest_reading_card(keep_id))
        self.assertEqual(len(self.repository.list_qa_history(keep_id)), 1)

    def test_cancelled_deletion_does_not_delete(self) -> None:
        paper_id = "4" * 64
        self._save_paper_bundle(paper_id, "cancel-delete.pdf")
        deleted_count = delete_selected_papers(self.repository, [paper_id], confirmed=False)
        self.assertEqual(deleted_count, 0)
        self.assertIsNotNone(self.repository.get_paper(paper_id))
        papers = self.repository.list_papers()
        self.assertEqual(selected_paper_names(papers, [paper_id]), ["cancel-delete.pdf"])

    def test_delete_papers_rolls_back_when_delete_fails(self) -> None:
        first_id = "5" * 64
        second_id = "6" * 64
        self._save_paper_bundle(first_id, "rollback-delete-a.pdf")
        self._save_paper_bundle(second_id, "rollback-delete-b.pdf")

        def fail_before_execute(state) -> None:
            statement = str(state.statement)
            if "DELETE FROM paper_chunks" in statement:
                raise RuntimeError("forced delete failure")

        event.listen(Session, "do_orm_execute", fail_before_execute)
        try:
            with self.assertRaises(RepositoryError):
                self.repository.delete_papers([first_id, second_id])
        finally:
            event.remove(Session, "do_orm_execute", fail_before_execute)

        with self.repository.factory() as session:
            self.assertIsNotNone(session.get(Paper, first_id))
            self.assertIsNotNone(session.get(Paper, second_id))
            self.assertIsNotNone(session.execute(select(PaperChunk).where(PaperChunk.paper_id == first_id)).first())
            self.assertIsNotNone(session.execute(select(ReadingCardRow).where(ReadingCardRow.paper_id == first_id)).first())
            self.assertIsNotNone(session.execute(select(QAHistory).where(QAHistory.paper_id == first_id)).first())

    def test_successful_deletion_clears_ui_selection_state(self) -> None:
        deleted_id = "7" * 64
        remaining_id = "8" * 64
        state = {
            "selected_paper_id": deleted_id,
            "pending_delete_paper_ids": [deleted_id],
            "paper_delete_selection_reset": 3,
        }
        remaining_papers = [{"paper_id": remaining_id, "file_name": "remaining.pdf"}]

        apply_successful_deletion_state(state, [deleted_id], remaining_papers)

        self.assertEqual(state["pending_delete_paper_ids"], [])
        self.assertEqual(state["paper_delete_selection_reset"], 4)
        self.assertEqual(state["selected_paper_id"], remaining_id)

    def test_deleting_all_papers_clears_reading_card_generation_state(self) -> None:
        deleted_id = "9" * 64
        state = {
            "selected_paper_id": deleted_id,
            "pending_delete_paper_ids": [deleted_id],
            "last_upload_messages": ["Saved deleted.pdf to MySQL."],
            "reading_card_generation_summary": {
                "success": ["deleted.pdf"],
                "skipped": [],
                "failed": [],
            },
            "reading_card_action_message": "Successfully generated",
            "reading_card_selected_paper_ids": [deleted_id],
            "reading_card_selected_paper_names": ["deleted.pdf"],
        }

        apply_successful_deletion_state(state, [deleted_id], [])

        self.assertEqual(state["last_upload_messages"], [])
        self.assertIsNone(state["reading_card_generation_summary"])
        self.assertEqual(state["reading_card_action_message"], "")
        self.assertEqual(state["reading_card_selected_paper_ids"], [])
        self.assertEqual(state["reading_card_selected_paper_names"], [])
        self.assertEqual(state["selected_paper_id"], "")

if __name__ == "__main__":
    unittest.main()
