from __future__ import annotations

import unittest
import uuid

from sqlalchemy import create_engine, text

from paper_reader.database.migrate_workspace import migrate_legacy_rows
from paper_reader.workspace import LEGACY_WORKSPACE_ID


class WorkspaceMigrationTests(unittest.TestCase):
    @staticmethod
    def _legacy_engine():
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE papers (paper_id VARCHAR(64) PRIMARY KEY, workspace_id VARCHAR(36), sha256 VARCHAR(64))"))
            for table_name in ("paper_chunks", "reading_cards", "qa_history"):
                connection.execute(text(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, paper_id VARCHAR(64))"))
            connection.execute(text("CREATE TABLE conversations (conversation_id VARCHAR(64) PRIMARY KEY, paper_id VARCHAR(64))"))
            connection.execute(text("CREATE TABLE conversation_messages (message_id INTEGER PRIMARY KEY, conversation_id VARCHAR(64))"))
        return engine

    def test_legacy_relationships_are_preserved(self) -> None:
        engine = self._legacy_engine()
        old_id = "a" * 64
        new_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO papers (paper_id, workspace_id, sha256) VALUES (:paper_id, NULL, :sha)"), {"paper_id": old_id, "sha": old_id})
            for table_name in ("paper_chunks", "reading_cards", "qa_history"):
                connection.execute(text(f"INSERT INTO {table_name} (id, paper_id) VALUES (1, :paper_id)"), {"paper_id": old_id})
            connection.execute(text("INSERT INTO conversations VALUES ('conversation-1', :paper_id)"), {"paper_id": old_id})
            connection.execute(text("INSERT INTO conversation_messages VALUES (1, 'conversation-1')"))
            mapping = migrate_legacy_rows(connection, id_factory=lambda: new_id)

            self.assertEqual(mapping, {old_id: str(new_id)})
            paper = connection.execute(text("SELECT paper_id, workspace_id FROM papers")).one()
            self.assertEqual(tuple(paper), (str(new_id), LEGACY_WORKSPACE_ID))
            for table_name in ("paper_chunks", "reading_cards", "qa_history", "conversations"):
                child_id = connection.execute(text(f"SELECT paper_id FROM {table_name}")).scalar_one()
                self.assertEqual(child_id, str(new_id))
            message_conversation = connection.execute(text("SELECT conversation_id FROM conversation_messages")).scalar_one()
            self.assertEqual(message_conversation, "conversation-1")
        engine.dispose()

    def test_validation_failure_rolls_back_data_rewrite(self) -> None:
        engine = self._legacy_engine()
        old_id = "b" * 64
        with engine.begin() as connection:
            connection.execute(text("INSERT INTO papers VALUES (:paper_id, NULL, :sha)"), {"paper_id": old_id, "sha": old_id})
            connection.execute(text("INSERT INTO paper_chunks VALUES (1, 'missing-paper')"))
        with self.assertRaisesRegex(RuntimeError, "orphan rows"):
            with engine.begin() as connection:
                migrate_legacy_rows(
                    connection,
                    id_factory=lambda: uuid.UUID("33333333-3333-3333-3333-333333333333"),
                )
        with engine.connect() as connection:
            paper = connection.execute(text("SELECT paper_id, workspace_id FROM papers")).one()
            self.assertEqual(tuple(paper), (old_id, None))
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
