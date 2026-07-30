"""One-time MySQL migration for anonymous workspace isolation.

Run explicitly after taking a database backup. The application never calls this module.
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Callable

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

from paper_reader.database.session import create_app_engine
from paper_reader.workspace import LEGACY_WORKSPACE_ID

PAPER_CHILD_TABLES = ("paper_chunks", "reading_cards", "qa_history", "conversations")


def migrate_legacy_rows(
    connection: Connection,
    id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
) -> dict[str, str]:
    tables = set(inspect(connection).get_table_names())
    if "papers" not in tables:
        raise RuntimeError("The papers table does not exist.")
    paper_columns = {column["name"] for column in inspect(connection).get_columns("papers")}
    if "workspace_id" not in paper_columns:
        raise RuntimeError("Add the workspace_id column before migrating data.")

    old_ids = connection.execute(
        text("SELECT paper_id FROM papers WHERE workspace_id IS NULL OR workspace_id = ''")
    ).scalars().all()
    mapping = {str(old_id): str(id_factory()) for old_id in old_ids}
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("Generated paper IDs were not unique; migration was rolled back.")

    for old_id, new_id in mapping.items():
        for table_name in PAPER_CHILD_TABLES:
            if table_name in tables:
                connection.execute(
                    text(f"UPDATE {table_name} SET paper_id = :new_id WHERE paper_id = :old_id"),
                    {"new_id": new_id, "old_id": old_id},
                )
        connection.execute(
            text(
                "UPDATE papers SET paper_id = :new_id, workspace_id = :workspace_id "
                "WHERE paper_id = :old_id"
            ),
            {"new_id": new_id, "workspace_id": LEGACY_WORKSPACE_ID, "old_id": old_id},
        )

    for table_name in PAPER_CHILD_TABLES:
        if table_name not in tables:
            continue
        orphan_count = connection.execute(
            text(
                f"SELECT COUNT(*) FROM {table_name} child "
                "LEFT JOIN papers paper ON paper.paper_id = child.paper_id "
                "WHERE paper.paper_id IS NULL"
            )
        ).scalar_one()
        if orphan_count:
            raise RuntimeError(f"Migration validation found {orphan_count} orphan rows in {table_name}.")
    if "conversation_messages" in tables and "conversations" in tables:
        orphan_messages = connection.execute(
            text(
                "SELECT COUNT(*) FROM conversation_messages message "
                "LEFT JOIN conversations conversation "
                "ON conversation.conversation_id = message.conversation_id "
                "WHERE conversation.conversation_id IS NULL"
            )
        ).scalar_one()
        if orphan_messages:
            raise RuntimeError(
                f"Migration validation found {orphan_messages} orphan rows in conversation_messages."
            )
    return mapping

def _prepare_mysql_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("papers")}
    with engine.begin() as connection:
        if "workspace_id" not in columns:
            connection.execute(text("ALTER TABLE papers ADD COLUMN workspace_id VARCHAR(36) NULL"))

        for constraint in inspector.get_unique_constraints("papers"):
            if constraint.get("column_names") == ["sha256"] and constraint.get("name"):
                name = constraint["name"].replace("`", "``")
                connection.execute(text(f"ALTER TABLE papers DROP INDEX `{name}`"))


def _finalize_mysql_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    indexes = {index["name"] for index in inspector.get_indexes("papers")}
    unique_constraints = {item["name"] for item in inspector.get_unique_constraints("papers")}
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE papers MODIFY workspace_id VARCHAR(36) NOT NULL"))
        if "ix_papers_workspace_id" not in indexes:
            connection.execute(text("CREATE INDEX ix_papers_workspace_id ON papers (workspace_id)"))
        if "uq_papers_workspace_sha256" not in unique_constraints:
            connection.execute(
                text(
                    "ALTER TABLE papers ADD CONSTRAINT uq_papers_workspace_sha256 "
                    "UNIQUE (workspace_id, sha256)"
                )
            )


def migrate_database(engine: Engine) -> int:
    if engine.dialect.name != "mysql":
        raise RuntimeError("This one-time schema migration supports MySQL only.")
    _prepare_mysql_schema(engine)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            mapping = migrate_legacy_rows(connection)
            transaction.commit()
        except Exception:
            transaction.rollback()
            raise
        finally:
            connection.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
    _finalize_mysql_schema(engine)
    return len(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate existing papers into the legacy workspace.")
    parser.add_argument("--apply", action="store_true", help="Apply the migration to the configured MySQL database.")
    args = parser.parse_args()
    if not args.apply:
        parser.error("Back up the database, stop the app, then rerun with --apply.")
    engine = create_app_engine()
    try:
        count = migrate_database(engine)
        print(f"Migrated {count} paper(s) into legacy workspace {LEGACY_WORKSPACE_ID}.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
