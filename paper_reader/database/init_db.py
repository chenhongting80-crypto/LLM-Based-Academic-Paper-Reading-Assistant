"""Database initialization and legacy migration CLI."""

from __future__ import annotations

from pathlib import Path

from paper_reader.database.repository import PaperRepository
from paper_reader.database.session import create_app_engine, init_database, session_factory


def initialize_and_migrate(legacy_json_path: Path = Path("data") / "reading_cards.json") -> tuple[int, list[str]]:
    engine = create_app_engine()
    init_database(engine)
    repository = PaperRepository(session_factory(engine))
    return repository.migrate_reading_cards_json(legacy_json_path)


def main() -> None:
    migrated, warnings = initialize_and_migrate()
    print(f"Database initialized. Legacy reading cards migrated: {migrated}")
    for warning in warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
