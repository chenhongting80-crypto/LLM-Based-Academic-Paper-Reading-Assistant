"""Shared test helpers."""

from __future__ import annotations

import os
import tempfile
import unittest

import fitz
from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url

from paper_reader.database.models import Base
from paper_reader.database.repository import PaperRepository
from paper_reader.database.session import create_app_engine, init_database, session_factory

load_dotenv()


def make_pdf_bytes(page_texts: list[str]) -> bytes:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    return doc.tobytes()


def test_database_url() -> str | URL:
    explicit_url = os.getenv("PAPER_READER_TEST_DATABASE_URL", "").strip()
    if explicit_url:
        return explicit_url
    test_database = os.getenv("MYSQL_TEST_DATABASE", "").strip()
    if test_database:
        return URL.create(
            "mysql+pymysql",
            username=os.getenv("MYSQL_USER", "paper_reader_user"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=test_database,
            query={"charset": "utf8mb4"},
        )
    temp_dir = tempfile.mkdtemp()
    return f"sqlite:///{temp_dir}/paper_reader_test.sqlite"


def ensure_safe_test_database(database_url: str | URL) -> None:
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        return
    test_database = (url.database or "").lower()
    configured_database = os.getenv("MYSQL_DATABASE", "").strip().lower()
    if not test_database or "test" not in test_database:
        raise RuntimeError("Refusing to drop tables: the MySQL test database name must contain 'test'.")
    if configured_database and test_database == configured_database:
        raise RuntimeError("Refusing to drop tables: MYSQL_TEST_DATABASE must differ from MYSQL_DATABASE.")


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        database_url = test_database_url()
        ensure_safe_test_database(database_url)
        self.engine = create_app_engine(database_url)
        Base.metadata.drop_all(self.engine)
        init_database(self.engine)
        self.repository = PaperRepository(session_factory(self.engine))

    def tearDown(self) -> None:
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()
