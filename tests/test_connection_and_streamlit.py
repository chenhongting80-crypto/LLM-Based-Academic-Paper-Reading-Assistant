from __future__ import annotations

import importlib
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from paper_reader.database.config import DatabaseConfig
from paper_reader.database.session import check_database_connection
from tests.helpers import ensure_safe_test_database


class ConnectionAndStreamlitTests(unittest.TestCase):
    def test_database_settings_must_not_be_empty(self) -> None:
        settings = {
            "MYSQL_HOST": "localhost",
            "MYSQL_PORT": "3306",
            "MYSQL_DATABASE": "paper_reader",
            "MYSQL_USER": "app_user",
            "MYSQL_PASSWORD": " ",
        }
        with patch.dict(os.environ, settings, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MYSQL_PASSWORD"):
                DatabaseConfig.from_env()

    def test_mysql_port_must_be_an_integer(self) -> None:
        settings = {
            "MYSQL_HOST": "localhost",
            "MYSQL_PORT": "not-a-port",
            "MYSQL_DATABASE": "paper_reader",
            "MYSQL_USER": "app_user",
            "MYSQL_PASSWORD": "password",
        }
        with patch.dict(os.environ, settings, clear=True):
            with self.assertRaisesRegex(RuntimeError, "valid integer"):
                DatabaseConfig.from_env()

    def test_mysql_port_must_be_in_range(self) -> None:
        settings = {
            "MYSQL_HOST": "localhost",
            "MYSQL_PORT": "70000",
            "MYSQL_DATABASE": "paper_reader",
            "MYSQL_USER": "app_user",
            "MYSQL_PASSWORD": "password",
        }
        with patch.dict(os.environ, settings, clear=True):
            with self.assertRaisesRegex(RuntimeError, "between 1 and 65535"):
                DatabaseConfig.from_env()

    def test_repository_connection_failure_is_rechecked(self) -> None:
        with patch.dict(os.environ, {"PAPER_READER_SKIP_STREAMLIT_UI": "1"}):
            main = importlib.import_module("main")
        engine = object()
        repository = object()
        with (
            patch.object(main, "get_database_engine", return_value=engine),
            patch.object(
                main,
                "check_database_connection",
                side_effect=[(False, "unavailable"), (True, "connected")],
            ) as check_connection,
            patch.object(main, "init_database"),
            patch.object(main, "session_factory", return_value=object()),
            patch.object(main, "PaperRepository", return_value=repository),
        ):
            first = main.get_repository()
            second = main.get_repository()

        self.assertEqual(first, (None, "unavailable", False))
        self.assertEqual(second, (repository, "connected", True))
        self.assertEqual(check_connection.call_count, 2)

    def test_mysql_test_database_must_be_distinct(self) -> None:
        url = URL.create("mysql+pymysql", database="production_test")
        with patch.dict(os.environ, {"MYSQL_DATABASE": "production_test"}):
            with self.assertRaisesRegex(RuntimeError, "must differ"):
                ensure_safe_test_database(url)

    def test_mysql_test_database_name_must_identify_test_database(self) -> None:
        url = URL.create("mysql+pymysql", database="paper_reader_staging")
        with patch.dict(os.environ, {"MYSQL_DATABASE": "paper_reader"}):
            with self.assertRaisesRegex(RuntimeError, "must contain 'test'"):
                ensure_safe_test_database(url)

    def test_mysql_connection_failure_is_sanitized(self) -> None:
        engine = create_engine(
            URL.create(
                "mysql+pymysql",
                username="paper_reader_user",
                password="secret_password",
                host="127.0.0.1",
                port=1,
                database="missing",
            ),
            pool_pre_ping=True,
            future=True,
        )
        ok, message = check_database_connection(engine)
        self.assertFalse(ok)
        self.assertNotIn("secret_password", message)
        self.assertNotIn("paper_reader_user", message)

    def test_streamlit_module_imports_without_running_ui(self) -> None:
        env = dict(os.environ)
        env["PAPER_READER_SKIP_STREAMLIT_UI"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", "import main; print('ok')"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
