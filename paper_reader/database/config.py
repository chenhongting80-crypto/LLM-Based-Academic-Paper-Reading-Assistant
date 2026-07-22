"""Environment-driven database configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

REQUIRED_DATABASE_KEYS = (
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_DATABASE",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
)


def database_env() -> dict[str, str]:
    missing = [key for key in REQUIRED_DATABASE_KEYS if not os.getenv(key, "").strip()]
    if missing:
        raise RuntimeError(f"Missing required database settings: {', '.join(missing)}")
    return {key: os.environ[key] for key in REQUIRED_DATABASE_KEYS}


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        values = database_env()
        try:
            port = int(values["MYSQL_PORT"])
        except ValueError as exc:
            raise RuntimeError("MYSQL_PORT must be a valid integer.") from exc
        if not 1 <= port <= 65535:
            raise RuntimeError("MYSQL_PORT must be between 1 and 65535.")
        return cls(
            host=values["MYSQL_HOST"],
            port=port,
            database=values["MYSQL_DATABASE"],
            user=values["MYSQL_USER"],
            password=values["MYSQL_PASSWORD"],
        )

    def sqlalchemy_url(self) -> URL:
        return URL.create(
            "mysql+pymysql",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            query={"charset": "utf8mb4"},
        )

    def safe_label(self) -> str:
        return f"MySQL host={self.host} port={self.port} database={self.database}"


def database_url_from_env() -> URL:
    return DatabaseConfig.from_env().sqlalchemy_url()


def safe_database_label() -> str:
    return DatabaseConfig.from_env().safe_label()


def sanitize_error(exc: Exception) -> str:
    message = str(exc)
    for secret in [os.getenv("MYSQL_PASSWORD", ""), os.getenv("OPENAI_API_KEY", "")]:
        if secret:
            message = message.replace(secret, "***")
    message = re.sub(r":([^:@/]+)@", r":***@", message)
    message = re.sub(r"user '.*?'", "user '***'", message, flags=re.IGNORECASE)
    message = re.sub(r"\([^)]+://[^)]+\)", "(connection string hidden)", message)
    return message.splitlines()[0][:300]
