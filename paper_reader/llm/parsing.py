"""Structured output parsing helpers."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from paper_reader.models.schemas import ReadingCard

ModelT = TypeVar("ModelT", bound=BaseModel)
logger = logging.getLogger(__name__)

READING_CARD_TEXT_FIELDS = (
    "research_question",
    "method_data",
    "key_findings",
    "limitations",
    "relevance_takeaway",
)


def _clear_text(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_clear_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "; ".join(_clear_text(item) for item in value if item is not None)
    return str(value)


def normalize_reading_card_data(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    for field_name in READING_CARD_TEXT_FIELDS:
        value = normalized.get(field_name)
        if value is None:
            normalized.pop(field_name, None)
        elif isinstance(value, (list, dict)):
            normalized[field_name] = _clear_text(value)

    keywords = normalized.get("keywords")
    if keywords is None:
        normalized.pop("keywords", None)
    elif isinstance(keywords, str):
        normalized["keywords"] = [part.strip() for part in keywords.split(",") if part.strip()]
    elif isinstance(keywords, list):
        normalized["keywords"] = [_clear_text(item).strip() for item in keywords if item is not None and _clear_text(item).strip()]
    elif isinstance(keywords, dict):
        normalized["keywords"] = [_clear_text(keywords)]
    else:
        normalized["keywords"] = [str(keywords)]
    return normalized


def _json_object(raw_text: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data


def _safe_output_preview(raw_text: str, limit: int = 500) -> str:
    preview = " ".join(raw_text.split())[:limit]
    preview = re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-[REDACTED]", preview)
    preview = re.sub(r"(?i)(api[_ -]?key|password)(\s*[:=]\s*)[^,\s}\"]+", r"\1\2[REDACTED]", preview)
    return re.sub(r"(?i)(mysql(?:\+pymysql)?://[^: /]+:)[^@ /]+@", r"\1[REDACTED]@", preview)


def _log_parse_failure(exc: Exception, raw_text: str) -> None:
    if isinstance(exc, ValidationError):
        details = ", ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['type']}"
            for error in exc.errors()
        )
    else:
        details = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    logger.warning("Reading Card parse failed (%s); output=%r", details, _safe_output_preview(raw_text))


def parse_json_model(raw_text: str, model_type: type[ModelT]) -> tuple[ModelT | None, str | None]:
    try:
        data = _json_object(raw_text)
        if issubclass(model_type, ReadingCard):
            data = normalize_reading_card_data(data)
        return model_type.model_validate(data), None
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        _log_parse_failure(exc, raw_text)
        return None, "LLM output could not be parsed as the required JSON object."
