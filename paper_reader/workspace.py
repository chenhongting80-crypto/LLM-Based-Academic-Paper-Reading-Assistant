"""Anonymous workspace identifiers shared by UI and database code."""

from __future__ import annotations

import uuid
from collections.abc import Callable

LEGACY_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"


def normalize_workspace_id(value: object) -> str | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value).strip()))
    except (ValueError, TypeError, AttributeError):
        return None


def resolve_workspace_id(
    query_value: object,
    session_value: object = None,
    id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
) -> str:
    if query_value is not None:
        return normalize_workspace_id(query_value) or str(id_factory())
    return normalize_workspace_id(session_value) or str(id_factory())
