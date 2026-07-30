"""Paper deletion business logic."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any

from paper_reader.database.repository import PaperRepository


def selected_paper_names(papers: list[dict[str, Any]], selected_ids: Iterable[str]) -> list[str]:
    selected = set(selected_ids)
    return [paper["file_name"] for paper in papers if paper["paper_id"] in selected]


def delete_selected_papers(repository: PaperRepository, selected_ids: Iterable[str], confirmed: bool) -> int:
    if not confirmed:
        return 0
    return repository.delete_papers(selected_ids)


def apply_successful_deletion_state(
    state: MutableMapping[str, Any],
    deleted_ids: Iterable[str],
    remaining_papers: list[dict[str, Any]],
) -> None:
    deleted = set(deleted_ids)
    state["pending_delete_paper_ids"] = []
    state["paper_delete_selection_reset"] = int(state.get("paper_delete_selection_reset", 0)) + 1
    state["reading_card_generation_summary"] = None
    state["reading_card_action_message"] = ""
    state["reading_card_selected_paper_ids"] = []
    state["reading_card_selected_paper_names"] = []
    if state.get("selected_paper_id") in deleted:
        state["selected_paper_id"] = remaining_papers[0]["paper_id"] if remaining_papers else ""
