"""Paper comparison service."""

from __future__ import annotations

from typing import Any

NOT_AVAILABLE = "Not available"

COMPARISON_DIMENSIONS = [
    ("Citation / Paper Information", "citation"),
    ("Research Question", "research_question"),
    ("Methods / Data", "method_data"),
    ("Key Findings", "key_findings"),
    ("Limitations", "limitations"),
    ("Environmental Engineering Relevance", "relevance_takeaway"),
    ("Keywords", "keywords"),
]


def _clean_value(value: Any) -> str:
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, list):
        value = ", ".join(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return text or NOT_AVAILABLE


def comparison_value(item: dict[str, Any], field: str) -> str:
    card = item.get("card", {}) or {}
    if field == "citation":
        parts = [
            f"File: {_clean_value(item.get('file_name'))}",
            f"Model: {_clean_value(item.get('model_name'))}",
            f"Generated: {_clean_value(item.get('generated_at'))}",
        ]
        return "\n".join(parts)
    return _clean_value(card.get(field))


def build_detailed_comparison(reading_cards: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    detailed: dict[str, list[dict[str, str]]] = {}
    for label, field in COMPARISON_DIMENSIONS:
        detailed[label] = [
            {
                "paper_id": str(item.get("paper_id", "")),
                "file_name": str(item.get("file_name", "Paper")),
                "value": comparison_value(item, field),
            }
            for item in reading_cards
        ]
    return detailed


def comparison_prompt_context(reading_cards: list[dict[str, Any]]) -> str:
    sections = []
    for index, item in enumerate(reading_cards, start=1):
        card = item.get("card", {}) or {}
        keywords = _clean_value(card.get("keywords"))
        sections.append(
            "\n".join(
                [
                    f"Paper {index}: {_clean_value(item.get('file_name'))}",
                    f"Research Question: {_clean_value(card.get('research_question'))}",
                    f"Methods / Data: {_clean_value(card.get('method_data'))}",
                    f"Key Findings: {_clean_value(card.get('key_findings'))}",
                    f"Limitations: {_clean_value(card.get('limitations'))}",
                    f"Environmental Engineering Relevance: {_clean_value(card.get('relevance_takeaway'))}",
                    f"Keywords: {keywords}",
                ]
            )
        )
    return "\n\n---\n\n".join(sections)
