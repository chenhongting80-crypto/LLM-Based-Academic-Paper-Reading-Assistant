"""Reading-card generation from a bounded set of chunks."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from paper_reader.database.repository import PaperRepository
from paper_reader.llm.client import invoke_text
from paper_reader.llm.parsing import parse_json_model
from paper_reader.llm.prompts import LOCAL_CHUNK_SUMMARY_PROMPT, READING_CARD_PROMPT, READING_CARD_REPAIR_PROMPT
from paper_reader.models.schemas import ReadingCard


def _chunk_label(chunk: dict[str, Any]) -> str:
    return f"page {chunk['page_number']}, chunk {chunk['chunk_index']}"


def local_chunk_summaries(
    file_name: str,
    chunks: list[dict[str, Any]],
    llm: ChatOpenAI | None,
    max_chunks: int = 24,
) -> tuple[list[str], str | None]:
    if not chunks:
        return [], "No chunks are available for reading-card generation."
    selected_chunks = chunks[:max_chunks]
    summaries: list[str] = []
    for chunk in selected_chunks:
        summary, error = invoke_text(
            LOCAL_CHUNK_SUMMARY_PROMPT,
            llm,
            {
                "source": file_name,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
            },
        )
        if error:
            return summaries, error
        summaries.append(f"[{_chunk_label(chunk)}]\n{summary.strip()}")
    return summaries, None


def generate_reading_card_from_chunks(
    file_name: str,
    chunks: list[dict[str, Any]],
    llm: ChatOpenAI | None,
) -> tuple[ReadingCard | None, str | None, list[str]]:
    summaries, error = local_chunk_summaries(file_name, chunks, llm)
    if error:
        return None, error, summaries

    raw, error = invoke_text(
        READING_CARD_PROMPT,
        llm,
        {
            "source": file_name,
            "chunk_summaries": "\n\n---\n\n".join(summaries),
        },
    )
    if error:
        return None, error, summaries

    card, parse_error = parse_json_model(raw, ReadingCard)
    if card:
        return card, None, summaries

    repaired, repair_error = invoke_text(
        READING_CARD_REPAIR_PROMPT,
        llm,
        {"original_output": raw},
    )
    if repair_error:
        return None, parse_error, summaries
    card, _ = parse_json_model(repaired, ReadingCard)
    if not card:
        return None, "LLM output could not be parsed as the required JSON object.", summaries
    return card, None, summaries


def save_or_replace_reading_card(
    repository: PaperRepository,
    paper_id: str,
    card: ReadingCard,
    model_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    return repository.save_reading_card(paper_id, card, model_name=model_name, overwrite=overwrite)


def reading_card_to_markdown(card: ReadingCard | dict[str, Any]) -> str:
    if isinstance(card, dict):
        data = card
    else:
        data = card.model_dump()
    keywords = data.get("keywords") or []
    if isinstance(keywords, list):
        keyword_text = ", ".join(str(item) for item in keywords)
    else:
        keyword_text = str(keywords)
    return "\n\n".join(
        [
            "# Research Question\n" + str(data.get("research_question", "")),
            "# Method / Data\n" + str(data.get("method_data", "")),
            "# Key Findings\n" + str(data.get("key_findings", "")),
            "# Limitations\n" + str(data.get("limitations", "")),
            "# Relevance / Takeaway\n" + str(data.get("relevance_takeaway", "")),
            "# Keywords\n" + keyword_text,
        ]
    )
