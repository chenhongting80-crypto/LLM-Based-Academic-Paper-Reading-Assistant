"""Grounded question-answering service."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from paper_reader.database.repository import PaperRepository
from paper_reader.llm.client import invoke_text
from paper_reader.llm.prompts import QA_PROMPT
from paper_reader.models.schemas import QAResult, RetrievedChunk
from paper_reader.retrieval.tfidf import TfidfRetriever

INSUFFICIENT_EVIDENCE_MESSAGE = "The current paper does not contain enough evidence to answer this question."

AUTHOR_TERMS = ("author", "authors", "affiliation", "affiliations")
CONCLUSION_TERMS = ("conclusion", "conclusions", "concluding", "discussion", "summary")


def format_context(citations: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(
        (
            f"Source: {item.file_name}\n"
            f"Page: {item.page_number}\n"
            f"Chunk: {item.chunk_index}\n"
            f"Score: {item.score:.3f}\n"
            f"Text: {item.chunk_text}"
        )
        for item in citations
    )


def _chunk_to_retrieved(chunk: dict[str, Any], score: float) -> RetrievedChunk:
    return RetrievedChunk(
        paper_id=str(chunk.get("paper_id", "")),
        file_name=str(chunk.get("file_name", "")),
        page_number=int(chunk.get("page_number", 0)),
        chunk_index=int(chunk.get("chunk_index", 0)),
        chunk_text=str(chunk.get("chunk_text", "")),
        score=score,
    )


def _merge_citations(primary: list[RetrievedChunk], extra: list[RetrievedChunk], limit: int = 8) -> list[RetrievedChunk]:
    merged: list[RetrievedChunk] = []
    seen: set[tuple[str, int, int]] = set()
    for item in [*primary, *extra]:
        key = (item.paper_id, item.page_number, item.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def augment_retrieval_for_question(
    question: str,
    chunks: list[dict[str, Any]],
    citations: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    lowered = question.lower()
    extras: list[RetrievedChunk] = []

    if any(term in lowered for term in AUTHOR_TERMS):
        first_pages = sorted({int(chunk.get("page_number", 0)) for chunk in chunks})[:2]
        for chunk in chunks:
            if int(chunk.get("page_number", 0)) in first_pages:
                extras.append(_chunk_to_retrieved(chunk, score=0.02))

    if any(term in lowered for term in CONCLUSION_TERMS):
        conclusion_chunks = [
            chunk
            for chunk in chunks
            if any(term in str(chunk.get("chunk_text", "")).lower() for term in CONCLUSION_TERMS)
        ]
        tail_chunks = chunks[-6:]
        extras.extend(_chunk_to_retrieved(chunk, score=0.02) for chunk in [*conclusion_chunks, *tail_chunks])

    return _merge_citations(citations, extras)


def answer_from_retrieved_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    llm: ChatOpenAI | None,
    chat_history: list[dict[str, str]] | None = None,
    top_k: int = 5,
    min_score: float = 0.03,
) -> tuple[QAResult, str | None]:
    if not question.strip():
        return QAResult(answer="", evidence_sufficient=False), "Enter a question first."
    if not chunks:
        return QAResult(answer=INSUFFICIENT_EVIDENCE_MESSAGE, evidence_sufficient=False), None

    retriever = TfidfRetriever(chunks)
    try:
        retriever.build()
    except ValueError:
        return QAResult(answer=INSUFFICIENT_EVIDENCE_MESSAGE, evidence_sufficient=False), None

    citations = retriever.search(question, top_k=top_k, min_score=min_score)
    citations = augment_retrieval_for_question(question, chunks, citations)
    if not citations:
        return QAResult(answer=INSUFFICIENT_EVIDENCE_MESSAGE, citations=[], evidence_sufficient=False), None

    history_text = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
        for turn in (chat_history or [])[-6:]
        if turn.get("content")
    )
    answer, error = invoke_text(
        QA_PROMPT,
        llm,
        {
            "question": question.strip(),
            "chat_history": history_text or "No prior conversation.",
            "context": format_context(citations),
        },
    )
    if error:
        return QAResult(answer="", citations=citations, evidence_sufficient=True), error
    return QAResult(answer=answer, citations=citations, evidence_sufficient=True), None


def ask_in_conversation(
    repository: PaperRepository,
    paper_id: str,
    conversation_id: str,
    question: str,
    chunks: list[dict[str, Any]],
    llm: ChatOpenAI | None,
    model_name: str,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[QAResult, str | None]:
    result, error = answer_from_retrieved_chunks(question, chunks, llm, chat_history)
    if error:
        return result, error
    repository.add_conversation_turn(
        conversation_id=conversation_id,
        question=question.strip(),
        answer=result.answer,
        citations=result.citations,
    )
    repository.save_qa_history(
        paper_id=paper_id,
        question=question.strip(),
        answer=result.answer,
        citations=result.citations,
        model_name=model_name,
    )
    return result, None
