"""Pydantic and lightweight domain schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReadingCard(BaseModel):
    research_question: str = "Not clearly stated in the paper."
    method_data: str = "Not clearly stated in the paper."
    key_findings: str = "Not clearly stated in the paper."
    limitations: str = "Not clearly stated in the paper."
    relevance_takeaway: str = "Not clearly stated in the paper."
    keywords: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    paper_id: str
    file_name: str
    page_number: int
    chunk_index: int
    chunk_text: str
    score: float = 0.0


class QAResult(BaseModel):
    answer: str
    citations: list[RetrievedChunk] = Field(default_factory=list)
    evidence_sufficient: bool = True
