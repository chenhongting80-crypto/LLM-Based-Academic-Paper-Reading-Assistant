"""Replaceable TF-IDF retrieval backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from paper_reader.models.schemas import RetrievedChunk


@dataclass
class TfidfRetriever:
    chunks: list[dict[str, Any]]
    vectorizer: TfidfVectorizer | None = None
    matrix: Any = None
    searchable_chunks: list[dict[str, Any]] = field(default_factory=list, init=False)

    def build(self) -> None:
        searchable = [
            (chunk, text)
            for chunk in self.chunks
            if (text := str(chunk.get("chunk_text", "")).strip())
        ]
        self.searchable_chunks = [chunk for chunk, _ in searchable]
        texts = [text for _, text in searchable]
        if not texts:
            raise ValueError("No searchable text chunks were found.")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.03) -> list[RetrievedChunk]:
        if not query.strip() or self.vectorizer is None or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = similarities.argsort()[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for index in ranked_indices:
            score = float(similarities[index])
            if score < min_score:
                continue
            chunk = self.searchable_chunks[index]
            results.append(
                RetrievedChunk(
                    paper_id=str(chunk.get("paper_id", "")),
                    file_name=str(chunk.get("file_name", "")),
                    page_number=int(chunk.get("page_number", 0)),
                    chunk_index=int(chunk.get("chunk_index", 0)),
                    chunk_text=str(chunk.get("chunk_text", "")),
                    score=score,
                )
            )
        return results
