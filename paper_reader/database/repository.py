"""Parameterized database operations isolated from Streamlit."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from paper_reader.database.config import sanitize_error
from paper_reader.database.models import Conversation, ConversationMessage, Paper, PaperChunk, QAHistory, ReadingCardRow
from paper_reader.models.schemas import ReadingCard, RetrievedChunk
from paper_reader.workspace import normalize_workspace_id


class RepositoryError(RuntimeError):
    pass


class DuplicatePaperError(RepositoryError):
    def __init__(self, paper_id: str, file_name: str) -> None:
        super().__init__(f"Duplicate paper already exists: {file_name}")
        self.paper_id = paper_id
        self.file_name = file_name


class PaperRepository:
    def __init__(self, factory: sessionmaker[Session], workspace_id: str) -> None:
        normalized = normalize_workspace_id(workspace_id)
        if not normalized:
            raise ValueError("workspace_id must be a valid UUID.")
        self.factory = factory
        self.workspace_id = normalized

    def _owned_paper(self, session: Session, paper_id: str) -> Paper | None:
        return session.execute(
            select(Paper).where(Paper.paper_id == paper_id, Paper.workspace_id == self.workspace_id)
        ).scalar_one_or_none()

    def _owned_conversation(self, session: Session, conversation_id: str) -> Conversation | None:
        return session.execute(
            select(Conversation)
            .join(Paper)
            .where(Conversation.conversation_id == conversation_id, Paper.workspace_id == self.workspace_id)
        ).scalar_one_or_none()

    def list_papers(self) -> list[dict[str, Any]]:
        with self.factory() as session:
            rows = session.execute(select(Paper).where(Paper.workspace_id == self.workspace_id).order_by(Paper.uploaded_at.desc())).scalars().all()
            return [self._paper_to_dict(row) for row in rows]

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        with self.factory() as session:
            paper = self._owned_paper(session, paper_id)
            return self._paper_to_dict(paper) if paper else None

    def get_by_sha256(self, sha256_value: str) -> dict[str, Any] | None:
        with self.factory() as session:
            paper = session.execute(select(Paper).where(Paper.workspace_id == self.workspace_id, Paper.sha256 == sha256_value)).scalar_one_or_none()
            return self._paper_to_dict(paper) if paper else None

    def save_paper_with_chunks(
        self,
        paper_id: str,
        file_name: str,
        sha256_value: str,
        page_count: int,
        parse_status: str,
        chunks: Iterable[dict[str, Any]],
    ) -> bool:
        with self.factory() as session:
            try:
                existing = session.execute(select(Paper).where(Paper.workspace_id == self.workspace_id, Paper.sha256 == sha256_value)).scalar_one_or_none()
                if existing:
                    raise DuplicatePaperError(existing.paper_id, existing.file_name)
                paper = Paper(
                    paper_id=paper_id,
                    file_name=file_name,
                    workspace_id=self.workspace_id,
                    sha256=sha256_value,
                    page_count=page_count,
                    parse_status=parse_status,
                )
                session.add(paper)
                session.flush()
                for item in chunks:
                    session.add(
                        PaperChunk(
                            paper_id=paper_id,
                            page_number=int(item["page_number"]),
                            chunk_index=int(item["chunk_index"]),
                            chunk_text=str(item["chunk_text"]),
                        )
                    )
                session.commit()
                return True
            except DuplicatePaperError:
                session.rollback()
                raise
            except (IntegrityError, SQLAlchemyError) as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def update_parse_status(self, paper_id: str, parse_status: str) -> None:
        with self.factory() as session:
            try:
                paper = self._owned_paper(session, paper_id)
                if paper:
                    paper.parse_status = parse_status
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def delete_paper(self, paper_id: str) -> None:
        self.delete_papers([paper_id])

    def delete_papers(self, paper_ids: Iterable[str]) -> int:
        unique_ids = list(dict.fromkeys(paper_id for paper_id in paper_ids if paper_id))
        if not unique_ids:
            return 0
        with self.factory() as session:
            try:
                papers = session.execute(select(Paper).where(Paper.paper_id.in_(unique_ids), Paper.workspace_id == self.workspace_id)).scalars().all()
                deleted_count = len(papers)
                owned_ids = [paper.paper_id for paper in papers]
                if deleted_count:
                    conversation_ids = (
                        session.execute(select(Conversation.conversation_id).where(Conversation.paper_id.in_(owned_ids)))
                        .scalars()
                        .all()
                    )
                    if conversation_ids:
                        session.execute(
                            delete(ConversationMessage).where(ConversationMessage.conversation_id.in_(conversation_ids))
                        )
                        session.execute(delete(Conversation).where(Conversation.conversation_id.in_(conversation_ids)))
                    session.execute(delete(QAHistory).where(QAHistory.paper_id.in_(owned_ids)))
                    session.execute(delete(ReadingCardRow).where(ReadingCardRow.paper_id.in_(owned_ids)))
                    session.execute(delete(PaperChunk).where(PaperChunk.paper_id.in_(owned_ids)))
                    session.execute(delete(Paper).where(Paper.paper_id.in_(owned_ids), Paper.workspace_id == self.workspace_id))
                session.commit()
                return deleted_count
            except Exception as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def get_chunks(self, paper_id: str) -> list[dict[str, Any]]:
        with self.factory() as session:
            rows = (
                session.execute(
                    select(PaperChunk)
                    .join(Paper)
                    .where(PaperChunk.paper_id == paper_id, Paper.workspace_id == self.workspace_id)
                    .order_by(PaperChunk.page_number, PaperChunk.chunk_index)
                )
                .scalars()
                .all()
            )
            return [
                {
                    "paper_id": row.paper_id,
                    "page_number": row.page_number,
                    "chunk_index": row.chunk_index,
                    "chunk_text": row.chunk_text,
                }
                for row in rows
            ]

    def save_reading_card(
        self,
        paper_id: str,
        card: ReadingCard,
        model_name: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        with self.factory() as session:
            try:
                paper = self._owned_paper(session, paper_id)
                if not paper:
                    raise RepositoryError("Paper not found.")
                existing_rows = (
                    session.execute(
                        select(ReadingCardRow)
                        .join(Paper)
                        .where(ReadingCardRow.paper_id == paper_id, Paper.workspace_id == self.workspace_id)
                        .order_by(ReadingCardRow.generated_at.desc(), ReadingCardRow.id.desc())
                    )
                    .scalars()
                    .all()
                )
                if existing_rows and overwrite:
                    row = existing_rows[0]
                    row.card_json = card.model_dump_json()
                    row.model_name = model_name
                    row.generated_at = datetime.now(UTC)
                    for old_card in existing_rows[1:]:
                        session.delete(old_card)
                elif existing_rows:
                    row = existing_rows[0]
                else:
                    row = ReadingCardRow(
                        paper_id=paper_id,
                        card_json=card.model_dump_json(),
                        model_name=model_name,
                        generated_at=datetime.now(UTC),
                    )
                    session.add(row)
                session.add(row)
                session.commit()
                return self._reading_card_to_dict(row, paper.file_name)
            except RepositoryError:
                session.rollback()
                raise
            except SQLAlchemyError as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def latest_reading_card(self, paper_id: str) -> dict[str, Any] | None:
        with self.factory() as session:
            row = (
                session.execute(
                    select(ReadingCardRow)
                        .join(Paper)
                        .where(ReadingCardRow.paper_id == paper_id, Paper.workspace_id == self.workspace_id)
                    .order_by(ReadingCardRow.generated_at.desc(), ReadingCardRow.id.desc())
                )
                .scalars()
                .first()
            )
            if not row:
                return None
            paper = self._owned_paper(session, paper_id)
            return self._reading_card_to_dict(row, paper.file_name if paper else "")

    def has_reading_card(self, paper_id: str) -> bool:
        with self.factory() as session:
            return (
                session.execute(select(ReadingCardRow.id).join(Paper).where(ReadingCardRow.paper_id == paper_id, Paper.workspace_id == self.workspace_id).limit(1)).first()
                is not None
            )

    def delete_reading_card_for_paper(self, paper_id: str) -> int:
        with self.factory() as session:
            try:
                if not self._owned_paper(session, paper_id):
                    return 0
                result = session.execute(delete(ReadingCardRow).where(ReadingCardRow.paper_id == paper_id))
                session.commit()
                return int(result.rowcount or 0)
            except SQLAlchemyError as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def list_saved_reading_cards(self) -> list[dict[str, Any]]:
        with self.factory() as session:
            rows_by_paper: dict[str, tuple[str, list[ReadingCardRow]]] = {}
            rows = session.execute(
                select(ReadingCardRow, Paper.file_name)
                .join(Paper)
                .where(Paper.workspace_id == self.workspace_id)
                .order_by(Paper.file_name, ReadingCardRow.generated_at, ReadingCardRow.id)
            ).all()
            for row, file_name in rows:
                rows_by_paper.setdefault(row.paper_id, (file_name, []))[1].append(row)

            records = []
            for file_name, paper_rows in rows_by_paper.values():
                first_row = paper_rows[0]
                latest_row = paper_rows[-1]
                record = self._reading_card_to_dict(latest_row, file_name)
                record["created_at"] = first_row.generated_at
                record["updated_at"] = latest_row.generated_at
                records.append(record)
            return records

    def list_latest_reading_cards(self) -> list[dict[str, Any]]:
        return self.list_saved_reading_cards()

    def save_qa_history(
        self,
        paper_id: str,
        question: str,
        answer: str,
        citations: list[RetrievedChunk],
        model_name: str,
    ) -> dict[str, Any]:
        with self.factory() as session:
            try:
                paper = self._owned_paper(session, paper_id)
                if not paper:
                    raise RepositoryError("Paper not found.")
                pages = sorted({citation.page_number for citation in citations})
                snippets = [
                    {
                        "page_number": citation.page_number,
                        "chunk_index": citation.chunk_index,
                        "score": citation.score,
                        "snippet": citation.chunk_text[:900],
                    }
                    for citation in citations
                ]
                row = QAHistory(
                    paper_id=paper_id,
                    question=question,
                    answer=answer,
                    citation_pages=json.dumps(pages),
                    citation_snippets=json.dumps(snippets, ensure_ascii=False),
                    model_name=model_name,
                    generated_at=datetime.now(UTC),
                )
                session.add(row)
                session.commit()
                return self._qa_to_dict(row, paper.file_name)
            except RepositoryError:
                session.rollback()
                raise
            except SQLAlchemyError as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def list_qa_history(self, paper_id: str | None = None) -> list[dict[str, Any]]:
        with self.factory() as session:
            statement = select(QAHistory, Paper.file_name).join(Paper).where(Paper.workspace_id == self.workspace_id)
            if paper_id:
                statement = statement.where(QAHistory.paper_id == paper_id)
            rows = session.execute(statement.order_by(QAHistory.generated_at.desc())).all()
            return [self._qa_to_dict(row, file_name) for row, file_name in rows]

    def create_conversation(self, paper_id: str, title: str = "New chat") -> dict[str, Any]:
        with self.factory() as session:
            try:
                paper = self._owned_paper(session, paper_id)
                if not paper:
                    raise RepositoryError("Paper not found.")
                now = datetime.now(UTC)
                row = Conversation(
                    conversation_id=uuid.uuid4().hex,
                    paper_id=paper_id,
                    title=title.strip()[:255] or "New chat",
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                session.commit()
                return self._conversation_to_dict(row, paper.file_name)
            except RepositoryError:
                session.rollback()
                raise
            except SQLAlchemyError as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def list_conversations(self, paper_id: str) -> list[dict[str, Any]]:
        with self.factory() as session:
            rows = (
                session.execute(
                    select(Conversation, Paper.file_name)
                    .join(Paper)
                    .where(Conversation.paper_id == paper_id, Paper.workspace_id == self.workspace_id)
                    .order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
                )
                .all()
            )
            return [self._conversation_to_dict(row, file_name) for row, file_name in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.factory() as session:
            row = self._owned_conversation(session, conversation_id)
            if not row:
                return None
            paper = session.get(Paper, row.paper_id)
            return self._conversation_to_dict(row, paper.file_name if paper else "")

    def update_conversation_title(self, conversation_id: str, title: str) -> dict[str, Any]:
        clean_title = " ".join(title.strip().split())[:80]
        if not clean_title:
            raise RepositoryError("Chat title cannot be empty.")
        with self.factory() as session:
            try:
                row = self._owned_conversation(session, conversation_id)
                if not row:
                    raise RepositoryError("Chat not found.")
                row.title = clean_title
                row.updated_at = datetime.now(UTC)
                session.commit()
                paper = session.get(Paper, row.paper_id)
                return self._conversation_to_dict(row, paper.file_name if paper else "")
            except RepositoryError:
                session.rollback()
                raise
            except SQLAlchemyError as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def delete_conversation(self, conversation_id: str) -> int:
        with self.factory() as session:
            try:
                row = self._owned_conversation(session, conversation_id)
                if not row:
                    return 0
                session.execute(
                    delete(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id)
                )
                session.delete(row)
                session.commit()
                return 1
            except SQLAlchemyError as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def list_conversation_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.factory() as session:
            rows = (
                session.execute(
                    select(ConversationMessage)
                    .join(Conversation)
                    .join(Paper)
                    .where(ConversationMessage.conversation_id == conversation_id, Paper.workspace_id == self.workspace_id)
                    .order_by(ConversationMessage.turn_id, ConversationMessage.message_id)
                )
                .scalars()
                .all()
            )
            return [self._conversation_message_to_dict(row) for row in rows]

    def add_conversation_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        citations: list[RetrievedChunk],
    ) -> list[dict[str, Any]]:
        with self.factory() as session:
            try:
                conversation = self._owned_conversation(session, conversation_id)
                if not conversation:
                    raise RepositoryError("Chat not found.")
                max_turn = session.execute(
                    select(func.max(ConversationMessage.turn_id)).where(
                        ConversationMessage.conversation_id == conversation_id
                    )
                ).scalar_one()
                turn_id = int(max_turn or 0) + 1
                sources = [
                    {
                        "paper_id": citation.paper_id,
                        "file_name": citation.file_name,
                        "page_number": citation.page_number,
                        "chunk_index": citation.chunk_index,
                        "score": citation.score,
                        "snippet": citation.chunk_text[:900],
                    }
                    for citation in citations
                ]
                now = datetime.now(UTC)
                existing_message_count = session.execute(
                    select(func.count(ConversationMessage.message_id)).where(
                        ConversationMessage.conversation_id == conversation_id
                    )
                ).scalar_one()
                if not existing_message_count:
                    conversation.title = self._title_from_question(question)
                conversation.updated_at = now
                user_row = ConversationMessage(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    role="user",
                    content=question,
                    sources="[]",
                    created_at=now,
                )
                assistant_row = ConversationMessage(
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    role="assistant",
                    content=answer,
                    sources=json.dumps(sources, ensure_ascii=False),
                    created_at=now,
                )
                session.add_all([user_row, assistant_row])
                session.commit()
                return [self._conversation_message_to_dict(user_row), self._conversation_message_to_dict(assistant_row)]
            except RepositoryError:
                session.rollback()
                raise
            except SQLAlchemyError as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def delete_conversation_turn(self, conversation_id: str, turn_id: int) -> int:
        with self.factory() as session:
            try:
                conversation = self._owned_conversation(session, conversation_id)
                if not conversation:
                    return 0
                result = session.execute(
                    delete(ConversationMessage).where(
                        ConversationMessage.conversation_id == conversation_id,
                        ConversationMessage.turn_id == turn_id,
                    )
                )
                conversation.updated_at = datetime.now(UTC)
                session.commit()
                return int(result.rowcount or 0)
            except SQLAlchemyError as exc:
                session.rollback()
                raise RepositoryError(f"Database transaction failed: {sanitize_error(exc)}") from exc

    def migrate_reading_cards_json_detailed(self, json_path: Path) -> dict[str, Any]:
        if not json_path.exists():
            return {"migrated": 0, "skipped": 0, "failed": 0, "reasons": []}
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"migrated": 0, "skipped": 0, "failed": 1, "reasons": [f"Could not read legacy reading_cards.json: {exc}"]}
        if not isinstance(loaded, list):
            return {"migrated": 0, "skipped": 0, "failed": 1, "reasons": ["Legacy reading_cards.json is not a list."]}

        migrated = 0
        skipped = 0
        failed = 0
        reasons: list[str] = []
        with self.factory() as session:
            try:
                for item in loaded:
                    if not isinstance(item, dict):
                        skipped += 1
                        reasons.append("Skipped non-object legacy record.")
                        continue
                    legacy_id = str(item.get("paper_id") or "").strip()
                    file_name = str(item.get("file_name") or "legacy-paper.pdf")
                    if not legacy_id:
                        skipped += 1
                        reasons.append(f"Skipped legacy card without paper_id: {file_name}")
                        continue
                    sha256_value = (
                        legacy_id.lower()
                        if len(legacy_id) == 64 and all(char in "0123456789abcdefABCDEF" for char in legacy_id)
                        else hashlib.sha256(legacy_id.encode("utf-8")).hexdigest()
                    )
                    paper = session.execute(
                        select(Paper).where(
                            Paper.workspace_id == self.workspace_id,
                            Paper.sha256 == sha256_value,
                        )
                    ).scalar_one_or_none()
                    if not paper:
                        paper = Paper(
                            paper_id=str(uuid.uuid4()),
                            file_name=file_name,
                            workspace_id=self.workspace_id,
                            sha256=sha256_value,
                            page_count=0,
                            parse_status="legacy_card_only",
                        )
                        session.add(paper)
                        session.flush()
                    paper_id = paper.paper_id
                    exists = session.execute(
                        select(ReadingCardRow).where(ReadingCardRow.paper_id == paper_id)
                    ).first()
                    if exists:
                        skipped += 1
                        reasons.append(f"Skipped existing legacy card: {file_name}")
                        continue
                    card = ReadingCard(
                        research_question=str(item.get("research_question") or "Not clearly stated in the paper."),
                        method_data=str(item.get("method_data") or "Not clearly stated in the paper."),
                        key_findings=str(item.get("key_findings") or "Not clearly stated in the paper."),
                        limitations=str(item.get("limitations") or "Not clearly stated in the paper."),
                        relevance_takeaway=str(item.get("relevance_takeaway") or "Not clearly stated in the paper."),
                        keywords=[
                            part.strip()
                            for part in str(item.get("keywords") or "").split(",")
                            if part.strip()
                        ],
                    )
                    session.add(
                        ReadingCardRow(
                            paper_id=paper_id,
                            card_json=card.model_dump_json(),
                            model_name=str(item.get("model_name") or "legacy-json"),
                        )
                    )
                    migrated += 1
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                failed += 1
                raise RepositoryError(f"Legacy migration failed: {sanitize_error(exc)}") from exc
        return {"migrated": migrated, "skipped": skipped, "failed": failed, "reasons": reasons}

    def migrate_reading_cards_json(self, json_path: Path) -> tuple[int, list[str]]:
        result = self.migrate_reading_cards_json_detailed(json_path)
        return int(result["migrated"]), list(result["reasons"])

    @staticmethod
    def _paper_to_dict(row: Paper) -> dict[str, Any]:
        return {
            "paper_id": row.paper_id,
            "file_name": row.file_name,
            "sha256": row.sha256,
            "page_count": row.page_count,
            "parse_status": row.parse_status,
            "uploaded_at": row.uploaded_at,
        }

    @staticmethod
    def _reading_card_to_dict(row: ReadingCardRow, file_name: str) -> dict[str, Any]:
        card = json.loads(row.card_json)
        return {
            "id": row.id,
            "paper_id": row.paper_id,
            "file_name": file_name,
            "card": card,
            "model_name": row.model_name,
            "generated_at": row.generated_at,
            "created_at": row.generated_at,
            "updated_at": row.generated_at,
        }

    @staticmethod
    def _qa_to_dict(row: QAHistory, file_name: str) -> dict[str, Any]:
        return {
            "id": row.id,
            "paper_id": row.paper_id,
            "file_name": file_name,
            "question": row.question,
            "answer": row.answer,
            "citation_pages": json.loads(row.citation_pages or "[]"),
            "citation_snippets": json.loads(row.citation_snippets or "[]"),
            "model_name": row.model_name,
            "generated_at": row.generated_at,
        }

    @staticmethod
    def _title_from_question(question: str) -> str:
        clean_question = " ".join(question.strip().split())
        if not clean_question:
            return "New chat"
        if len(clean_question) <= 64:
            return clean_question
        return clean_question[:61].rstrip() + "..."

    @staticmethod
    def _conversation_to_dict(row: Conversation, file_name: str) -> dict[str, Any]:
        return {
            "conversation_id": row.conversation_id,
            "paper_id": row.paper_id,
            "file_name": file_name,
            "title": row.title,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    @staticmethod
    def _conversation_message_to_dict(row: ConversationMessage) -> dict[str, Any]:
        return {
            "message_id": row.message_id,
            "conversation_id": row.conversation_id,
            "turn_id": row.turn_id,
            "role": row.role,
            "content": row.content,
            "sources": json.loads(row.sources or "[]"),
            "created_at": row.created_at,
        }
