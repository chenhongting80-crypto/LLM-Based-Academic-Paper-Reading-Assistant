"""Streamlit UI for LLM-Based Academic Paper Reading Assistant."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from paper_reader.database.config import sanitize_error
from paper_reader.database.repository import PaperRepository, RepositoryError
from paper_reader.database.session import (
    check_database_connection,
    create_app_engine,
    init_database,
    session_factory,
)
from paper_reader.exporting.exporters import (
    comparison_to_markdown,
    dataframe_to_csv,
    dicts_to_json,
    json_to_zip,
    markdown_to_docx,
    markdown_to_pdf,
    report_to_markdown,
)
from paper_reader.llm.client import get_llm, invoke_text, select_chat_model
from paper_reader.services.comparison import build_detailed_comparison, comparison_prompt_context
from paper_reader.services.deletion import apply_successful_deletion_state, delete_selected_papers, selected_paper_names
from paper_reader.services.papers import ingest_pdf, paper_chunks_for_retrieval
from paper_reader.services.qa import ask_in_conversation
from paper_reader.services.reading_cards import (
    generate_reading_card_from_chunks,
    reading_card_to_markdown,
    save_or_replace_reading_card,
)
from paper_reader.ui.styles import APP_CSS
from paper_reader.workspace import LEGACY_WORKSPACE_ID, normalize_workspace_id, resolve_workspace_id

APP_SUBTITLE = "Persistent, source-grounded academic reading support for papers."
logger = logging.getLogger(__name__)
CHAT_TITLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Create a concise title for an academic paper Q&A chat. "
            "Use 4-10 English words. Do not use quotes, periods, or filler prefixes.",
        ),
        (
            "human",
            """Conversation:
{conversation}

Return only the title.
""",
        ),
    ]
)
COMPARISON_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Compare environmental engineering papers using only the provided reading-card data. "
            "Do not add outside facts. If information is insufficient, say so clearly.",
        ),
        (
            "human",
            """Reading-card data:
{comparison_context}

Write a concise comparison summary with these headings and bullet points:

Key Similarities
Key Differences
Methodological Differences
Differences in Findings
Common Limitations
Overall Environmental Engineering Relevance
""",
        ),
    ]
)


@st.cache_resource(show_spinner=False)
def get_database_engine():
    return create_app_engine()


def get_repository(workspace_id: str = LEGACY_WORKSPACE_ID) -> tuple[PaperRepository | None, str, bool]:
    try:
        engine = get_database_engine()
        ok, message = check_database_connection(engine)
        if not ok:
            return None, message, False
        init_database(engine)
        return PaperRepository(session_factory(engine), workspace_id), message, True
    except Exception as exc:
        detail = sanitize_error(exc)
        logger.error("Database initialization failed: %s: %s", type(exc).__name__, detail)
        return None, "Database connection is unavailable.", False


def init_state() -> None:
    defaults = {
        "last_upload_messages": [],
        "selected_paper_id": "",
        "pending_delete_paper_ids": [],
        "paper_delete_selection_reset": 0,
        "reading_card_selected_paper_ids": [],
        "reading_card_selected_paper_names": [],
        "reading_card_dialog_reset": 0,
        "reading_card_generation_summary": None,
        "reading_card_action_message": "",
        "qna_selected_paper_id": "",
        "qna_selected_paper_name": "",
        "qna_active_conversation_id": "",
        "qna_active_conversation_title": "",
        "qna_action_message": "",
        "qna_expanded_project_id": "",
        "qna_pending_new_chat": False,
        "compare_selected_paper_ids": [],
        "compare_dialog_reset": 0,
        "compare_result": None,
        "export_step": 1,
        "export_content_type": "",
        "export_selected_ids": [],
        "export_format": "",
        "export_options": {},
        "export_generated_file": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_WORKSPACE_STATE_KEYS = (
    "last_upload_messages",
    "selected_paper_id",
    "pending_delete_paper_ids",
    "reading_card_selected_paper_ids",
    "reading_card_selected_paper_names",
    "reading_card_generation_summary",
    "reading_card_action_message",
    "qna_selected_paper_id",
    "qna_selected_paper_name",
    "qna_active_conversation_id",
    "qna_active_conversation_title",
    "qna_action_message",
    "qna_expanded_project_id",
    "qna_pending_new_chat",
    "compare_selected_paper_ids",
    "compare_result",
    "export_selected_ids",
    "export_generated_file",
)


def initialize_workspace() -> str:
    query_value = st.query_params.get("workspace")
    previous = normalize_workspace_id(st.session_state.get("workspace_id"))
    workspace_id = resolve_workspace_id(query_value, previous)
    if previous and previous != workspace_id:
        for key in _WORKSPACE_STATE_KEYS:
            st.session_state.pop(key, None)
        init_state()
    st.session_state["workspace_id"] = workspace_id
    if normalize_workspace_id(query_value) != workspace_id:
        st.query_params["workspace"] = workspace_id
    return workspace_id


def card_dataframe(cards: list[dict]) -> pd.DataFrame:
    rows = []
    for item in cards:
        card = item.get("card", {})
        rows.append(
            {
                "Paper": item.get("file_name", ""),
                "Research Question": card.get("research_question", ""),
                "Method / Data": card.get("method_data", ""),
                "Key Findings": card.get("key_findings", ""),
                "Limitations": card.get("limitations", ""),
                "Relevance / Takeaway": card.get("relevance_takeaway", ""),
                "Keywords": ", ".join(card.get("keywords", [])) if isinstance(card.get("keywords"), list) else card.get("keywords", ""),
                "Model": item.get("model_name", ""),
                "Generated At": item.get("generated_at", ""),
            }
        )
    return pd.DataFrame(rows)


def render_source_snippets(citations: list[dict]) -> None:
    if not citations:
        st.info("No citation snippets were retrieved.")
        return
    for citation in citations:
        st.markdown(
            f"**Page {citation.get('page_number')} - chunk {citation.get('chunk_index')} (score {float(citation.get('score', 0.0)):.3f})**"
        )
        st.write(citation.get("snippet", ""))


def format_timestamp(value: object) -> str:
    if not value:
        return "Not available"
    return str(value).split(".")[0]


def render_name_list(names: list[str]) -> None:
    for name in names:
        st.write(f"- {name}")


def compact_label(text: str, limit: int = 86) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def set_compare_selection(selected_ids: list[str]) -> None:
    st.session_state["compare_selected_paper_ids"] = selected_ids
    st.session_state["compare_result"] = None


def selected_compare_cards(saved_cards: list[dict]) -> list[dict]:
    lookup = {item["paper_id"]: item for item in saved_cards}
    return [lookup[paper_id] for paper_id in st.session_state.get("compare_selected_paper_ids", []) if paper_id in lookup]


def render_detailed_comparison(detailed: dict[str, list[dict[str, str]]], selected_count: int) -> None:
    default_open = {"Research Question", "Methods / Data", "Key Findings"}
    for dimension, rows in detailed.items():
        with st.expander(dimension, expanded=dimension in default_open):
            if selected_count == 2:
                cols = st.columns(2)
                for col, row in zip(cols, rows, strict=False):
                    col.markdown(f"**{row.get('file_name', 'Paper')}**")
                    col.write(row.get("value", "Not available"))
            else:
                for row in rows:
                    st.markdown(f"**{row.get('file_name', 'Paper')}**")
                    st.write(row.get("value", "Not available"))


EXPORT_FORMATS = {
    "Reading Cards": ["PDF", "DOCX", "Markdown", "CSV", "JSON"],
    "Paper Comparison": ["PDF", "DOCX", "Markdown", "CSV"],
    "Q&A Chats": ["PDF", "DOCX", "Markdown", "JSON"],
    "Full Library Report": ["PDF", "DOCX", "Markdown"],
    "Metadata and Generated Results Export": ["JSON", "ZIP"],
}


def reset_export_state() -> None:
    st.session_state["export_step"] = 1
    st.session_state["export_content_type"] = ""
    st.session_state["export_selected_ids"] = []
    st.session_state["export_format"] = ""
    st.session_state["export_options"] = {}
    st.session_state["export_generated_file"] = None


def export_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def qna_chat_records(repository: PaperRepository, papers: list[dict]) -> list[dict]:
    records = []
    for paper in papers:
        for conversation in repository.list_conversations(paper["paper_id"]):
            records.append(
                {
                    "conversation_id": conversation["conversation_id"],
                    "paper_id": paper["paper_id"],
                    "paper_name": paper["file_name"],
                    "title": conversation["title"],
                    "updated_at": conversation["updated_at"],
                    "messages": repository.list_conversation_messages(conversation["conversation_id"]),
                }
            )
    return records


def export_selected_records(content_type: str, selected_ids: list[str], cards: list[dict], chats: list[dict]) -> list[dict]:
    if content_type == "Reading Cards":
        lookup = {str(item["paper_id"]): item for item in cards}
        return [lookup[item_id] for item_id in selected_ids if item_id in lookup]
    if content_type == "Q&A Chats":
        lookup = {str(item["conversation_id"]): item for item in chats}
        return [lookup[item_id] for item_id in selected_ids if item_id in lookup]
    return []


def export_markdown(
    content_type: str,
    records: list[dict],
    cards: list[dict],
    qa_rows: list[dict],
    comparison_result: dict | None,
    options: dict,
) -> str:
    include_sources = bool(options.get("include_sources", True))
    include_timestamps = bool(options.get("include_timestamps", True))
    if content_type == "Reading Cards":
        return report_to_markdown(records, [])
    if content_type == "Paper Comparison":
        return comparison_to_markdown(comparison_result or {})
    if content_type == "Q&A Chats":
        parts = ["# Q&A Chats"]
        for chat in records:
            parts.append(f"## {chat.get('title', 'Chat')}\n**Paper:** {chat.get('paper_name', '')}")
            if include_timestamps:
                parts.append(f"Updated: {chat.get('updated_at', '')}")
            for message in chat.get("messages", []):
                parts.append(f"### {message.get('role', '').title()}\n{message.get('content', '')}")
                if include_sources and message.get("sources"):
                    source_lines = [
                        f"- Page {source.get('page_number')}: {source.get('snippet', '')[:300]}" for source in message.get("sources", [])
                    ]
                    parts.append("Sources:\n" + "\n".join(source_lines))
        return "\n\n".join(parts)
    if content_type == "Full Library Report":
        return report_to_markdown(cards, qa_rows)
    return "# Metadata and Generated Results Export\n\nUse JSON or ZIP for metadata and generated results."


def export_filename(content_type: str, export_format: str) -> str:
    slug = content_type.lower().replace("&", "and").replace(" ", "_")
    extension = {"Markdown": "md"}.get(export_format, export_format.lower())
    return f"{slug}_{export_timestamp()}.{extension}"


def build_export_file(
    content_type: str,
    export_format: str,
    records: list[dict],
    cards: list[dict],
    qa_rows: list[dict],
    comparison_result: dict | None,
    chats: list[dict],
    papers: list[dict],
    options: dict,
) -> dict:
    if content_type == "Metadata and Generated Results Export":
        payload = {
            "papers": papers,
            "reading_cards": cards,
            "qa_history": qa_rows,
            "qna_chats": chats,
            "paper_comparison": comparison_result,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        if export_format == "ZIP":
            data = json_to_zip({"metadata_and_generated_results.json": payload})
            mime = "application/zip"
        else:
            data = dicts_to_json(payload)
            mime = "application/json"
        return {"file_name": export_filename(content_type, export_format), "data": data, "mime": mime}

    markdown = export_markdown(content_type, records, cards, qa_rows, comparison_result, options)
    if export_format == "PDF":
        data = markdown_to_pdf(markdown, title=content_type)
        mime = "application/pdf"
    elif export_format == "DOCX":
        data = markdown_to_docx(markdown)
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif export_format == "Markdown":
        data = markdown
        mime = "text/markdown"
    elif export_format == "JSON":
        data = dicts_to_json({"content_type": content_type, "records": records, "generated_at": datetime.now(UTC).isoformat()})
        mime = "application/json"
    elif export_format == "CSV":
        if content_type == "Reading Cards":
            data = dataframe_to_csv(card_dataframe(records))
        elif content_type == "Paper Comparison" and comparison_result:
            rows = [
                {"Dimension": dimension, "Paper": row.get("file_name"), "Value": row.get("value")}
                for dimension, values in (comparison_result.get("detailed", {}) or {}).items()
                for row in values
            ]
            data = dataframe_to_csv(pd.DataFrame(rows))
        else:
            data = ""
        mime = "text/csv"
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
    return {"file_name": export_filename(content_type, export_format), "data": data, "mime": mime}


def file_size_label(data: bytes | str) -> str:
    size = len(data.encode("utf-8")) if isinstance(data, str) else len(data)
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def clear_active_qna_state() -> None:
    for key, value in {
        "qna_active_conversation_id": "",
        "qna_active_conversation_title": "",
        "qna_pending_new_chat": False,
    }.items():
        st.session_state[key] = value


def clear_qna_project_selection() -> None:
    st.session_state["qna_selected_paper_id"] = ""
    st.session_state["qna_selected_paper_name"] = ""
    st.session_state["qna_expanded_project_id"] = ""
    clear_active_qna_state()


def open_qna_conversation(repository: PaperRepository, conversation: dict) -> None:
    clear_active_qna_state()
    st.session_state["qna_selected_paper_id"] = conversation["paper_id"]
    st.session_state["qna_selected_paper_name"] = conversation.get("file_name", "")
    st.session_state["qna_expanded_project_id"] = conversation["paper_id"]
    st.session_state["qna_active_conversation_id"] = conversation["conversation_id"]
    st.session_state["qna_active_conversation_title"] = conversation["title"]


def start_pending_qna_chat(paper: dict) -> None:
    clear_active_qna_state()
    st.session_state["qna_selected_paper_id"] = paper["paper_id"]
    st.session_state["qna_selected_paper_name"] = paper["file_name"]
    st.session_state["qna_expanded_project_id"] = paper["paper_id"]
    st.session_state["qna_active_conversation_title"] = "New chat"
    st.session_state["qna_pending_new_chat"] = True
    st.session_state["qna_action_message"] = "Start a new conversation about this paper."


def normalize_chat_title(title: str) -> str:
    return " ".join(title.strip().split())[:80]


def conversation_text_for_title(messages: list[dict]) -> str:
    return "\n".join(f"{message.get('role', '')}: {message.get('content', '')}" for message in messages if message.get("content"))


def render_generation_summary(summary: dict | None) -> None:
    if not summary:
        return
    success = summary.get("success", [])
    skipped = summary.get("skipped", [])
    failed = summary.get("failed", [])
    st.success(f"{len(success)} reading card(s) generated successfully")
    st.info(f"{len(skipped)} skipped")
    if failed:
        st.error(f"{len(failed)} failed")
    else:
        st.info("0 failed")

    if success:
        with st.expander("Successfully generated"):
            render_name_list(success)
    if skipped:
        with st.expander("Skipped"):
            render_name_list(skipped)
    if failed:
        with st.expander("Failed"):
            render_name_list(failed)


@st.dialog("Edit chat title")
def edit_qna_chat_title_dialog(
    repository: PaperRepository,
    conversation: dict,
    llm: object,
) -> None:
    current_title = conversation.get("title", "New chat")
    custom_title = st.text_input("Current title", value=current_title, max_chars=80)
    save_col, generate_col, cancel_col = st.columns([1.2, 2.1, 1])

    if save_col.button("Save custom title", type="primary"):
        clean_title = normalize_chat_title(custom_title)
        if not clean_title:
            st.error("Enter a title before saving.")
        else:
            try:
                updated = repository.update_conversation_title(conversation["conversation_id"], clean_title)
                if st.session_state.get("qna_active_conversation_id") == conversation["conversation_id"]:
                    st.session_state["qna_active_conversation_title"] = updated["title"]
                st.session_state["qna_action_message"] = "Chat title updated."
                st.rerun()
            except RepositoryError as exc:
                st.error(f"Could not update chat title: {exc}")

    if generate_col.button("Generate title from conversation"):
        messages = repository.list_conversation_messages(conversation["conversation_id"])
        if not messages:
            st.warning("This chat has no messages yet, so a title cannot be generated.")
        else:
            raw_title, error = invoke_text(
                CHAT_TITLE_PROMPT,
                llm,
                {"conversation": conversation_text_for_title(messages)},
            )
            if error:
                st.error(error)
            else:
                clean_title = normalize_chat_title(raw_title.strip().strip('"').strip("'").rstrip("."))
                if not clean_title:
                    st.error("The model did not return a usable title.")
                else:
                    try:
                        updated = repository.update_conversation_title(conversation["conversation_id"], clean_title)
                        if st.session_state.get("qna_active_conversation_id") == conversation["conversation_id"]:
                            st.session_state["qna_active_conversation_title"] = updated["title"]
                        st.session_state["qna_action_message"] = "Chat title updated."
                        st.rerun()
                    except RepositoryError as exc:
                        st.error(f"Could not update chat title: {exc}")

    if cancel_col.button("Cancel"):
        st.rerun()


@st.dialog("Select papers to compare")
def select_compare_papers_dialog(saved_cards: list[dict]) -> None:
    confirmed_ids = set(st.session_state.get("compare_selected_paper_ids", []))
    reset = st.session_state.get("compare_dialog_reset", 0)
    selected_ids = []
    for item in saved_cards:
        paper_id = item["paper_id"]
        checked = st.checkbox(
            item["file_name"],
            value=paper_id in confirmed_ids,
            key=f"compare_dialog_{reset}_{paper_id}",
        )
        if checked:
            selected_ids.append(paper_id)

    if len(selected_ids) > 4:
        st.warning("Select at most 4 papers.")
    if len(selected_ids) < 2:
        st.info("Select at least 2 papers to compare.")

    cols = st.columns(2)
    if cols[0].button("Confirm selection", type="primary", disabled=len(selected_ids) < 2 or len(selected_ids) > 4):
        set_compare_selection(selected_ids)
        st.session_state["compare_dialog_reset"] = reset + 1
        st.rerun()
    if cols[1].button("Cancel"):
        st.session_state["compare_dialog_reset"] = reset + 1
        st.rerun()


@st.dialog("Delete chat?")
def delete_qna_chat_dialog(repository: PaperRepository, conversation: dict) -> None:
    st.write(f"Delete **{conversation.get('title', 'this chat')}** and all of its messages?")
    st.warning("This will not delete the paper or its reading card.")
    cols = st.columns(2)
    if cols[0].button("Confirm delete chat", type="primary"):
        try:
            repository.delete_conversation(conversation["conversation_id"])
            if st.session_state.get("qna_active_conversation_id") == conversation["conversation_id"]:
                clear_active_qna_state()
            st.session_state["qna_action_message"] = "Chat deleted."
        except RepositoryError as exc:
            st.session_state["qna_action_message"] = f"Could not delete chat: {exc}"
        st.rerun()
    if cols[1].button("Cancel"):
        st.rerun()


@st.dialog("Delete this Q&A?")
def delete_qna_turn_dialog(repository: PaperRepository, conversation_id: str, turn_id: int) -> None:
    st.write("Delete this question and its corresponding answer?")
    st.warning("Other Q&A turns in this chat will not be changed.")
    cols = st.columns(2)
    if cols[0].button("Confirm delete Q&A", type="primary"):
        try:
            repository.delete_conversation_turn(conversation_id, turn_id)
            st.session_state["qna_action_message"] = "Q&A deleted."
        except RepositoryError as exc:
            st.session_state["qna_action_message"] = f"Could not delete this Q&A: {exc}"
        st.rerun()
    if cols[1].button("Cancel"):
        st.rerun()


@st.dialog("Select papers for reading cards")
def select_reading_card_papers_dialog(papers: list[dict]) -> None:
    confirmed_ids = set(st.session_state.get("reading_card_selected_paper_ids", []))
    reset = st.session_state.get("reading_card_dialog_reset", 0)
    selected_ids = []
    for paper in papers:
        paper_id = paper["paper_id"]
        checked = st.checkbox(
            paper["file_name"],
            value=paper_id in confirmed_ids,
            key=f"reading_card_dialog_{reset}_{paper_id}",
        )
        if checked:
            selected_ids.append(paper_id)

    cols = st.columns(2)
    if cols[0].button("Confirm selection", type="primary"):
        lookup = {paper["paper_id"]: paper for paper in papers}
        st.session_state["reading_card_selected_paper_ids"] = selected_ids
        st.session_state["reading_card_selected_paper_names"] = [
            lookup[paper_id]["file_name"] for paper_id in selected_ids if paper_id in lookup
        ]
        st.session_state["reading_card_dialog_reset"] = reset + 1
        st.rerun()
    if cols[1].button("Cancel"):
        st.session_state["reading_card_dialog_reset"] = reset + 1
        st.rerun()


@st.dialog("Reading Card")
def view_reading_card_dialog(record: dict) -> None:
    st.subheader(record.get("file_name", "Paper"))
    st.caption(f"Created: {format_timestamp(record.get('created_at'))}")
    st.caption(f"Updated: {format_timestamp(record.get('updated_at'))}")
    st.markdown(reading_card_to_markdown(record.get("card", {})))
    markdown = reading_card_to_markdown(record.get("card", {}))
    st.download_button(
        "Export",
        markdown,
        f"{Path(record.get('file_name', 'reading_card')).stem}_reading_card.md",
        "text/markdown",
    )
    if st.button("Close"):
        st.rerun()


@st.dialog("Regenerate this reading card?")
def regenerate_reading_card_dialog(
    record: dict,
    repository: PaperRepository,
    llm: object,
    model_name: str,
) -> None:
    st.write(f"Regenerate the reading card for **{record.get('file_name', 'this paper')}**?")
    st.write("The saved reading card will be overwritten. Other papers will not be changed.")
    cols = st.columns(2)
    if cols[0].button("Confirm regeneration", type="primary"):
        try:
            chunks = paper_chunks_for_retrieval(repository, record["paper_id"], record.get("file_name", ""))
            card, error, _local_summaries = generate_reading_card_from_chunks(record.get("file_name", ""), chunks, llm)
            if error or not card:
                st.session_state["reading_card_action_message"] = (
                    f"Could not regenerate {record.get('file_name', 'the reading card')}: {error or 'No reading card was returned.'}"
                )
            else:
                save_or_replace_reading_card(
                    repository,
                    record["paper_id"],
                    card,
                    model_name=model_name,
                    overwrite=True,
                )
                st.session_state["reading_card_action_message"] = (
                    f"Regenerated reading card for {record.get('file_name', 'the selected paper')}."
                )
        except RepositoryError as exc:
            st.session_state["reading_card_action_message"] = f"Could not regenerate the reading card: {exc}"
        except Exception as exc:
            st.session_state["reading_card_action_message"] = f"Could not regenerate the reading card: {str(exc).splitlines()[0][:250]}"
        st.rerun()
    if cols[1].button("Cancel"):
        st.rerun()


@st.dialog("Delete reading card?")
def delete_reading_card_dialog(record: dict, repository: PaperRepository) -> None:
    st.write(f"Delete the reading card for **{record.get('file_name', 'this paper')}**?")
    st.warning("This will delete the reading card but will not delete the original paper.")
    cols = st.columns(2)
    if cols[0].button("Confirm deletion", type="primary"):
        try:
            deleted_count = repository.delete_reading_card_for_paper(record["paper_id"])
            if deleted_count:
                st.session_state["reading_card_action_message"] = "Reading card deleted."
            else:
                st.session_state["reading_card_action_message"] = "No saved reading card was found to delete."
        except RepositoryError as exc:
            st.session_state["reading_card_action_message"] = f"Could not delete the reading card: {exc}"
        st.rerun()
    if cols[1].button("Cancel"):
        st.rerun()


def app() -> None:
    st.set_page_config(page_title="AI Paper Reader", page_icon="EE", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    init_state()
    workspace_id = initialize_workspace()

    repository, _, db_ok = get_repository(workspace_id)

    st.title("AI Paper Reader")
    st.caption(APP_SUBTITLE)

    with st.sidebar:
        st.header("API Settings")
        backend_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        backend_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        backend_api_configured = bool(backend_api_key and backend_base_url)
        if backend_api_configured:
            st.text_input(
                "API Key",
                type="password",
                value="••••••••••••",
                disabled=True,
                key="api_key_masked_display",
            )
            st.text_input(
                "Base URL",
                type="password",
                value="••••••••••••",
                disabled=True,
                key="base_url_masked_display",
            )
            sidebar_api_key = backend_api_key
            sidebar_base_url = backend_base_url
        else:
            sidebar_api_key = st.text_input(
                "API Key",
                type="password",
                value="",
                key="api_key_input",
            )
            sidebar_base_url = st.text_input(
                "Base URL",
                value="",
                key="base_url_input",
            )

        clean_api_key = sidebar_api_key.strip()
        clean_base_url = sidebar_base_url.strip()
        selected_model = ""
        model_warning = None
        if clean_api_key and clean_base_url:
            model_config_fingerprint = sha256(f"{clean_base_url}\0{clean_api_key}".encode()).hexdigest()
            if st.session_state.get("automatic_model_config") == model_config_fingerprint:
                selected_model = st.session_state.get("automatic_model", "")
            if not selected_model:
                selected_model, model_warning = select_chat_model(clean_api_key, clean_base_url)
                if selected_model:
                    st.session_state["automatic_model"] = selected_model
                    st.session_state["automatic_model_config"] = model_config_fingerprint
            if model_warning:
                st.warning(model_warning)
        api_settings = {
            "api_key": clean_api_key,
            "base_url": clean_base_url,
            "model_name": selected_model,
        }
        if all(api_settings.values()):
            llm, llm_warning = get_llm(**api_settings)
            if llm_warning:
                st.warning(llm_warning)
        else:
            llm = None
        st.markdown(
            '<small><a href="https://developers.openai.com/api/docs/quickstart" '
            'target="_blank" rel="noopener noreferrer">OpenAI API setup guide ↗</a></small>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.header("Upload Papers")
        uploaded_files = st.file_uploader("Upload research PDFs", type=["pdf"], accept_multiple_files=True)
        if not repository:
            st.caption("Configure and connect to the database before processing PDFs.")
        if st.button("Process PDFs", type="primary", disabled=not uploaded_files or repository is None):
            st.session_state["last_upload_messages"] = []
            with st.spinner("Parsing PDFs, chunking text, and saving metadata..."):
                for uploaded_file in uploaded_files or []:
                    parsed, message = ingest_pdf(uploaded_file.name, uploaded_file.getvalue(), repository)
                    st.session_state["last_upload_messages"].append(message)
                    if parsed.paper_id and parsed.parse_status not in {"failed", "duplicate"}:
                        st.session_state["selected_paper_id"] = parsed.paper_id
            st.rerun()

        for message in st.session_state.get("last_upload_messages", []):
            if "Duplicate" in message or "failed" in message or "no selectable" in message:
                st.warning(message)
            else:
                st.info(message)

        st.divider()
        if db_ok:
            st.markdown('<small style="color:#16803a;">● Database connected</small>', unsafe_allow_html=True)
        else:
            st.markdown('<small style="color:#b42318;">● Database unavailable</small>', unsafe_allow_html=True)
            st.caption("Check that MySQL is running and the project .env settings are valid.")

    papers = repository.list_papers() if repository else []
    paper_lookup = {paper["paper_id"]: paper for paper in papers}
    if papers and st.session_state["selected_paper_id"] not in paper_lookup:
        st.session_state["selected_paper_id"] = papers[0]["paper_id"]
    confirmed_card_ids = [paper_id for paper_id in st.session_state.get("reading_card_selected_paper_ids", []) if paper_id in paper_lookup]
    if confirmed_card_ids != st.session_state.get("reading_card_selected_paper_ids", []):
        st.session_state["reading_card_selected_paper_ids"] = confirmed_card_ids
        st.session_state["reading_card_selected_paper_names"] = [paper_lookup[paper_id]["file_name"] for paper_id in confirmed_card_ids]
    qna_paper_id = st.session_state.get("qna_selected_paper_id", "")
    if st.session_state.get("qna_expanded_project_id", "") not in {"", *paper_lookup.keys()}:
        st.session_state["qna_expanded_project_id"] = ""
    if qna_paper_id and qna_paper_id not in paper_lookup:
        clear_qna_project_selection()
    elif qna_paper_id:
        st.session_state["qna_selected_paper_name"] = paper_lookup[qna_paper_id]["file_name"]
        if repository and st.session_state.get("qna_active_conversation_id"):
            conversation = repository.get_conversation(st.session_state["qna_active_conversation_id"])
            if not conversation or conversation.get("paper_id") != qna_paper_id:
                clear_active_qna_state()

    tabs = st.tabs(["Paper Library", "Reading Cards", "Paper Q&A", "Compare Papers", "Export"])

    with tabs[0]:
        st.subheader("Paper Library")
        if not repository:
            st.error("Database connection is unavailable.")
        elif not papers:
            st.info("No papers saved yet. Upload PDFs from the sidebar.")
        else:
            df = pd.DataFrame(papers)
            st.dataframe(
                df[["file_name", "paper_id", "page_count", "parse_status", "uploaded_at"]],
                use_container_width=True,
                hide_index=True,
            )
            st.write("Select papers to delete")
            checkbox_key_prefix = f"paper_delete_selection_{st.session_state['paper_delete_selection_reset']}"
            selected_delete_ids = []
            for paper in papers:
                checked = st.checkbox(
                    paper["file_name"],
                    key=f"{checkbox_key_prefix}_{paper['paper_id']}",
                    help=f"Select {paper['file_name']} for deletion.",
                )
                if checked:
                    selected_delete_ids.append(paper["paper_id"])
            if st.button("Delete selected papers", disabled=not selected_delete_ids):
                st.session_state["pending_delete_paper_ids"] = list(selected_delete_ids)
            pending_delete_ids = [paper_id for paper_id in st.session_state.get("pending_delete_paper_ids", []) if paper_id in paper_lookup]
            if pending_delete_ids:
                pending_names = selected_paper_names(papers, pending_delete_ids)
                st.warning(
                    "Confirm deletion of "
                    f"{len(pending_names)} paper(s): "
                    + ", ".join(pending_names)
                    + ". Related chunks, reading cards, and Q&A history will also be deleted."
                )
                confirm_cols = st.columns(2)
                if confirm_cols[0].button("Confirm deletion of selected papers"):
                    try:
                        deleted_count = delete_selected_papers(repository, pending_delete_ids, confirmed=True)
                        remaining = [paper for paper in repository.list_papers() if paper["paper_id"] not in pending_delete_ids]
                        apply_successful_deletion_state(st.session_state, pending_delete_ids, remaining)
                        if st.session_state.get("qna_selected_paper_id") in pending_delete_ids:
                            clear_qna_project_selection()
                        st.success(f"Deleted {deleted_count} paper(s).")
                        st.rerun()
                    except RepositoryError as exc:
                        logger.error("Paper deletion failed: %s", exc)
                        st.error("Could not delete the selected papers. No records were changed.")
                if confirm_cols[1].button("Cancel deletion"):
                    st.session_state["pending_delete_paper_ids"] = []
                    st.info("Deletion cancelled. No papers were deleted.")
    with tabs[1]:
        st.subheader("Reading Cards")
        selected_card_ids = st.session_state.get("reading_card_selected_paper_ids", [])
        selected_card_names = st.session_state.get("reading_card_selected_paper_names", [])

        if not repository:
            st.error("Database connection is unavailable.")
        elif not papers:
            st.session_state["reading_card_generation_summary"] = None
            st.session_state["reading_card_action_message"] = ""
            st.session_state["reading_card_selected_paper_ids"] = []
            st.session_state["reading_card_selected_paper_names"] = []
            st.info("No papers found in Paper Library. Please upload a paper first.")
        else:
            if st.button("Select papers"):
                select_reading_card_papers_dialog(papers)

            selected_count = len(selected_card_ids)
            if selected_count == 0:
                st.write("No papers selected.")
            elif selected_count == 1:
                st.write("1 paper selected")
            else:
                st.write(f"{selected_count} papers selected")
            if selected_card_names:
                render_name_list(selected_card_names)
                if st.button("Clear selection"):
                    st.session_state["reading_card_selected_paper_ids"] = []
                    st.session_state["reading_card_selected_paper_names"] = []
                    st.session_state["reading_card_dialog_reset"] += 1
                    st.rerun()

            overwrite = st.checkbox("Overwrite existing reading cards", value=False)
            generate_disabled = selected_count == 0
            if st.button("Generate reading cards", type="primary", disabled=generate_disabled):
                success: list[str] = []
                skipped: list[str] = []
                failed: list[str] = []
                progress = st.progress(0)
                status_text = st.empty()
                total = len(selected_card_ids)
                for index, paper_id in enumerate(selected_card_ids, start=1):
                    paper = paper_lookup.get(paper_id)
                    if not paper:
                        failed.append(f"{paper_id} - paper is no longer in the Paper Library.")
                        progress.progress(index / total)
                        continue
                    file_name = paper["file_name"]
                    status_text.write(f"Generating reading card {index} of {total}: {file_name}")
                    try:
                        if repository.has_reading_card(paper_id) and not overwrite:
                            skipped.append(f"{file_name} - a saved reading card already exists.")
                            progress.progress(index / total)
                            continue
                        if paper.get("parse_status") != "completed":
                            failed.append(f"{file_name} - parsed text is not available.")
                            progress.progress(index / total)
                            continue
                        chunks = paper_chunks_for_retrieval(repository, paper_id, file_name)
                        card, error, _local_summaries = generate_reading_card_from_chunks(file_name, chunks, llm)
                        if error or not card:
                            failed.append(f"{file_name} - {error or 'No reading card was returned.'}")
                        else:
                            save_or_replace_reading_card(
                                repository,
                                paper_id,
                                card,
                                model_name=api_settings["model_name"],
                                overwrite=True,
                            )
                            success.append(file_name)
                    except RepositoryError as exc:
                        failed.append(f"{file_name} - {exc}")
                    except Exception as exc:
                        failed.append(f"{file_name} - {str(exc).splitlines()[0][:250]}")
                    progress.progress(index / total)
                status_text.write("Reading-card generation complete.")
                st.session_state["reading_card_generation_summary"] = {
                    "success": success,
                    "skipped": skipped,
                    "failed": failed,
                }
                st.session_state["reading_card_selected_paper_ids"] = []
                st.session_state["reading_card_selected_paper_names"] = []
                st.session_state["reading_card_dialog_reset"] += 1

        if not repository or papers:
            render_generation_summary(st.session_state.get("reading_card_generation_summary"))

            st.divider()
            st.subheader("Saved Reading Cards")
            action_message = st.session_state.get("reading_card_action_message", "")
            if action_message:
                if action_message.lower().startswith("could not"):
                    st.error(action_message)
                else:
                    st.success(action_message)
                st.session_state["reading_card_action_message"] = ""

            if not repository:
                st.error("Database connection is unavailable.")
            else:
                saved_cards = repository.list_saved_reading_cards()
                search_text = st.text_input("Search saved reading cards", value="")
                if search_text.strip():
                    lowered = search_text.strip().lower()
                    saved_cards = [item for item in saved_cards if lowered in item.get("file_name", "").lower()]

                if not saved_cards:
                    st.info("No saved reading cards found.")
                else:
                    page_size = 8
                    total_pages = max(1, (len(saved_cards) + page_size - 1) // page_size)
                    page = 1
                    if total_pages > 1:
                        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
                    start = (int(page) - 1) * page_size
                    page_cards = saved_cards[start : start + page_size]

                    for record in page_cards:
                        with st.container(border=True):
                            info_col, view_col, regenerate_col, delete_col = st.columns([5, 1, 1.4, 1])
                            info_col.write(f"**{record.get('file_name', 'Paper')}**")
                            info_col.caption(f"Created: {format_timestamp(record.get('created_at'))}")
                            info_col.caption(f"Updated: {format_timestamp(record.get('updated_at'))}")
                            if view_col.button("View", key=f"view_card_{record['paper_id']}"):
                                view_reading_card_dialog(record)
                            if regenerate_col.button("Regenerate", key=f"regenerate_card_{record['paper_id']}"):
                                regenerate_reading_card_dialog(
                                    record,
                                    repository,
                                    llm,
                                    api_settings["model_name"],
                                )
                            if delete_col.button("Delete", key=f"delete_card_{record['paper_id']}"):
                                delete_reading_card_dialog(record, repository)

    with tabs[2]:
        st.subheader("Paper Q&A")
        st.markdown(
            """
            <style>
            [data-testid="stVerticalBlock"]:has(.qna-layout-anchor) [data-testid="stExpander"] {
                width: 100%;
            }
            [data-testid="stVerticalBlock"]:has(.qna-layout-anchor) [data-testid="stExpander"] summary p {
                display: -webkit-box;
                -webkit-line-clamp: 3;
                -webkit-box-orient: vertical;
                overflow: hidden;
                overflow-wrap: normal;
                word-break: normal;
                line-height: 1.25;
            }
            [data-testid="stVerticalBlock"]:has(.qna-layout-anchor) .stButton > button {
                min-height: 2rem;
                padding: 0.3rem 0.55rem;
                width: 100%;
                white-space: nowrap;
                overflow-wrap: normal;
                word-break: normal;
            }
            [data-testid="stVerticalBlock"]:has(.qna-layout-anchor) .stChatMessage {
                max-width: min(980px, 100%);
            }
            [data-testid="stVerticalBlock"]:has(.qna-chat-menu-anchor) [data-testid="stPopover"] button {
                width: 34px !important;
                min-width: 34px !important;
                max-width: 34px !important;
                height: 34px !important;
                min-height: 34px !important;
                padding: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                line-height: 1 !important;
                font-size: 16px !important;
                white-space: nowrap !important;
            }
            </style>
            <div class="qna-layout-anchor"></div>
            """,
            unsafe_allow_html=True,
        )
        if not repository:
            st.error("Database connection is unavailable.")
        elif not papers:
            st.info("No paper projects found. Please upload a paper first.")
        else:
            action_message = st.session_state.get("qna_action_message", "")
            if action_message:
                if action_message.lower().startswith("could not"):
                    st.error(action_message)
                else:
                    st.success(action_message)
                st.session_state["qna_action_message"] = ""

            left_col, right_col = st.columns([1.4, 4.6], gap="large")
            active_paper_id = st.session_state.get("qna_selected_paper_id", "")
            active_conversation_id = st.session_state.get("qna_active_conversation_id", "")
            expanded_project_id = st.session_state.get("qna_expanded_project_id", "")

            with left_col:
                st.write("Projects")
                with st.container(height=760):
                    for paper in papers:
                        paper_id = paper["paper_id"]
                        is_expanded = expanded_project_id == paper_id
                        project_label = compact_label(paper["file_name"], 104)
                        with st.expander(project_label, expanded=is_expanded):
                            if st.button("+ New chat", key=f"new_qna_chat_{paper_id}", type="primary"):
                                start_pending_qna_chat(paper)
                                st.rerun()

                            conversations = repository.list_conversations(paper_id)
                            if not conversations:
                                st.caption("No chats yet.")
                            for conversation in conversations:
                                title = conversation.get("title", "New chat")
                                display_title = f"● {title}" if conversation["conversation_id"] == active_conversation_id else title
                                chat_cols = st.columns([8, 1], gap="small", vertical_alignment="center")
                                if chat_cols[0].button(
                                    compact_label(display_title, 68),
                                    key=f"open_qna_chat_{conversation['conversation_id']}",
                                    use_container_width=True,
                                    help=title,
                                ):
                                    open_qna_conversation(repository, conversation)
                                    st.rerun()
                                chat_cols[1].markdown(
                                    '<span class="qna-chat-menu-anchor"></span>',
                                    unsafe_allow_html=True,
                                )
                                with chat_cols[1].popover("", help="Chat options", use_container_width=True):
                                    if st.button(
                                        "✏️ Edit title",
                                        key=f"edit_qna_chat_title_{conversation['conversation_id']}",
                                        use_container_width=True,
                                    ):
                                        edit_qna_chat_title_dialog(repository, conversation, llm)
                                    if st.button(
                                        "🗑️ Delete chat",
                                        key=f"delete_qna_chat_{conversation['conversation_id']}",
                                        use_container_width=True,
                                    ):
                                        delete_qna_chat_dialog(repository, conversation)
                                st.caption(f"Last updated: {format_timestamp(conversation.get('updated_at'))}")

            with right_col:
                if not active_paper_id:
                    st.info("Select or create a chat from a paper project.")
                else:
                    active_paper = paper_lookup.get(active_paper_id)
                    if not active_paper:
                        clear_qna_project_selection()
                        st.warning("The active paper project is no longer available.")
                    else:
                        active_conversation = repository.get_conversation(active_conversation_id) if active_conversation_id else None
                        if active_conversation_id and (not active_conversation or active_conversation.get("paper_id") != active_paper_id):
                            clear_active_qna_state()
                            st.warning("The active chat is no longer available.")
                        else:
                            st.caption(active_paper["file_name"])
                            chat_title = (
                                st.session_state.get("qna_active_conversation_title")
                                or (active_conversation or {}).get("title")
                                or "New chat"
                            )
                            st.subheader(chat_title)

                            messages = repository.list_conversation_messages(active_conversation_id) if active_conversation_id else []
                            if not messages:
                                st.info("Start a new conversation about this paper.")

                            grouped_turns: dict[int, list[dict]] = {}
                            for message in messages:
                                grouped_turns.setdefault(int(message["turn_id"]), []).append(message)
                            for turn_id, turn_messages in grouped_turns.items():
                                user_message = next((item for item in turn_messages if item["role"] == "user"), None)
                                assistant_message = next(
                                    (item for item in turn_messages if item["role"] == "assistant"),
                                    None,
                                )
                                if user_message:
                                    with st.chat_message("user"):
                                        st.write(user_message["content"])
                                if assistant_message:
                                    with st.chat_message("assistant"):
                                        st.write(assistant_message["content"])
                                        sources = assistant_message.get("sources", [])
                                        if sources:
                                            with st.expander("Sources"):
                                                render_source_snippets(sources)
                                        qa_action_cols = st.columns([5, 1.35])
                                        qa_action_cols[0].caption("")
                                        if qa_action_cols[1].button(
                                            "Delete Q&A",
                                            key=f"delete_qna_turn_{active_conversation_id}_{turn_id}",
                                            help="Delete this question and answer",
                                        ):
                                            delete_qna_turn_dialog(repository, active_conversation_id, turn_id)

                            question = st.chat_input("Ask a question about this paper.")
                            if question:
                                if llm is None:
                                    st.error("Complete API Key and Base URL in API Settings.")
                                else:
                                    try:
                                        chunks = paper_chunks_for_retrieval(
                                            repository,
                                            active_paper_id,
                                            active_paper["file_name"],
                                        )
                                        if not chunks:
                                            st.error("No extracted text was found for the selected paper.")
                                            st.stop()
                                        conversation_id = active_conversation_id
                                        if st.session_state.get("qna_pending_new_chat") or not conversation_id:
                                            conversation = repository.create_conversation(active_paper_id, title="New chat")
                                            conversation_id = conversation["conversation_id"]
                                            st.session_state["qna_active_conversation_id"] = conversation_id
                                            st.session_state["qna_active_conversation_title"] = conversation["title"]
                                            st.session_state["qna_pending_new_chat"] = False
                                        chat_history = [
                                            {"role": item["role"], "content": item["content"]}
                                            for item in repository.list_conversation_messages(conversation_id)
                                        ]
                                        with st.spinner("Retrieving this paper's chunks and answering in the active chat..."):
                                            result, error = ask_in_conversation(
                                                repository,
                                                active_paper_id,
                                                conversation_id,
                                                question,
                                                chunks,
                                                llm,
                                                model_name=api_settings["model_name"],
                                                chat_history=chat_history,
                                            )
                                        if error:
                                            st.error(error)
                                        else:
                                            conversation = repository.get_conversation(conversation_id)
                                            if conversation:
                                                st.session_state["qna_active_conversation_title"] = conversation["title"]
                                                st.session_state["qna_expanded_project_id"] = conversation["paper_id"]
                                            st.rerun()
                                    except RepositoryError as exc:
                                        st.error(f"Could not save this chat: {exc}")

    with tabs[3]:
        st.subheader("Compare Papers")
        if not repository:
            st.error("Database connection is unavailable.")
        else:
            saved_cards = repository.list_saved_reading_cards()
            saved_lookup = {item["paper_id"]: item for item in saved_cards}
            valid_selected_ids = [
                paper_id for paper_id in st.session_state.get("compare_selected_paper_ids", []) if paper_id in saved_lookup
            ]
            if valid_selected_ids != st.session_state.get("compare_selected_paper_ids", []):
                set_compare_selection(valid_selected_ids)

            if len(saved_cards) < 2:
                st.info("At least 2 saved reading cards are required for paper comparison.")
                st.caption("Please generate reading cards from the Reading Cards page first.")
                st.button("Select papers", disabled=True, key="compare_select_disabled")
                st.button("Compare", disabled=True, key="compare_run_disabled")
            else:
                if st.button("Select papers", key="compare_select_papers"):
                    select_compare_papers_dialog(saved_cards)

                selected_cards = selected_compare_cards(saved_cards)
                selected_count = len(selected_cards)
                st.write(f"{selected_count} paper(s) selected" if selected_count else "No papers selected.")
                if selected_cards:
                    for item in selected_cards:
                        chip_cols = st.columns([6, 1])
                        chip_cols[0].markdown(f"**{item['file_name']}**")
                        if chip_cols[1].button("×", key=f"remove_compare_{item['paper_id']}", help="Remove from comparison"):
                            remaining_ids = [
                                paper_id for paper_id in st.session_state["compare_selected_paper_ids"] if paper_id != item["paper_id"]
                            ]
                            set_compare_selection(remaining_ids)
                            st.rerun()

                action_cols = st.columns([1, 1, 5])
                if action_cols[0].button("Clear selection", disabled=not selected_cards, key="compare_clear_selection"):
                    set_compare_selection([])
                    st.rerun()

                can_compare = 2 <= selected_count <= 4
                if action_cols[1].button("Compare", type="primary", disabled=not can_compare, key="compare_run"):
                    detailed = build_detailed_comparison(selected_cards)
                    summary, error = invoke_text(
                        COMPARISON_SUMMARY_PROMPT,
                        llm,
                        {"comparison_context": comparison_prompt_context(selected_cards)},
                    )
                    if error:
                        st.warning(error)
                        summary = ""
                    st.session_state["compare_result"] = {
                        "selected_paper_ids": [item["paper_id"] for item in selected_cards],
                        "paper_names": [item["file_name"] for item in selected_cards],
                        "detailed": detailed,
                        "summary": summary,
                        "summary_error": error,
                        "generated_at": datetime.now(UTC).isoformat(),
                    }
                    st.rerun()

                result = st.session_state.get("compare_result")
                if result:
                    st.divider()
                    st.subheader("Comparison Summary")
                    if result.get("summary"):
                        st.markdown(result["summary"])
                    elif result.get("summary_error"):
                        st.warning(result["summary_error"])
                    else:
                        st.info("Comparison summary is not available.")

                    st.divider()
                    st.subheader("Detailed Comparison")
                    render_detailed_comparison(result.get("detailed", {}), len(result.get("selected_paper_ids", [])))

                    st.divider()
                    st.download_button(
                        "Export comparison",
                        comparison_to_markdown(result),
                        "paper_comparison.md",
                        "text/markdown",
                    )

    with tabs[4]:
        st.subheader("Export")
        if not repository:
            st.error("Database connection is unavailable.")
        else:
            cards = repository.list_saved_reading_cards()
            qa_rows = repository.list_qa_history()
            chats = qna_chat_records(repository, papers)
            comparison_result = st.session_state.get("compare_result")
            step = int(st.session_state.get("export_step", 1))
            step_labels = ["Content", "Records", "Format", "Review"]
            progress = " → ".join(
                f"**{index}. {label}**" if index == step else f"{index}. {label}" for index, label in enumerate(step_labels, start=1)
            )
            st.caption(f"Step {step} of 4")
            st.markdown(progress)

            if step == 1:
                st.subheader("Step 1: Select content")
                content_options = list(EXPORT_FORMATS)
                current = st.session_state.get("export_content_type", "")
                index = content_options.index(current) if current in content_options else None
                selected_content = st.radio("Content type", content_options, index=index)
                next_cols = st.columns([1, 5, 1])
                if next_cols[2].button("Next", disabled=not selected_content, key="export_step1_next"):
                    if selected_content != st.session_state.get("export_content_type"):
                        st.session_state["export_selected_ids"] = []
                        st.session_state["export_format"] = ""
                        st.session_state["export_options"] = {}
                        st.session_state["export_generated_file"] = None
                    st.session_state["export_content_type"] = selected_content
                    st.session_state["export_step"] = 2
                    st.rerun()

            elif step == 2:
                st.subheader("Step 2: Select records")
                content_type = st.session_state.get("export_content_type", "")
                selected_ids = list(st.session_state.get("export_selected_ids", []))
                next_enabled = True

                if content_type == "Reading Cards":
                    if not cards:
                        st.info("No saved reading cards are available.")
                        next_enabled = False
                    for item in cards:
                        checked = st.checkbox(
                            item["file_name"],
                            value=item["paper_id"] in selected_ids,
                            key=f"export_card_{item['paper_id']}",
                        )
                        if checked and item["paper_id"] not in selected_ids:
                            selected_ids.append(item["paper_id"])
                        if not checked and item["paper_id"] in selected_ids:
                            selected_ids.remove(item["paper_id"])
                    next_enabled = bool(selected_ids)
                elif content_type == "Paper Comparison":
                    if comparison_result:
                        st.write("Current paper comparison result is available.")
                        st.write(f"Compared papers: {len(comparison_result.get('paper_names', []))}")
                        selected_ids = ["current_comparison"]
                    else:
                        st.info("No current comparison result is available. Create one on the Compare Papers page first.")
                        selected_ids = []
                        next_enabled = False
                elif content_type == "Q&A Chats":
                    if not chats:
                        st.info("No saved Q&A chats are available.")
                        next_enabled = False
                    for paper in papers:
                        paper_chats = [chat for chat in chats if chat["paper_id"] == paper["paper_id"]]
                        if not paper_chats:
                            continue
                        with st.expander(paper["file_name"]):
                            for chat in paper_chats:
                                checked = st.checkbox(
                                    chat["title"],
                                    value=chat["conversation_id"] in selected_ids,
                                    key=f"export_chat_{chat['conversation_id']}",
                                )
                                if checked and chat["conversation_id"] not in selected_ids:
                                    selected_ids.append(chat["conversation_id"])
                                if not checked and chat["conversation_id"] in selected_ids:
                                    selected_ids.remove(chat["conversation_id"])
                    next_enabled = bool(selected_ids)
                elif content_type == "Full Library Report":
                    st.write(f"Includes {len(papers)} paper(s), {len(cards)} reading card(s), and {len(qa_rows)} Q&A history row(s).")
                    selected_ids = ["full_library_report"]
                elif content_type == "Metadata and Generated Results Export":
                    st.write("Includes paper metadata, saved reading cards, Q&A history, Q&A chats, and the current comparison result if available.")
                    selected_ids = ["metadata_and_generated_results"]

                st.session_state["export_selected_ids"] = selected_ids
                nav_cols = st.columns([1, 5, 1])
                if nav_cols[0].button("Back", key="export_step2_back"):
                    st.session_state["export_step"] = 1
                    st.rerun()
                if nav_cols[2].button("Next", disabled=not next_enabled, key="export_step2_next"):
                    st.session_state["export_step"] = 3
                    st.rerun()

            elif step == 3:
                st.subheader("Step 3: Select format")
                content_type = st.session_state.get("export_content_type", "")
                formats = EXPORT_FORMATS.get(content_type, [])
                current_format = st.session_state.get("export_format", "")
                index = formats.index(current_format) if current_format in formats else None
                export_format = st.radio("Export format", formats, horizontal=True, index=index)
                options = {}
                if content_type in {"Q&A Chats"}:
                    saved_options = st.session_state.get("export_options", {})
                    options["include_sources"] = st.checkbox(
                        "Include source references",
                        value=saved_options.get("include_sources", True),
                    )
                if content_type == "Q&A Chats":
                    options["include_timestamps"] = st.checkbox(
                        "Include timestamps",
                        value=saved_options.get("include_timestamps", True),
                    )
                st.session_state["export_options"] = options
                nav_cols = st.columns([1, 5, 1])
                if nav_cols[0].button("Back", key="export_step3_back"):
                    st.session_state["export_step"] = 2
                    st.rerun()
                if nav_cols[2].button("Next", disabled=not export_format, key="export_step3_next"):
                    st.session_state["export_format"] = export_format
                    st.session_state["export_step"] = 4
                    st.session_state["export_generated_file"] = None
                    st.rerun()

            elif step == 4:
                st.subheader("Step 4: Review export")
                content_type = st.session_state.get("export_content_type", "")
                selected_ids = list(st.session_state.get("export_selected_ids", []))
                export_format = st.session_state.get("export_format", "")
                options = dict(st.session_state.get("export_options", {}))
                records = export_selected_records(content_type, selected_ids, cards, chats)
                if content_type == "Paper Comparison" and comparison_result:
                    record_names = comparison_result.get("paper_names", [])
                    record_count = len(record_names)
                elif content_type in {"Full Library Report", "Metadata and Generated Results Export"}:
                    record_names = ["All available records"]
                    record_count = len(papers)
                else:
                    record_names = [item.get("file_name") or item.get("title", "Record") for item in records]
                    record_count = len(records)
                estimated_name = export_filename(content_type, export_format)

                st.write(f"**Content:** {content_type}")
                st.write(f"**Selected:** {record_count} record(s)")
                for name in record_names:
                    st.write(f"- {name}")
                st.write(f"**Format:** {export_format}")
                st.write(f"**Additional options:** {', '.join(f'{key}: {value}' for key, value in options.items()) or 'None'}")
                st.write(f"**Output:** {estimated_name}")

                nav_cols = st.columns([1, 5, 1])
                if nav_cols[0].button("Back", key="export_step4_back"):
                    st.session_state["export_step"] = 3
                    st.rerun()
                if nav_cols[2].button("Export", type="primary", key="export_step4_export"):
                    try:
                        with st.spinner("Generating export..."):
                            generated = build_export_file(
                                content_type,
                                export_format,
                                records,
                                cards,
                                qa_rows,
                                comparison_result,
                                chats,
                                papers,
                                options,
                            )
                        st.session_state["export_generated_file"] = generated
                    except Exception as exc:
                        st.error(f"Export failed: {str(exc).splitlines()[0][:250]}")

                generated_file = st.session_state.get("export_generated_file")
                if generated_file:
                    st.success("Export generated.")
                    st.write(f"File: {generated_file['file_name']}")
                    st.write(f"Size: {file_size_label(generated_file['data'])}")
                    st.download_button(
                        "Download",
                        generated_file["data"],
                        generated_file["file_name"],
                        generated_file["mime"],
                        key="export_download_generated",
                    )
                    if st.button("Start another export", key="export_start_another"):
                        reset_export_state()
                        st.rerun()


if os.getenv("PAPER_READER_SKIP_STREAMLIT_UI") != "1":
    app()
