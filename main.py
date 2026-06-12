"""Streamlit UI for AI Paper Reader."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import (
    ask_paper_question,
    build_tfidf_retriever,
    dataframe_download,
    delete_saved_reading_card,
    generate_reading_card,
    get_llm,
    json_download,
    load_saved_reading_cards,
    parse_pdfs,
    save_reading_card,
)


st.set_page_config(
    page_title="AI Paper Reader",
    page_icon="EE",
    layout="wide",
)

APP_SUBTITLE = "Upload a research paper and generate a structured academic reading card."
API_SETTINGS_PATH = Path("data") / "api_settings.json"

APP_CSS = """
<style>
:root {
    --app-bg: #f5f8fa;
    --surface: #ffffff;
    --surface-soft: #f8fbfb;
    --surface-accent: #e8f3f3;
    --border: #dbe5e8;
    --border-strong: #c8d5da;
    --text: #17202a;
    --muted: #667085;
    --accent: #236f73;
    --accent-dark: #19585c;
    --accent-soft: #eef7f7;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(35, 111, 115, 0.09), transparent 30rem),
        linear-gradient(180deg, var(--app-bg) 0%, #ffffff 24rem);
    color: var(--text);
    font-size: 16px;
}

.block-container {
    max-width: 1120px;
    padding: 2rem 2.2rem 3.2rem;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: 0;
    font-weight: 650;
    line-height: 1.22;
}

h1 {
    font-size: 2rem !important;
    margin-bottom: 0.2rem !important;
}

h2 {
    font-size: 1.32rem !important;
    margin: 0.5rem 0 0.6rem !important;
}

h3 {
    font-size: 1.12rem !important;
    margin: 0.45rem 0 !important;
}

p, li, div[data-testid="stMarkdownContainer"] {
    line-height: 1.6;
}

label, [data-testid="stWidgetLabel"] {
    color: #23313d !important;
    font-size: 0.96rem !important;
    font-weight: 620 !important;
}

[data-testid="stCaptionContainer"],
.section-note {
    color: var(--muted);
    font-size: 0.94rem !important;
    line-height: 1.48;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef6f6 0%, #f8fbfb 42%, #ffffff 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] section {
    padding-top: 1.15rem;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 1.05rem !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    font-size: 0.97rem !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] p {
    font-size: 0.93rem !important;
    line-height: 1.46;
}

.stTextInput input,
[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border-strong) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.97rem !important;
    min-height: 2.45rem;
}

.stTextInput input:focus,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px rgba(35, 111, 115, 0.18) !important;
}

[data-testid="stFileUploader"] section {
    background: rgba(255, 255, 255, 0.72);
    border: 1px dashed var(--border-strong);
    border-radius: 10px;
}

.stButton > button,
.stDownloadButton > button {
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    box-shadow: none;
    font-size: 0.94rem;
    font-weight: 620;
    min-height: 2.45rem;
    transition: border-color 120ms ease, background 120ms ease, color 120ms ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent);
    color: var(--accent-dark);
}

[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
}

[data-testid="stBaseButton-primary"]:hover {
    background: var(--accent-dark) !important;
    border-color: var(--accent-dark) !important;
}

.stDownloadButton > button {
    width: 100%;
}

div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.72rem 0.85rem;
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-size: 0.84rem !important;
    font-weight: 620 !important;
}

div[data-testid="stMetricValue"] {
    color: var(--text);
    font-size: 1.32rem !important;
    font-weight: 680 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.74);
    border: 1px solid var(--border);
    border-radius: 10px;
    gap: 0.18rem;
    padding: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--muted);
    font-size: 0.96rem;
    font-weight: 620;
    padding: 0.58rem 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: var(--surface-accent);
    color: var(--accent-dark);
}

[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(17, 24, 39, 0.035);
}

[data-testid="stExpander"] summary {
    color: var(--text);
    font-size: 0.97rem;
    font-weight: 620;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
    font-size: 0.94rem;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

.paper-row {
    background: rgba(255, 255, 255, 0.66);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 0.94rem;
    font-weight: 560;
    margin-bottom: 0.35rem;
    padding: 0.55rem 0.65rem;
}

hr {
    border-color: var(--border) !important;
    margin: 1rem 0 !important;
}

.element-container {
    margin-bottom: 0.28rem;
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


def load_saved_api_settings() -> dict[str, str]:
    if not API_SETTINGS_PATH.exists():
        return {"api_key": "", "base_url": "", "model_name": ""}
    try:
        data = json.loads(API_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"api_key": "", "base_url": "", "model_name": ""}
    if not isinstance(data, dict):
        return {"api_key": "", "base_url": "", "model_name": ""}
    return {
        "api_key": str(data.get("api_key", "")),
        "base_url": str(data.get("base_url", "")),
        "model_name": str(data.get("model_name", "")),
    }


def save_api_settings(settings: dict[str, str]) -> None:
    API_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    API_SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def init_state() -> None:
    saved_api_settings = load_saved_api_settings()
    defaults = {
        "chunks": [],
        "chunks_by_paper": {},
        "paper_texts": {},
        "vectorizer": None,
        "tfidf_matrix": None,
        "chat_history": [],
        "qa_history": [],
        "reading_card": "",
        "reading_cards": {},
        "summaries": {},
        "parameters": {},
        "keywords": {},
        "comparison": pd.DataFrame(),
        "parse_warnings": [],
        "api_settings": saved_api_settings,
        "save_api_settings": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state["chunks_by_paper"] and st.session_state["chunks"]:
        for chunk in st.session_state["chunks"]:
            file_name = chunk.get("file_name")
            if file_name:
                st.session_state["chunks_by_paper"].setdefault(file_name, []).append(chunk)


def rebuild_chunks_and_retriever() -> None:
    chunks_by_paper = st.session_state.get("chunks_by_paper", {})
    chunks = [
        chunk
        for paper_chunks in chunks_by_paper.values()
        for chunk in paper_chunks
    ]
    st.session_state["chunks"] = chunks
    if chunks:
        vectorizer, matrix, retriever_error = build_tfidf_retriever(chunks)
        st.session_state["vectorizer"] = vectorizer
        st.session_state["tfidf_matrix"] = matrix
        st.session_state["parse_warnings"] = [
            warning
            for warning in st.session_state.get("parse_warnings", [])
            if warning and "No searchable text chunks were found." not in warning
        ]
        if retriever_error:
            st.session_state["parse_warnings"].append(retriever_error)
    else:
        st.session_state["vectorizer"] = None
        st.session_state["tfidf_matrix"] = None
        st.session_state["parse_warnings"] = []


def delete_processed_paper(file_name: str) -> None:
    paper_text = st.session_state["paper_texts"].pop(file_name, "")
    st.session_state["chunks_by_paper"].pop(file_name, None)
    st.session_state["summaries"].pop(file_name, None)
    st.session_state["parameters"].pop(file_name, None)
    st.session_state["keywords"].pop(file_name, None)
    st.session_state["reading_cards"].pop(file_name, None)
    st.session_state["qa_history"] = [
        item for item in st.session_state["qa_history"] if item.get("paper") != file_name
    ]
    st.session_state["chat_history"] = [
        item for item in st.session_state["chat_history"] if item.get("paper") != file_name
    ]
    st.session_state["parse_warnings"] = [
        warning for warning in st.session_state["parse_warnings"] if file_name not in warning
    ]

    comparison = st.session_state.get("comparison", pd.DataFrame())
    if not comparison.empty:
        for column in ["source", "Paper", "paper", "file_name"]:
            if column in comparison.columns:
                comparison = comparison[comparison[column] != file_name]
        st.session_state["comparison"] = comparison

    if st.session_state["reading_card"] and file_name not in st.session_state["reading_cards"]:
        st.session_state["reading_card"] = next(iter(st.session_state["reading_cards"].values()), "")

    if paper_text:
        delete_saved_reading_card(file_name, paper_text)

    remaining = list(st.session_state["paper_texts"].keys())
    for key in ["reading_card_select", "ask_paper_select"]:
        if st.session_state.get(key) == file_name:
            if remaining:
                st.session_state[key] = remaining[0]
            else:
                st.session_state.pop(key, None)

    rebuild_chunks_and_retriever()
    st.session_state.pop("pending_delete_paper", None)


def safe_pdf_text(text: str) -> str:
    return text.encode("latin-1", errors="replace").decode("latin-1")


def add_pdf_section(pdf: object, heading: str, body: str) -> None:
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, safe_pdf_text(heading))
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, safe_pdf_text(body or "Not available."))
    pdf.ln(4)


def build_pdf_report(paper_names: list[str], reading_card: str, qa_history: list[dict[str, str]]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, "AI Paper Reader Report")
    pdf.ln(4)

    if paper_names:
        add_pdf_section(pdf, "Uploaded Paper File Name", ", ".join(paper_names))

    add_pdf_section(pdf, "Reading Card", reading_card or "No reading card generated yet.")

    if qa_history:
        qa_blocks = []
        for index, item in enumerate(qa_history, start=1):
            qa_blocks.append(
                f"Question {index}\n"
                f"Paper: {item.get('paper', 'Not specified')}\n"
                f"Question: {item.get('question', '')}\n"
                f"Answer: {item.get('answer', '')}"
            )
        add_pdf_section(pdf, "Q&A History", "\n\n".join(qa_blocks))

    output = pdf.output(dest="S")
    if isinstance(output, bytearray):
        pdf_bytes = bytes(output)
    elif isinstance(output, bytes):
        pdf_bytes = output
    else:
        pdf_bytes = output.encode("latin-1", errors="replace")
    pdf_buffer = BytesIO(pdf_bytes)
    return pdf_buffer.getvalue()


def build_comparison_dataframe() -> pd.DataFrame:
    processed_names = set(st.session_state.get("paper_texts", {}).keys())
    saved_cards = [
        item
        for item in load_saved_reading_cards()
        if item.get("file_name") in processed_names
    ]
    if len(saved_cards) < 2:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Paper": item.get("file_name", "Not specified"),
                "Research Question": item.get("research_question", ""),
                "Method / Data": item.get("method_data", ""),
                "Key Findings": item.get("key_findings", ""),
                "Limitations": item.get("limitations", ""),
                "Relevance / Takeaway": item.get("relevance_takeaway", ""),
            }
            for item in saved_cards
        ]
    )


def build_qa_history_dataframe(qa_history: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Paper": item.get("paper", "Not specified"),
                "Question": item.get("question", ""),
                "Answer": item.get("answer", ""),
                "Timestamp": item.get("timestamp", ""),
            }
            for item in qa_history
        ]
    )


def render_downloads() -> None:
    st.subheader("Exports")
    st.caption("Download the current paper matrix, Q&A history, and report artifacts.")
    comparison_df = build_comparison_dataframe()
    st.session_state["comparison"] = comparison_df
    reading_cards = st.session_state.get("reading_cards", {})
    reading_card = "\n\n".join(
        f"# {file_name}\n\n{card.strip()}" for file_name, card in reading_cards.items() if card.strip()
    ).strip()
    if not reading_card:
        reading_card = st.session_state.get("reading_card", "").strip()
    qa_history = st.session_state.get("qa_history", [])
    chat_export = {
        "chat_history": st.session_state.get("chat_history", []),
        "qa_history": qa_history,
    }
    has_chat_export = bool(chat_export["chat_history"] or chat_export["qa_history"])
    qa_df = build_qa_history_dataframe(qa_history)
    has_qa_export = bool(qa_history)

    cols = st.columns(3)
    with cols[0]:
        st.download_button(
            "Paper Summary CSV",
            dataframe_download(comparison_df),
            "paper_summary.csv",
            "text/csv",
            disabled=comparison_df.empty,
        )
        if comparison_df.empty:
            st.caption("Process one or more papers first.")
    with cols[1]:
        st.download_button(
            "Chat JSON",
            json_download(chat_export),
            "chat_history.json",
            "application/json",
            disabled=not has_chat_export,
        )
        if not has_chat_export:
            st.caption("Chat JSON needs Ask Paper Q&A or chat history.")
    with cols[2]:
        st.download_button(
            "Ask Paper Q&A CSV",
            dataframe_download(qa_df),
            "ask_paper_qa_history.csv",
            "text/csv",
            disabled=not has_qa_export,
        )
        if not has_qa_export:
            st.caption("Ask a question first.")

    st.divider()
    if not reading_card and not qa_history:
        st.info("No notes available yet. Generate a reading card or ask questions first.")
        return

    paper_names = list(st.session_state.get("paper_texts", {}).keys())
    report_parts = ["# AI Paper Reader Report"]
    if paper_names:
        report_parts.append("\n## Uploaded Paper File Name\n" + ", ".join(paper_names))

    report_parts.append("\n## Reading Card")
    report_parts.append(reading_card or "No reading card generated yet.")

    report_parts.append("\n## Q&A History")
    if qa_history:
        for index, item in enumerate(qa_history, start=1):
            report_parts.append(
                f"### Question {index}\n"
                f"**Paper:** {item.get('paper', 'Not specified')}\n\n"
                f"**Question:** {item.get('question', '')}\n\n"
                f"**Answer:** {item.get('answer', '')}"
            )
    else:
        report_parts.append("No questions asked yet.")

    markdown_report = "\n\n".join(report_parts)
    st.download_button(
        "Download Markdown Report",
        markdown_report,
        "ai_paper_reader_report.md",
        "text/markdown",
    )

    try:
        pdf_report = build_pdf_report(paper_names, reading_card, qa_history)
        st.download_button(
            "Download PDF Report",
            pdf_report,
            "ai_paper_reader_report.pdf",
            "application/pdf",
        )
    except Exception as exc:
        st.warning(f"PDF export is unavailable: {exc}")


init_state()

st.title("AI Paper Reader")
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("API Settings")
    st.caption("Configure the model used for reading cards and paper questions.")
    saved_api_settings = st.session_state.get("api_settings", {})
    sidebar_api_key = st.text_input(
        "API Key",
        type="password",
        value=saved_api_settings.get("api_key", ""),
        placeholder="Paste your API key",
    )
    sidebar_base_url = st.text_input(
        "API URL",
        value=saved_api_settings.get("base_url", ""),
        placeholder="Enter an OpenAI-compatible API URL",
    )
    sidebar_model_name = st.text_input(
        "Model Name",
        value=saved_api_settings.get("model_name", ""),
        placeholder="Enter the model name",
    )

    resolved_api_key = sidebar_api_key.strip()
    resolved_base_url = sidebar_base_url.strip()
    resolved_model_name = sidebar_model_name.strip()
    st.session_state["api_settings"] = {
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "model_name": resolved_model_name,
    }

    save_config = st.checkbox(
        "Save configuration for next time",
        key="save_api_settings",
        help="Stores these settings locally on this computer. Do not commit the saved file.",
    )
    if st.button("Save API Settings", disabled=not save_config):
        save_api_settings(st.session_state["api_settings"])
        st.success("API settings saved locally.")

    llm, llm_warning = get_llm(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model_name=resolved_model_name,
    )
    if llm_warning:
        st.warning(llm_warning)

    st.markdown("[API Setup Guide](https://platform.openai.com/docs)")

    st.divider()
    st.header("PDF Workspace")
    st.caption("Upload research PDFs, process their text, and manage the active workspace.")

    uploaded_files = st.file_uploader(
        "Upload research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Process PDFs", type="primary", disabled=not uploaded_files):
        with st.spinner("Processing PDFs..."):
            chunks, paper_texts, warnings = parse_pdfs(uploaded_files)
            processed_names = set(paper_texts.keys())
            chunks_by_paper = dict(st.session_state["chunks_by_paper"])
            for file_name in processed_names:
                chunks_by_paper[file_name] = [
                    chunk for chunk in chunks if chunk.get("file_name") == file_name
                ]
            combined_paper_texts = dict(st.session_state["paper_texts"])
            combined_paper_texts.update(paper_texts)
            st.session_state["chunks_by_paper"] = chunks_by_paper
            st.session_state["paper_texts"] = combined_paper_texts
            st.session_state["parse_warnings"] = warnings
            st.session_state["chat_history"] = []
            st.session_state["qa_history"] = []
            st.session_state["reading_card"] = ""
            st.session_state["summaries"] = {}
            st.session_state["parameters"] = {}
            st.session_state["keywords"] = {}
            st.session_state["comparison"] = pd.DataFrame()
            rebuild_chunks_and_retriever()
        st.success(f"Processed {len(paper_texts)} PDF(s) and {len(chunks)} text section(s).")

    if st.session_state["paper_texts"]:
        metric_cols = st.columns(2)
        metric_cols[0].metric("Papers", len(st.session_state["paper_texts"]))
        metric_cols[1].metric("Chunks", len(st.session_state["chunks"]))
        st.divider()
        st.subheader("Processed Papers")
        for file_name in list(st.session_state["paper_texts"].keys()):
            cols = st.columns([3, 1])
            cols[0].markdown(f"<div class='paper-row'>{file_name}</div>", unsafe_allow_html=True)
            if cols[1].button("Delete", key=f"delete_{file_name}"):
                st.session_state["pending_delete_paper"] = file_name

        pending_delete = st.session_state.get("pending_delete_paper")
        if pending_delete:
            st.warning(f"Are you sure you want to delete this processed paper? {pending_delete}")
            confirm_cols = st.columns(2)
            if confirm_cols[0].button("Confirm Delete", key="confirm_delete_paper"):
                delete_processed_paper(pending_delete)
                st.success(f"Deleted {pending_delete}.")
                st.rerun()
            if confirm_cols[1].button("Cancel", key="cancel_delete_paper"):
                st.session_state.pop("pending_delete_paper", None)
                st.rerun()
    for warning in st.session_state["parse_warnings"]:
        if warning:
            st.warning(warning)

tabs = st.tabs(
    [
        "Reading Card",
        "Ask Paper",
        "Compare Papers",
        "Export",
    ]
)

with tabs[0]:
    st.subheader("Academic Reading Card")
    st.markdown(
        "<div class='section-note'>Sections: Research Question, Method / Data, Key Findings, Limitations, Relevance / Takeaway, Keywords.</div>",
        unsafe_allow_html=True,
    )
    if not st.session_state["paper_texts"]:
        st.warning("Upload and process a paper first.")
    else:
        selected = st.selectbox("Paper", list(st.session_state["paper_texts"].keys()), key="reading_card_select")
        if st.button("Generate Reading Card"):
            paper_names = [
                file_name
                for file_name in st.session_state["paper_texts"]
                if file_name == selected or file_name not in st.session_state["reading_cards"]
            ]
            saved_count = 0
            with st.spinner("Generating reading cards..."):
                for file_name in paper_names:
                    card, error = generate_reading_card(file_name, st.session_state["paper_texts"][file_name], llm)
                    if error:
                        st.error(f"{file_name}: {error}")
                        continue
                    if card:
                        st.session_state["reading_card"] = card
                        st.session_state["reading_cards"][file_name] = card
                        save_reading_card(file_name, st.session_state["paper_texts"][file_name], card)
                        saved_count += 1
            if saved_count:
                st.success(f"Saved {saved_count} reading card(s) to data/reading_cards.json.")

        if st.session_state["reading_cards"]:
            st.divider()
            for file_name, card in st.session_state["reading_cards"].items():
                with st.expander(file_name, expanded=file_name == selected):
                    st.markdown(card)
        else:
            st.info("Generate a reading card for a processed paper to display it here.")

with tabs[1]:
    st.subheader("Ask Paper")
    st.caption("Ask concise questions grounded in the selected processed paper.")
    if not st.session_state["paper_texts"]:
        st.warning("Upload and process a paper first.")
    else:
        selected = st.selectbox("Paper", list(st.session_state["paper_texts"].keys()), key="ask_paper_select")
        question = st.text_input("Question", placeholder="What is the main research question?")
        if st.button("Ask"):
            with st.spinner("Answering from the uploaded paper text..."):
                answer, error = ask_paper_question(
                    selected,
                    st.session_state["paper_texts"][selected],
                    question,
                    llm,
                    st.session_state["chat_history"],
                )
            if error:
                st.error(error)
            else:
                st.session_state["chat_history"].extend(
                    [
                        {"role": "human", "content": question, "paper": selected},
                        {"role": "ai", "content": answer, "paper": selected},
                    ]
                )
                st.session_state["qa_history"].append(
                    {
                        "paper": selected,
                        "question": question,
                        "answer": answer,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                st.markdown(answer)

    if st.session_state["qa_history"]:
        st.divider()
        st.write("Previous Q&A")
        for item in reversed(st.session_state["qa_history"]):
            question_preview = item.get("question", "")[:90] or "Question"
            with st.expander(f"{item.get('paper', 'Not specified')} - {question_preview}"):
                st.markdown(f"**Paper:** {item.get('paper', 'Not specified')}")
                st.markdown(f"**Question:** {item.get('question', '')}")
                st.markdown(f"**Answer:** {item.get('answer', '')}")
                if item.get("timestamp"):
                    st.caption(f"Asked at {item['timestamp']}")

with tabs[2]:
    st.subheader("Compare Papers")
    st.caption("Review saved reading cards side by side across processed papers.")
    comparison_df = build_comparison_dataframe()
    st.session_state["comparison"] = comparison_df
    if comparison_df.empty:
        st.info("Generate at least two reading cards before comparing papers.")
    else:
        st.dataframe(comparison_df, use_container_width=True)

with tabs[3]:
    render_downloads()
