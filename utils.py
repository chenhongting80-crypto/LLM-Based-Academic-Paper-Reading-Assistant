"""Core utilities for AI Paper Reader."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from prompts import (
    AGENT_ROUTER_PROMPT,
    KEYWORD_PROMPT,
    PARAMETER_EXTRACTION_PROMPT,
    PAPER_QA_PROMPT,
    QA_PROMPT,
    READING_CARD_PROMPT,
    SUMMARY_PROMPT,
    TERM_EXPLANATION_PROMPT,
)


load_dotenv()


MAX_LLM_CHARS = 24000
READING_CARDS_PATH = Path("data") / "reading_cards.json"


class PaperSummary(BaseModel):
    title: str = Field(description="Paper title or Not specified")
    field: str = Field(description="Environmental engineering research field")
    objective: str = Field(description="Study objective")
    medium: str = Field(description="Environmental medium, such as water, air, soil, waste, or mixed")
    target_pollutant_subject: str = Field(description="Target pollutant, material, organism, system, or subject")
    method: str = Field(description="Main experimental, modeling, monitoring, or assessment method")
    setup: str = Field(description="Study setup, reactor, sampling design, model, or scenario")
    key_conditions: str = Field(description="Important conditions such as pH, dose, time, temperature, scale, or assumptions")
    analytical_methods: str = Field(description="Measurement, characterization, or analytical methods")
    findings: str = Field(description="Main findings")
    limitations: str = Field(description="Study limitations")
    relevance: str = Field(description="Why this paper matters")
    source: str = Field(description="Source file name")


class ResearchParameter(BaseModel):
    research_field: str
    medium: str
    pollutant_subject: str
    method: str
    setup: str
    conditions: str
    metrics: str
    results: str
    limitations: str
    source: str


@dataclass
class Chunk:
    text: str
    file_name: str
    page: int
    chunk_id: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "file_name": self.file_name,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }


def get_env_config() -> Dict[str, str]:
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
        "MODEL_NAME": os.getenv("MODEL_NAME", ""),
    }


def get_llm(
    api_key: str,
    base_url: str = "",
    model_name: str = "",
) -> Tuple[Optional[ChatOpenAI], Optional[str]]:
    clean_api_key = api_key.strip()
    clean_base_url = base_url.strip()
    clean_model_name = model_name.strip()

    if not clean_api_key:
        return None, "API key is missing. Enter API settings in the sidebar to enable LLM features."
    if not clean_base_url:
        return None, "API URL is missing. Enter it in the sidebar to enable LLM features."
    if not clean_model_name:
        return None, "Model name is missing. Enter it in the sidebar to enable LLM features."

    kwargs: Dict[str, Any] = {
        "model": clean_model_name,
        "api_key": clean_api_key,
        "temperature": 0,
    }
    if clean_base_url:
        kwargs["base_url"] = clean_base_url
    return ChatOpenAI(**kwargs), None


def chunk_text(text: str, file_name: str, page: int, chunk_size: int = 1200, overlap: int = 180) -> List[Chunk]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    chunks: List[Chunk] = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        snippet = clean[start:end].strip()
        if snippet:
            chunks.append(Chunk(snippet, file_name, page, f"{file_name}-p{page}-c{index}"))
        if end == len(clean):
            break
        start = max(0, end - overlap)
        index += 1
    return chunks


def parse_pdfs(uploaded_files: Iterable[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    chunks: List[Chunk] = []
    paper_texts: Dict[str, str] = {}
    warnings: List[str] = []

    for uploaded in uploaded_files:
        file_name = uploaded.name
        try:
            pdf_bytes = uploaded.getvalue()
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                page_texts: List[str] = []
                for page_index, page in enumerate(doc, start=1):
                    text = page.get_text("text") or ""
                    if text.strip():
                        page_texts.append(f"\n\n[Page {page_index}]\n{text}")
                        chunks.extend(chunk_text(text, file_name, page_index))
                combined = "".join(page_texts).strip()
                if combined:
                    paper_texts[file_name] = combined
                else:
                    warnings.append(f"{file_name}: no selectable text found. It may be scanned or image-only.")
        except Exception as exc:
            warnings.append(f"{file_name}: failed to parse PDF ({exc}).")

    return [chunk.as_dict() for chunk in chunks], paper_texts, warnings


def build_tfidf_retriever(chunks: List[Dict[str, Any]]) -> Tuple[Optional[TfidfVectorizer], Any, Optional[str]]:
    texts = [chunk["text"] for chunk in chunks if chunk.get("text", "").strip()]
    if not texts:
        return None, None, "No searchable text chunks were found."
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        matrix = vectorizer.fit_transform(texts)
        return vectorizer, matrix, None
    except ValueError as exc:
        return None, None, f"Could not build TF-IDF retriever: {exc}"


def retrieve_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    vectorizer: Optional[TfidfVectorizer],
    matrix: Any,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not query.strip() or vectorizer is None or matrix is None or not chunks:
        return []

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix).flatten()
    ranked_indices = similarities.argsort()[::-1][:top_k]
    results: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        if similarities[idx] <= 0:
            continue
        item = dict(chunks[idx])
        item["score"] = float(similarities[idx])
        results.append(item)
    return results


def truncate_for_llm(text: str, limit: int = MAX_LLM_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.65)]
    tail = text[-int(limit * 0.35) :]
    return f"{head}\n\n[... text truncated for length ...]\n\n{tail}"


def invoke_text_chain(prompt_template: Any, llm: Optional[ChatOpenAI], inputs: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    if llm is None:
        return "", "LLM is unavailable because the API key is missing."
    try:
        chain = prompt_template | llm | StrOutputParser()
        return chain.invoke(inputs), None
    except Exception as exc:
        return "", f"LLM call failed: {exc}"


def normalize_chat_history(chat_history: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    messages: List[Tuple[str, str]] = []
    for item in chat_history[-8:]:
        role = item.get("role", "").lower()
        content = item.get("content", "")
        if not content:
            continue
        if role in {"human", "user"}:
            messages.append(("human", content))
        elif role in {"ai", "assistant"}:
            messages.append(("ai", content))
    return messages


def parse_with_fallback(parser: PydanticOutputParser, raw_text: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return parser.parse(raw_text), None
    except Exception as first_exc:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return parser.pydantic_object.model_validate(data), None
            except Exception:
                pass
        return None, f"Could not parse model output. Raw output preserved. Parser error: {first_exc}"


def explain_term(term: str, llm: Optional[ChatOpenAI]) -> Tuple[str, Optional[str]]:
    if not term.strip():
        return "", "Enter a term to explain."
    return invoke_text_chain(TERM_EXPLANATION_PROMPT, llm, {"term": term.strip()})


def summarize_paper(file_name: str, paper_text: str, llm: Optional[ChatOpenAI]) -> Tuple[Optional[PaperSummary], str, Optional[str]]:
    if llm is None:
        return None, "", "LLM is unavailable because the API key is missing."
    parser = PydanticOutputParser(pydantic_object=PaperSummary)
    inputs = {
        "source": file_name,
        "paper_text": truncate_for_llm(paper_text),
        "format_instructions": parser.get_format_instructions(),
    }
    try:
        chain = SUMMARY_PROMPT | llm | parser
        parsed = chain.invoke(inputs)
        return parsed, parsed.model_dump_json(indent=2), None
    except Exception:
        raw, error = invoke_text_chain(SUMMARY_PROMPT, llm, inputs)
        if error:
            return None, "", error
        parsed, parse_error = parse_with_fallback(parser, raw)
        return parsed, raw, parse_error


def generate_reading_card(file_name: str, paper_text: str, llm: Optional[ChatOpenAI]) -> Tuple[str, Optional[str]]:
    return invoke_text_chain(
        READING_CARD_PROMPT,
        llm,
        {
            "source": file_name,
            "paper_text": truncate_for_llm(paper_text),
        },
    )


def extract_reading_card_section(card_text: str, heading: str) -> str:
    pattern = rf"(?ims)^#\s*{re.escape(heading)}\s*$\s*(.*?)(?=^#\s+|\Z)"
    match = re.search(pattern, card_text)
    return match.group(1).strip() if match else "Not clearly stated in the paper."


def build_reading_card_record(file_name: str, paper_text: str, card_text: str) -> Dict[str, Any]:
    paper_id = sha256(f"{file_name}\n{paper_text}".encode("utf-8", errors="ignore")).hexdigest()[:16]
    return {
        "paper_id": paper_id,
        "file_name": file_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_question": extract_reading_card_section(card_text, "Research Question"),
        "method_data": extract_reading_card_section(card_text, "Method / Data"),
        "key_findings": extract_reading_card_section(card_text, "Key Findings"),
        "limitations": extract_reading_card_section(card_text, "Limitations"),
        "relevance_takeaway": extract_reading_card_section(card_text, "Relevance / Takeaway"),
        "keywords": extract_reading_card_section(card_text, "Keywords"),
    }


def save_reading_card(file_name: str, paper_text: str, card_text: str) -> Dict[str, Any]:
    READING_CARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = build_reading_card_record(file_name, paper_text, card_text)

    records: List[Dict[str, Any]] = []
    if READING_CARDS_PATH.exists():
        try:
            loaded = json.loads(READING_CARDS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = [item for item in loaded if isinstance(item, dict)]
        except json.JSONDecodeError:
            records = []

    updated = False
    for index, existing in enumerate(records):
        if existing.get("paper_id") == record["paper_id"]:
            records[index] = record
            updated = True
            break
    if not updated:
        records.append(record)

    READING_CARDS_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def load_saved_reading_cards() -> List[Dict[str, Any]]:
    if not READING_CARDS_PATH.exists():
        return []
    try:
        loaded = json.loads(READING_CARDS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def delete_saved_reading_card(file_name: str, paper_text: str) -> None:
    if not READING_CARDS_PATH.exists():
        return
    paper_id = sha256(f"{file_name}\n{paper_text}".encode("utf-8", errors="ignore")).hexdigest()[:16]
    records = load_saved_reading_cards()
    remaining = [
        item
        for item in records
        if item.get("paper_id") != paper_id and item.get("file_name") != file_name
    ]
    READING_CARDS_PATH.write_text(json.dumps(remaining, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_parameters(file_name: str, paper_text: str, llm: Optional[ChatOpenAI]) -> Tuple[Optional[ResearchParameter], str, Optional[str]]:
    if llm is None:
        return None, "", "LLM is unavailable because the API key is missing."
    parser = PydanticOutputParser(pydantic_object=ResearchParameter)
    prompt = PARAMETER_EXTRACTION_PROMPT.partial(format_instructions=parser.get_format_instructions())
    inputs = {
        "source": file_name,
        "paper_text": truncate_for_llm(paper_text),
    }
    try:
        chain = prompt | llm | parser
        parsed = chain.invoke(inputs)
        return parsed, parsed.model_dump_json(indent=2), None
    except Exception:
        raw, error = invoke_text_chain(prompt, llm, inputs)
        if error:
            return None, "", error
        parsed, parse_error = parse_with_fallback(parser, raw)
        return parsed, raw, parse_error


def extract_keywords(file_name: str, paper_text: str, llm: Optional[ChatOpenAI]) -> Tuple[List[str], Optional[str]]:
    if llm is None:
        return [], "LLM is unavailable because the API key is missing."
    parser = CommaSeparatedListOutputParser()
    try:
        chain = KEYWORD_PROMPT | llm | parser
        return chain.invoke({"paper_text": truncate_for_llm(paper_text, limit=10000)}), None
    except Exception as exc:
        raw, error = invoke_text_chain(
            KEYWORD_PROMPT,
            llm,
            {"paper_text": truncate_for_llm(paper_text, limit=10000)},
        )
        if error:
            return [], error
        fallback = [item.strip() for item in re.split(r",|\n", raw or "") if item.strip()]
        if fallback:
            return fallback, f"Parser fallback used: {exc}"
        return [], f"Could not parse keywords: {exc}"


def ask_paper_question(
    file_name: str,
    paper_text: str,
    question: str,
    llm: Optional[ChatOpenAI],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, Optional[str]]:
    if not question.strip():
        return "", "Enter a question to ask about the paper."
    return invoke_text_chain(
        PAPER_QA_PROMPT,
        llm,
        {
            "source": file_name,
            "paper_text": truncate_for_llm(paper_text),
            "question": question.strip(),
            "chat_history": normalize_chat_history(chat_history or []),
        },
    )


def answer_question(
    question: str,
    chunks: List[Dict[str, Any]],
    vectorizer: Optional[TfidfVectorizer],
    matrix: Any,
    chat_history: List[Dict[str, str]],
    llm: Optional[ChatOpenAI],
    top_k: int = 5,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    if not chunks:
        return "", [], "Upload at least one text-based PDF before asking questions."
    retrieved = retrieve_chunks(question, chunks, vectorizer, matrix, top_k=top_k)
    if not retrieved:
        return "I could not find relevant text in the uploaded PDFs.", [], None

    context_blocks = []
    for item in retrieved:
        context_blocks.append(
            f"Source: {item['file_name']}, page {item['page']}, chunk {item['chunk_id']}\n{item['text']}"
        )
    output, error = invoke_text_chain(
        QA_PROMPT,
        llm,
        {
            "chat_history": normalize_chat_history(chat_history),
            "context": "\n\n---\n\n".join(context_blocks),
            "question": question,
        },
    )
    if error:
        return "", retrieved, error
    return output or "", retrieved, None


def summaries_to_dataframe(summaries: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if not summaries:
        return pd.DataFrame()
    return pd.DataFrame(list(summaries.values()))


def parameters_to_dataframe(parameters: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if not parameters:
        return pd.DataFrame()
    return pd.DataFrame(list(parameters.values()))


def compare_papers(summaries: Dict[str, Dict[str, Any]], parameters: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, Optional[str]]:
    sources = sorted(set(summaries.keys()) | set(parameters.keys()))
    if len(sources) < 2:
        return pd.DataFrame(), "Upload and process at least two papers before comparing."

    rows = []
    for source in sources:
        summary = summaries.get(source, {})
        params = parameters.get(source, {})
        rows.append(
            {
                "source": source,
                "field": summary.get("field") or params.get("research_field") or "Not specified",
                "medium": summary.get("medium") or params.get("medium") or "Not specified",
                "pollutant_or_subject": summary.get("target_pollutant_subject") or params.get("pollutant_subject") or "Not specified",
                "method": summary.get("method") or params.get("method") or "Not specified",
                "setup": summary.get("setup") or params.get("setup") or "Not specified",
                "conditions": summary.get("key_conditions") or params.get("conditions") or "Not specified",
                "findings_or_results": summary.get("findings") or params.get("results") or "Not specified",
                "limitations": summary.get("limitations") or params.get("limitations") or "Not specified",
                "relevance": summary.get("relevance") or "Not specified",
            }
        )
    return pd.DataFrame(rows), None


def route_intent(user_input: str, llm: Optional[ChatOpenAI]) -> Tuple[str, List[str]]:
    logs = ["Agent started.", f"User input: {user_input}"]
    allowed = {"explain_term", "summarize", "extract", "compare", "question_answering", "unknown"}
    text = user_input.lower()

    heuristic_pairs = [
        ("compare", ["compare", "difference", "versus", "vs."]),
        ("summarize", ["summarize", "summary", "overview"]),
        ("extract", ["extract", "parameter", "condition", "metric", "result"]),
        ("explain_term", ["explain", "define", "what is", "meaning of"]),
        ("question_answering", ["?", "which", "why", "how", "what", "where", "when"]),
    ]
    heuristic = "unknown"
    for label, needles in heuristic_pairs:
        if any(needle in text for needle in needles):
            heuristic = label
            break
    logs.append(f"Heuristic route: {heuristic}")

    if llm is None:
        logs.append("LLM router unavailable; using heuristic route.")
        return heuristic, logs

    try:
        chain = AGENT_ROUTER_PROMPT | llm | StrOutputParser()
        label = chain.invoke({"user_input": user_input}).strip().lower()
        label = re.sub(r"[^a-z_]", "", label)
        if label in allowed:
            logs.append(f"LLM router route: {label}")
            return label, logs
        logs.append(f"LLM router returned invalid label '{label}'; using heuristic fallback.")
        return heuristic, logs
    except Exception as exc:
        logs.append(f"LLM router failed: {exc}; using heuristic fallback.")
        return heuristic, logs


def run_agent(user_input: str, state: Dict[str, Any], llm: Optional[ChatOpenAI]) -> Dict[str, Any]:
    intent, logs = route_intent(user_input, llm)
    result: Dict[str, Any] = {"intent": intent, "logs": logs, "answer": "", "table": None, "error": None}
    paper_texts = state.get("paper_texts", {})

    try:
        if intent == "explain_term":
            term = re.sub(r"^(explain|define|what is|meaning of)\s+", "", user_input, flags=re.I).strip(" ?.")
            answer, error = explain_term(term or user_input, llm)
            result.update(answer=answer, error=error)
        elif intent == "summarize":
            if not paper_texts:
                result["error"] = "Upload at least one PDF before summarizing."
            else:
                source = next(iter(paper_texts))
                parsed, raw, error = summarize_paper(source, paper_texts[source], llm)
                result.update(answer=raw, error=error)
                if parsed:
                    result["table"] = pd.DataFrame([parsed.model_dump()])
        elif intent == "extract":
            if not paper_texts:
                result["error"] = "Upload at least one PDF before extracting parameters."
            else:
                source = next(iter(paper_texts))
                parsed, raw, error = extract_parameters(source, paper_texts[source], llm)
                result.update(answer=raw, error=error)
                if parsed:
                    result["table"] = pd.DataFrame([parsed.model_dump()])
        elif intent == "compare":
            table, error = compare_papers(state.get("summaries", {}), state.get("parameters", {}))
            result.update(table=table, error=error)
        elif intent == "question_answering":
            answer, retrieved, error = answer_question(
                user_input,
                state.get("chunks", []),
                state.get("vectorizer"),
                state.get("tfidf_matrix"),
                state.get("chat_history", []),
                llm,
            )
            result.update(answer=answer, error=error, retrieved=retrieved)
        else:
            result["error"] = "I could not confidently route that request. Try asking to explain, summarize, extract, compare, or answer a paper question."
    except Exception as exc:
        result["error"] = f"Agent routing failure: {exc}"
        result["logs"].append(result["error"])
    return result


def dataframe_download(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return df.to_csv(index=False)


def json_download(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)
