"""Markdown, PDF, CSV, and JSON export helpers."""

from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from typing import Any

import pandas as pd

from paper_reader.services.reading_cards import reading_card_to_markdown


def dataframe_to_csv(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return df.to_csv(index=False)


def dicts_to_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def report_to_markdown(reading_cards: list[dict[str, Any]], qa_history: list[dict[str, Any]]) -> str:
    parts = ["# AI Paper Reader Report"]
    if reading_cards:
        parts.append("## Reading Cards")
        for item in reading_cards:
            parts.append(f"### {item.get('file_name', 'Paper')}")
            parts.append(reading_card_to_markdown(item.get("card", {})))
    else:
        parts.append("## Reading Cards\nNo reading cards generated yet.")

    parts.append("## Q&A History")
    if qa_history:
        for index, item in enumerate(qa_history, start=1):
            snippets = item.get("citation_snippets", [])
            citation_lines = [
                f"- Page {snippet.get('page_number')}: {snippet.get('snippet', '')[:300]}"
                for snippet in snippets
            ]
            parts.append(
                f"### Question {index}\n"
                f"**Paper:** {item.get('file_name', '')}\n\n"
                f"**Question:** {item.get('question', '')}\n\n"
                f"**Answer:** {item.get('answer', '')}\n\n"
                f"**Citations:**\n" + ("\n".join(citation_lines) if citation_lines else "None")
            )
    else:
        parts.append("No questions asked yet.")
    return "\n\n".join(parts)


def comparison_to_markdown(comparison_result: dict[str, Any]) -> str:
    parts = ["# Paper Comparison"]
    generated_at = comparison_result.get("generated_at", "")
    if generated_at:
        parts.append(f"Generated: {generated_at}")

    papers = comparison_result.get("paper_names", [])
    if papers:
        parts.append("## Compared Papers\n" + "\n".join(f"- {paper}" for paper in papers))

    summary = str(comparison_result.get("summary") or "").strip()
    parts.append("## Comparison Summary\n" + (summary or "Not available"))

    detailed = comparison_result.get("detailed", {}) or {}
    detail_parts = ["## Detailed Comparison"]
    for dimension, rows in detailed.items():
        detail_parts.append(f"### {dimension}")
        for row in rows:
            detail_parts.append(f"**{row.get('file_name', 'Paper')}**\n\n{row.get('value', 'Not available')}")
    parts.append("\n\n".join(detail_parts))
    return "\n\n".join(parts)


def _safe_pdf_text(text: str) -> str:
    return text.encode("latin-1", errors="replace").decode("latin-1")


def markdown_to_pdf(markdown: str, title: str = "AI Paper Reader Export") -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, _safe_pdf_text(title))
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _safe_pdf_text(markdown))
    output = pdf.output(dest="S")
    if isinstance(output, bytearray):
        pdf_bytes = bytes(output)
    elif isinstance(output, bytes):
        pdf_bytes = output
    else:
        pdf_bytes = output.encode("latin-1", errors="replace")
    return BytesIO(pdf_bytes).getvalue()


def markdown_to_docx(markdown: str) -> bytes:
    markdown = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", markdown)
    escaped = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    paragraphs = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{line}</w:t></w:r></w:p>"
        for line in escaped.split("\n")
    )
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{paragraphs}<w:sectPr/></w:body>
</w:document>"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def json_to_zip(files: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name, payload in files.items():
            archive.writestr(file_name, dicts_to_json(payload))
    return buffer.getvalue()
