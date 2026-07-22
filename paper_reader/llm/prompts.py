"""LangChain prompt templates."""

from langchain_core.prompts import ChatPromptTemplate

LOCAL_CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize environmental engineering paper chunks. Use only the supplied text and keep source details.",
        ),
        (
            "human",
            """Summarize this source chunk in 4-6 concise bullets.

Source: {source}
Page: {page_number}
Chunk: {chunk_index}

Text:
{chunk_text}
""",
        ),
    ]
)


READING_CARD_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Create structured academic reading cards from environmental engineering papers. "
            "Use only provided chunk summaries. Do not invent information. "
            "Use 'Not clearly stated in the paper.' when evidence is missing. "
            "Return exactly one JSON object with no Markdown code fence or surrounding text.",
        ),
        (
            "human",
            """Create a structured reading card for this paper.

Source: {source}

The fields research_question, method_data, key_findings, limitations, and relevance_takeaway
must each be a plain JSON string, never an array or object. keywords must be an array of strings.

Return exactly this shape:
{{
  "research_question": "What question does the study investigate?",
  "method_data": "The study uses the methods and data stated in the supplied chunks.",
  "key_findings": "The main finding stated in the supplied chunks.",
  "limitations": "The limitations stated in the supplied chunks.",
  "relevance_takeaway": "The environmental engineering relevance stated in the supplied chunks.",
  "keywords": ["keyword one", "keyword two"]
}}

Chunk summaries with provenance:
{chunk_summaries}
""",
        ),
    ]
)


READING_CARD_REPAIR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Repair formatting only. Return exactly one valid JSON object with no Markdown code fence or surrounding text. "
            "Do not add facts, infer missing facts, or change the meaning of the original output.",
        ),
        (
            "human",
            """Target schema:
{{
  "research_question": "string",
  "method_data": "string",
  "key_findings": "string",
  "limitations": "string",
  "relevance_takeaway": "string",
  "keywords": ["string"]
}}

Original output:
{original_output}
""",
        ),
    ]
)


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer questions using only retrieved paper chunks. "
            "If the chunks do not provide enough evidence, say: "
            "The current paper does not contain enough evidence to answer this question. "
            "Do not invent citations.",
        ),
        (
            "human",
            """Question:
{question}

Recent conversation:
{chat_history}

Retrieved evidence:
{context}

Answer concisely and cite pages in the answer.
""",
        ),
    ]
)
