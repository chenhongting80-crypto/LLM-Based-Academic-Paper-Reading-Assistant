"""Prompt templates for AI Paper Reader."""

from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder, PromptTemplate


TERM_EXPLANATION_PROMPT = PromptTemplate.from_template(
    """You are an environmental engineering teaching assistant.

Explain the term below in simple language for a graduate student.

Term: {term}

Return:
1. Plain-language definition
2. Why it matters in environmental engineering
3. A concrete research or field example
4. Related concepts or measurements
"""
)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured information from environmental engineering research papers. "
            "Be accurate, concise, and use 'Not specified' when the provided text does not contain an answer.",
        ),
        (
            "human",
            """Summarize this paper using only the supplied text.

File/source: {source}

{format_instructions}

Paper text:
{paper_text}
""",
        ),
    ]
)


READING_CARD_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You create structured academic reading cards from research papers. "
            "Use only the provided paper text. Do not invent information. "
            "Use concise academic English.",
        ),
        (
            "human",
            """Create a structured academic reading card from the paper text below.

Rules:
- Base the answer only on the provided paper text.
- Do not invent information.
- If a section is not clearly stated in the paper, write:
  "Not clearly stated in the paper."
- Key Findings should describe the main findings, conclusions, or takeaways supported by the paper.
- Use concise academic English.

Return exactly this Markdown structure:

# Research Question
...

# Method / Data
...

# Key Findings
...

# Limitations
...

# Relevance / Takeaway
...

# Keywords
...

Paper text:
{paper_text}
""",
        ),
    ]
)


PARAMETER_EXAMPLES = [
    {
        "paper_excerpt": (
            "Batch adsorption tests evaluated PFAS removal from groundwater using granular activated carbon. "
            "Experiments varied pH 5-9, contact time 15-240 min, and initial PFOS concentration 1-100 ug/L. "
            "Removal reached 92% at pH 7 after 180 min. LC-MS/MS quantified PFAS."
        ),
        "answer": (
            "{\n"
            '  "research_field": "PFAS treatment",\n'
            '  "medium": "Groundwater",\n'
            '  "pollutant_subject": "PFOS and related PFAS",\n'
            '  "method": "Granular activated carbon adsorption",\n'
            '  "setup": "Batch adsorption experiments",\n'
            '  "conditions": "pH 5-9; contact time 15-240 min; initial concentration 1-100 ug/L",\n'
            '  "metrics": "Removal efficiency; LC-MS/MS concentration measurements",\n'
            '  "results": "Up to 92% PFOS removal at pH 7 after 180 min",\n'
            '  "limitations": "Not specified",\n'
            '  "source": "example_pfas.pdf"\n'
            "}"
        ),
    },
    {
        "paper_excerpt": (
            "A life-cycle assessment compared incineration, composting, and anaerobic digestion for food waste. "
            "The functional unit was one tonne of wet waste. Impacts included global warming potential, "
            "eutrophication, and energy recovery. Results depended strongly on electricity displacement assumptions."
        ),
        "answer": (
            "{\n"
            '  "research_field": "Waste management and life-cycle assessment",\n'
            '  "medium": "Municipal food waste",\n'
            '  "pollutant_subject": "Waste treatment impacts",\n'
            '  "method": "Life-cycle assessment",\n'
            '  "setup": "Scenario comparison of incineration, composting, and anaerobic digestion",\n'
            '  "conditions": "Functional unit: one tonne wet waste",\n'
            '  "metrics": "Global warming potential; eutrophication; energy recovery",\n'
            '  "results": "Ranking depended strongly on electricity displacement assumptions",\n'
            '  "limitations": "Sensitivity to electricity displacement assumptions",\n'
            '  "source": "example_lca.pdf"\n'
            "}"
        ),
    },
]


PARAMETER_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["paper_excerpt", "answer"],
    template="Paper excerpt:\n{paper_excerpt}\n\nExtracted JSON:\n{answer}",
)


PARAMETER_EXTRACTION_PROMPT = FewShotPromptTemplate(
    examples=PARAMETER_EXAMPLES,
    example_prompt=PARAMETER_EXAMPLE_PROMPT,
    prefix=(
        "You extract structured research parameters from environmental engineering papers. "
        "The domain may include water, air, soil, waste, PFAS, microplastics, heavy metals, "
        "monitoring, LCA, risk assessment, and related areas. Use exactly the requested schema. "
        "Use 'Not specified' for missing values."
    ),
    suffix=(
        "Paper source: {source}\n\n"
        "{format_instructions}\n\n"
        "Paper excerpt:\n{paper_text}\n\nExtracted JSON:"
    ),
    input_variables=["paper_text", "source"],
    partial_variables={},
)


KEYWORD_PROMPT = PromptTemplate.from_template(
    """Extract 8 to 15 concise environmental engineering keywords from the text.

Return only a comma-separated list.

Text:
{paper_text}
"""
)


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about uploaded environmental engineering papers. "
            "Use only the retrieved context. If the context does not answer the question, say so clearly. "
            "Include source references using file name and page numbers.",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        (
            "human",
            """Retrieved context:
{context}

Question: {question}

Answer with citations:
""",
        ),
    ]
)


PAPER_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about an uploaded academic paper. "
            "Use only the provided paper text. Do not use outside knowledge. "
            "Keep answers clear and concise.",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        (
            "human",
            """Answer the user's question using only the uploaded paper text.

Rules:
- Answer only based on the uploaded paper text.
- Do not use outside knowledge.
- If the answer is not clearly stated in the paper, say:
  "This is not clearly stated in the paper."
- Keep the answer clear and concise.
- When useful, mention which part of the paper the answer comes from, but do not invent page numbers unless they are available.

Paper text:
{paper_text}

Question:
{question}
""",
        ),
    ]
)


AGENT_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user's intent for an environmental engineering paper copilot. "
            "Return exactly one label from: explain_term, summarize, extract, compare, question_answering, unknown.",
        ),
        ("human", "{user_input}"),
    ]
)
