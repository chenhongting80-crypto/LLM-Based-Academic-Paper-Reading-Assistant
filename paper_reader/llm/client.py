"""LLM setup and safe invocation."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def _safe_llm_error(exc: Exception) -> str:
    message = str(exc).splitlines()[0][:300]
    message = re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-[REDACTED]", message)
    message = re.sub(r"(?i)(api[_ -]?key[^:]*:\s*)[^,\s}]+", r"\1[REDACTED]", message)
    return message


def get_llm(
    api_key: str,
    base_url: str,
    model_name: str,
    timeout: float = 60,
) -> tuple[ChatOpenAI | None, str | None]:
    clean_api_key = api_key.strip()
    clean_base_url = base_url.strip()
    clean_model_name = model_name.strip()

    if not all((clean_api_key, clean_base_url, clean_model_name)):
        return None, "Complete API Key, Base URL, and Model Name in API Settings."

    try:
        return (
            ChatOpenAI(
                model=clean_model_name,
                api_key=clean_api_key,
                base_url=clean_base_url,
                temperature=0,
                timeout=timeout,
            ),
            None,
        )
    except Exception as exc:
        return None, f"LLM is unavailable: {_safe_llm_error(exc)}"


def invoke_text(prompt_template: Any, llm: ChatOpenAI | None, inputs: dict[str, Any]) -> tuple[str, str | None]:
    if llm is None:
        return "", "Complete API Key, Base URL, and Model Name in API Settings."
    try:
        chain = prompt_template | llm | StrOutputParser()
        return chain.invoke(inputs), None
    except Exception as exc:
        return "", f"LLM request failed or timed out: {_safe_llm_error(exc)}"
