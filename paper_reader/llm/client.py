"""LLM setup and safe invocation."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import OpenAI

PREFERRED_CHAT_MODELS = (
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o",
    "gpt-5-mini",
    "gpt-5",
)
NON_CHAT_MODEL_TERMS = (
    "audio",
    "dall-e",
    "embedding",
    "image",
    "moderation",
    "realtime",
    "rerank",
    "speech",
    "transcribe",
    "tts",
    "whisper",
)
CHAT_MODEL_TERMS = ("chat", "gpt", "instruct", "deepseek", "qwen", "llama", "mistral", "command", "gemma")


def _safe_llm_error(exc: Exception, sensitive_values: tuple[str, ...] = ()) -> str:
    message = str(exc).splitlines()[0][:300]
    for value in sensitive_values:
        if value:
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-[REDACTED]", message)
    message = re.sub(r"(?i)(api[_ -]?key[^:]*:\s*)[^,\s}]+", r"\1[REDACTED]", message)
    return message


def select_chat_model(api_key: str, base_url: str, timeout: float = 20) -> tuple[str | None, str | None]:
    clean_api_key = api_key.strip()
    clean_base_url = base_url.strip()
    if not clean_api_key or not clean_base_url:
        return None, "Complete API Key and Base URL in API Settings."

    try:
        response = OpenAI(api_key=clean_api_key, base_url=clean_base_url, timeout=timeout).models.list()
        model_ids = sorted({str(model.id).strip() for model in response.data if str(model.id).strip()})
    except Exception as exc:
        detail = _safe_llm_error(exc, (clean_api_key,))
        return None, f"Available models could not be loaded: {detail}"

    candidates = [
        model_id
        for model_id in model_ids
        if not any(term in model_id.lower() for term in NON_CHAT_MODEL_TERMS)
    ]
    for preferred in PREFERRED_CHAT_MODELS:
        if preferred in candidates:
            return preferred, None
    for term in CHAT_MODEL_TERMS:
        match = next((model_id for model_id in candidates if term in model_id.lower()), None)
        if match:
            return match, None
    if candidates:
        return candidates[0], None
    return None, "No compatible text chat model was returned by the configured API."


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
        return None, "API configuration or automatic model selection is unavailable."

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
        return None, f"LLM is unavailable: {_safe_llm_error(exc, (clean_api_key,))}"


def invoke_text(prompt_template: Any, llm: ChatOpenAI | None, inputs: dict[str, Any]) -> tuple[str, str | None]:
    if llm is None:
        return "", "Complete API Key and Base URL in API Settings."
    try:
        chain = prompt_template | llm | StrOutputParser()
        return chain.invoke(inputs), None
    except Exception as exc:
        return "", f"LLM request failed or timed out: {_safe_llm_error(exc)}"
