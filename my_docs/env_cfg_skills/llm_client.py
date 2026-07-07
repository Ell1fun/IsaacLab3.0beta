from __future__ import annotations

import json
import os
import socket
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from skill_types import LlmConfig


def get_api_key(cfg: LlmConfig) -> str | None:
    """Read API key from local config first, then env var."""
    if cfg.api_key:
        return cfg.api_key
    key = os.environ.get(cfg.api_key_env, "").strip()
    return key or None


def call_llm_chat_completion(cfg: LlmConfig, api_key: str, messages: list[dict[str, str]]) -> str:
    """Call configured LLM API and return assistant text."""
    if cfg.api_style == "responses":
        return _call_responses_api(cfg, api_key=api_key, messages=messages)
    return _call_chat_completions_api(cfg, api_key=api_key, messages=messages)


def _post_json(cfg: LlmConfig, url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON with Bearer auth, retry, timeout, and progress logs."""
    body = json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    last_err: Exception | None = None
    max_attempts = max(1, cfg.max_retries + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            sys.stderr.write(f"[LLM] request {attempt}/{max_attempts}: POST {url} (timeout={cfg.timeout_s}s)\n")
            sys.stderr.flush()
            with urlopen(req, timeout=cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except HTTPError as e:
            last_err = e
            retryable = e.code in (429,) or (500 <= int(e.code) <= 599)
            if retryable and attempt < max_attempts:
                time.sleep(cfg.retry_backoff_s * attempt)
                continue
            raise RuntimeError(f"LLM API call failed (HTTP {e.code}): {e}") from e
        except (socket.timeout, TimeoutError) as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(cfg.retry_backoff_s * attempt)
                continue
            raise RuntimeError(
                f"模型接口响应超时（timeout_s={cfg.timeout_s}）。"
                "可能是网络问题、服务端排队或返回很慢；可以在 orchestrator_config.local.yml 里调大 llm.timeout_s。"
            ) from e
        except URLError as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(cfg.retry_backoff_s * attempt)
                continue
            raise RuntimeError(f"LLM API call failed: {e}") from e
        except Exception as e:
            last_err = e
            raise RuntimeError(f"LLM API call failed: {e}") from e
    raise RuntimeError(f"LLM API call failed: {last_err}")


def _call_chat_completions_api(cfg: LlmConfig, api_key: str, messages: list[dict[str, str]]) -> str:
    """Call an OpenAI-compatible chat completions endpoint."""
    url = cfg.api_base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/v1/chat/completions"
    payload = {"model": cfg.model, "messages": messages, "temperature": 0.2}
    data = _post_json(cfg, url, api_key=api_key, payload=payload)
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError("Unexpected chat completions response format") from e


def _call_responses_api(cfg: LlmConfig, api_key: str, messages: list[dict[str, str]]) -> str:
    """Call a Responses API endpoint, such as Volcano Ark /api/v3/responses."""
    url = cfg.api_base_url.rstrip("/")
    input_messages = [
        {"role": m["role"], "content": [{"type": "input_text", "text": m["content"]}]}
        for m in messages
    ]
    payload = {"model": cfg.model, "input": input_messages}
    data = _post_json(cfg, url, api_key=api_key, payload=payload)
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    try:
        texts: list[str] = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str):
                    texts.append(text)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass
    raise RuntimeError("Unexpected responses API response format")


def extract_structured_object(text: str) -> dict[str, Any]:
    """Parse a JSON/YAML object from model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("Parsing model YAML output requires PyYAML") from e
        parsed = yaml.safe_load(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON/YAML object")
    return parsed
