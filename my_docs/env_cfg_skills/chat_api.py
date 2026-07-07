from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from config_loader import load_orchestrator_config
from env_utils import default_local_config_path
from llm_client import call_llm_chat_completion, get_api_key


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="env_cfg_skills.chat_api", add_help=True)
    p.add_argument("--config", default=None, help="Path to orchestrator_config.local.yml")
    p.add_argument("--max_history_chars", type=int, default=24000)
    p.add_argument("--max_turns", type=int, default=20)
    return p


def _approx_tokens_from_text(text: str) -> int:
    return max(1, len(text) // 4)


def _history_chars(history: list[dict[str, Any]]) -> int:
    total = 0
    for m in history:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def _trim_history(history: list[dict[str, Any]], *, max_chars: int, max_turns: int) -> None:
    if max_turns > 0:
        max_msgs = max_turns * 2
        if len(history) > max_msgs:
            del history[: len(history) - max_msgs]
    if max_chars <= 0:
        history.clear()
        return
    while history and _history_chars(history) > max_chars:
        del history[0]


def main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    config_path = (args.config or "").strip() or default_local_config_path()
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        sys.stderr.write(f"Config not found: {config_path}\n")
        sys.stderr.write(
            "Usage: python my_docs/env_cfg_skills/chat_api.py --config my_docs/env_cfg_skills/orchestrator_config.local.yml\n"
        )
        return 2

    cfg = load_orchestrator_config(config_path)
    api_key = get_api_key(cfg.llm)
    if not api_key:
        sys.stderr.write(f"Missing API key. Set llm.api_key or env var: {cfg.llm.api_key_env}\n")
        return 2

    sys.stdout.write("LLM API chat demo\n")
    sys.stdout.write(f"api_base_url: {cfg.llm.api_base_url}\n")
    sys.stdout.write(f"api_style: {cfg.llm.api_style}\n")
    sys.stdout.write(f"model: {cfg.llm.model}\n")
    sys.stdout.write(
        "输入内容开始聊天；输入 /quit 退出；输入 /reset 清空对话上下文；输入 /stats 查看上下文长度。\n\n"
    )

    history: list[dict[str, Any]] = []
    while True:
        try:
            user_text = input("你> ").strip()
        except EOFError:
            sys.stdout.write("\n")
            return 0

        if not user_text:
            continue
        if user_text in ("/q", "/quit", "/exit"):
            return 0
        if user_text == "/reset":
            history.clear()
            sys.stdout.write("已清空对话上下文。\n\n")
            continue
        if user_text == "/stats":
            chars = _history_chars(history)
            sys.stdout.write(
                f"history_messages: {len(history)}\n"
                f"history_chars: {chars}\n"
                f"history_tokens_approx: {_approx_tokens_from_text(''.join([m.get('content','') for m in history if isinstance(m.get('content'), str)]))}\n\n"
            )
            continue

        history.append({"role": "user", "content": user_text})
        _trim_history(history, max_chars=int(args.max_history_chars), max_turns=int(args.max_turns))
        try:
            reply = call_llm_chat_completion(cfg.llm, api_key=api_key, messages=history)  # type: ignore[arg-type]
        except Exception as e:
            sys.stderr.write(f"API call failed: {e}\n")
            sys.stderr.write(
                "如果你看到 AuthenticationError/Unauthorized，通常是 api_key 缺失或无效。\n"
            )
            return 1

        reply = reply.strip()
        history.append({"role": "assistant", "content": reply})
        _trim_history(history, max_chars=int(args.max_history_chars), max_turns=int(args.max_turns))
        sys.stdout.write("AI> " + reply + "\n\n")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
