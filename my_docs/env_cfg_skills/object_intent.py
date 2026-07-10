from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from env_utils import pascal_from_slug
from presets import load_skill_presets


@dataclass(frozen=True)
class MentionedObject:
    """Object mention detected in the user request."""

    label: str
    preset: str | None
    alias: str
    is_primary_candidate: bool


UNSUPPORTED_OBJECT_ALIASES = {
    "水管": ("水管", "管子", "pipe", "water pipe"),
}

PRIMARY_VERBS = ("抓取", "抓", "拿取", "拿", "操作", "搬运", "pick", "grasp", "grab", "manipulate")


def object_catalog() -> dict[str, dict[str, Any]]:
    """Return object preset catalog keyed by preset name."""
    return load_skill_presets().get("objects", {})


def object_recommendation_lines(catalog: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Build user-facing recommendation lines from object presets."""
    catalog = catalog if catalog is not None else object_catalog()
    lines: list[str] = []
    for preset, data in catalog.items():
        if not isinstance(data, dict):
            continue
        display = str(data.get("display_name", pascal_from_slug(str(preset))))
        description = str(data.get("description", "")).strip()
        status = str(data.get("status", "ready"))
        suffix = f" - {description}" if description else ""
        if status != "ready":
            reason = str(data.get("blocked_reason", "not ready"))
            suffix += f"（当前状态：{status}，原因：{reason}）"
        lines.append(f"- {preset} / {display}{suffix}")
    return lines


def _aliases_for_catalog(catalog: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    for preset, data in catalog.items():
        if not isinstance(data, dict):
            continue
        raw_aliases = data.get("aliases", [])
        if isinstance(raw_aliases, list):
            aliases.extend((str(alias), str(preset)) for alias in raw_aliases)
        aliases.append((str(preset), str(preset)))
        if data.get("object_slug"):
            aliases.append((str(data["object_slug"]), str(preset)))
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    return aliases


def _primary_context(text: str, alias: str) -> bool:
    escaped = re.escape(alias)
    for verb in PRIMARY_VERBS:
        if re.search(f"{re.escape(verb)}[^，,。；;]*{escaped}", text, flags=re.IGNORECASE):
            return True
    return False


def analyze_object_intent(user_prompt: str) -> dict[str, Any]:
    """Analyze object mentions before spec generation and return blocking questions if needed."""
    catalog = object_catalog()
    text = user_prompt.lower()
    mentions: dict[str, MentionedObject] = {}

    for alias, preset in _aliases_for_catalog(catalog):
        if alias.lower() not in text:
            continue
        key = f"known:{preset}"
        existing = mentions.get(key)
        primary = _primary_context(text, alias)
        if existing is None or (primary and not existing.is_primary_candidate):
            mentions[key] = MentionedObject(
                label=str(catalog.get(preset, {}).get("display_name", preset)),
                preset=preset,
                alias=alias,
                is_primary_candidate=primary,
            )

    for label, aliases in UNSUPPORTED_OBJECT_ALIASES.items():
        matched_alias = next((alias for alias in aliases if alias.lower() in text), "")
        if not matched_alias:
            continue
        mentions[f"unknown:{label}"] = MentionedObject(
            label=label,
            preset=None,
            alias=matched_alias,
            is_primary_candidate=_primary_context(text, matched_alias),
        )

    known_mentions = [m for m in mentions.values() if m.preset]
    unknown_mentions = [m for m in mentions.values() if not m.preset]
    primary_candidates = [m for m in mentions.values() if m.is_primary_candidate]
    blocked_mentions = [
        m
        for m in known_mentions
        if isinstance(catalog.get(str(m.preset)), dict) and str(catalog[str(m.preset)].get("status", "ready")) != "ready"
    ]
    errors: list[str] = []
    questions: list[str] = []

    if unknown_mentions:
        unknown_labels = "、".join(sorted({m.label for m in unknown_mentions}))
        errors.append(f"unknown object mention(s): {unknown_labels}")
        recommendations = "\n".join(object_recommendation_lines(catalog))
        questions.append(
            "我检测到这些物体当前不在 object_presets.yml 中："
            f"{unknown_labels}。\n"
            "当前不能为未知物体编造 preset 或 USD 路径。\n"
            "你希望怎么处理？\n"
            "1. 改用下面已有物体之一。\n"
            "2. 提供该物体的 USD 路径、scale、是否 single_rigid_body，我先帮你新增 object preset。\n"
            "3. 本次不添加这个未知物体。\n"
            f"当前已有 object presets：\n{recommendations}"
        )

    if len(primary_candidates) > 1:
        candidate_labels = "、".join(m.label for m in primary_candidates)
        errors.append(f"ambiguous primary object candidates: {candidate_labels}")
        questions.append(
            "我检测到多个可能的主操作物体："
            f"{candidate_labels}。\n"
            "当前阶段只支持一个主 pick-place object，其余物体只能作为 scene_objects 附加摆放。\n"
            "请明确哪个物体作为本次主操作物体；其余物体是否作为 prop/distractor/obstacle 保留。"
        )

    if len(known_mentions) > 1 and not primary_candidates:
        candidate_labels = "、".join(m.label for m in known_mentions)
        errors.append(f"primary object is unclear among mentioned objects: {candidate_labels}")
        questions.append(
            "我检测到多个已知物体，但没有明确哪个是主 pick-place object："
            f"{candidate_labels}。\n"
            "请明确一个主操作物体；其余物体可以作为 scene_objects 附加摆放。"
        )

    if blocked_mentions:
        blocked_labels = "、".join(m.label for m in blocked_mentions)
        errors.append(f"object preset is not ready: {blocked_labels}")
        details = []
        for mention in blocked_mentions:
            data = catalog.get(str(mention.preset), {})
            reason = str(data.get("blocked_reason", "not ready")) if isinstance(data, dict) else "not ready"
            details.append(f"- {mention.preset}: {reason}")
        questions.append(
            "我检测到这些物体 preset 当前还不是 ready，不能直接生成可运行 env_cfg：\n"
            + "\n".join(details)
            + "\n请选择改用 ready 物体，或先补齐该 preset 的资产信息。"
        )

    if not known_mentions and not unknown_mentions:
        errors.append("no object mention detected")
        questions.append(
            "我还没有识别到要操作或摆放的物体。请从当前 object presets 中选择一个主物体：\n"
            + "\n".join(object_recommendation_lines(catalog))
        )

    return {
        "ok": not errors,
        "errors": errors,
        "questions": questions,
        "known_mentions": [m.__dict__ for m in known_mentions],
        "unknown_mentions": [m.__dict__ for m in unknown_mentions],
        "primary_candidates": [m.__dict__ for m in primary_candidates],
    }
