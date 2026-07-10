from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from clarification_resolver import build_intent_from_llm_payload, build_or_update_intent, intent_to_spec, validate_intent
from env_utils import read_text
from llm_client import call_llm_chat_completion, extract_structured_object
from object_intent import object_catalog
from presets import enrich_pick_place_vr_spec, load_skill_presets
from schema_validation import validate_spec
from skill_types import OrchestratorConfig, SkillDef


def infer_pick_place_vr_spec_from_text(user_prompt: str) -> dict[str, Any]:
    """Infer a minimal PickPlaceVRSpec from common Chinese/English keywords."""
    text = user_prompt.lower()
    wants_motion_controller = any(
        k in text
        for k in (
            "motion controller",
            "motion-controller",
            "controller",
            "手柄",
            "控制器",
            "pico手柄",
            "pico controller",
        )
    )
    def detect_robot_preset_from_presets(prompt: str) -> str:
        presets = load_skill_presets()
        robots = presets.get("robots", {})
        if not isinstance(robots, dict):
            return ""
        lowered = prompt.lower()
        aliases: list[tuple[str, str]] = []
        for preset, data in robots.items():
            if not isinstance(preset, str) or not isinstance(data, dict):
                continue
            raw_aliases = data.get("aliases", [])
            if isinstance(raw_aliases, list):
                aliases.extend((str(alias), preset) for alias in raw_aliases if alias)
            aliases.append((preset, preset))
            if data.get("robot_slug"):
                aliases.append((str(data["robot_slug"]), preset))
            if data.get("display_name"):
                aliases.append((str(data["display_name"]), preset))
        aliases = [(a.lower(), p) for a, p in aliases if a]
        aliases.sort(key=lambda item: len(item[0]), reverse=True)
        for alias, preset in aliases:
            if alias and alias in lowered:
                return preset
        return ""

    robot_preset = detect_robot_preset_from_presets(text)

    object_aliases: list[tuple[str, str, tuple[str, ...]]] = []
    for preset, data in object_catalog().items():
        if not isinstance(data, dict) or str(data.get("status", "ready")) != "ready":
            continue
        aliases = [str(alias) for alias in data.get("aliases", []) if alias]
        aliases.extend([str(preset), str(data.get("object_slug", preset))])
        object_aliases.append((str(data.get("object_slug", preset)), str(preset), tuple(aliases)))
    mentioned_objects = [
        (name, preset, aliases) for name, preset, aliases in object_aliases if any(alias in text for alias in aliases)
    ]
    object_preset = mentioned_objects[0][1] if mentioned_objects else ""

    def semantic_for_object(aliases: tuple[str, ...]) -> str:
        for alias in aliases:
            if re.search(f"桌.{{0,16}}{re.escape(alias)}", text):
                return "table_back"
        for alias in aliases:
            if re.search(f"{re.escape(alias)}.{{0,12}}左手|左手.{{0,12}}{re.escape(alias)}", text):
                return "near_left_hand"
            if re.search(f"{re.escape(alias)}.{{0,12}}右手|右手.{{0,12}}{re.escape(alias)}", text):
                return "near_right_hand"
        if any(k in text for k in ("桌上", "桌面", "table")):
            return "table_back"
        return "table_center"

    scene_objects: list[dict[str, Any]] = []
    for name, preset, aliases in mentioned_objects[1:]:
        scene_objects.append(
            {
                "name": name,
                "preset": preset,
                "role": "distractor" if preset == "bottled_water_c01" else "prop",
                "placement": {"semantic": semantic_for_object(aliases)},
            }
        )

    spec = {
        "spec_version": "0.1",
        "task": {"family": "manager_based/manipulation/pick_place"},
        "robot": {"preset": robot_preset},
        "scene": {},
        "object": {
            "preset": object_preset,
            **({"placement": {"semantic": semantic_for_object(mentioned_objects[0][2])}} if mentioned_objects else {}),
        },
        "target": {},
        "control": {
            "mode": "motion_controller" if wants_motion_controller else "pink_ik",
            "teleop": {
                "enabled": any(k in text for k in ("遥操", "teleop", "vr", "xr")) or True,
                "input_source": "openxr_controller" if wants_motion_controller else "openxr_hand_tracking",
                "world_T_anchor": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
                **({"hand": {"mode": "trigger_open_close"}} if wants_motion_controller else {}),
            },
        },
    }
    if scene_objects:
        spec["scene_objects"] = scene_objects
    return spec


def local_spec_builder(skill: SkillDef, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Local spec builder used when LLM is disabled or not needed."""
    _ = schema
    if skill.name == "pick_place_vr":
        return infer_pick_place_vr_spec_from_text(user_prompt)
    return {"spec_version": "0.1"}


def derive_questions_from_validation_errors(errors: list[str]) -> list[str]:
    """Convert validator errors into user-facing questions."""
    return [f"需要补齐/修正：{e}" for e in errors]


def build_spec_builder_messages(
    user_text: str,
    schema: dict[str, Any],
    draft_spec: dict[str, Any] | None,
    validation_errors: list[str],
    spec_builder_prompt: str | None,
) -> list[dict[str, str]]:
    """Build chat messages for the Spec Builder workflow."""
    system = (
        "你是 PickPlaceVRSpec 的 Spec Builder。"
        "你的任务是把用户自然语言需求整理成符合 schema 的 YAML/JSON 对象。"
        "只输出 spec 对象，不要输出解释文字。"
        "如果字段缺失但 schema 必填，请尽量从上下文补齐；无法确定时保留空值或合理占位，等待校验 gate 追问。"
    )
    if spec_builder_prompt:
        system += "\n\nSpec Builder 规则：\n" + spec_builder_prompt
    content = {
        "user_request": user_text,
        "schema": schema,
        "draft_spec": draft_spec or {},
        "validation_errors": validation_errors,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(content, indent=2, ensure_ascii=False)},
    ]


def build_clarification_resolver_messages(
    user_text: str,
    draft_intent: dict[str, Any] | None,
    resolver_prompt: str,
) -> list[dict[str, str]]:
    """Build messages for the optional LLM clarification resolver."""
    catalog = object_catalog()
    robot_catalog = load_skill_presets().get("robots", {})
    slim_catalog = {
        key: {
            "display_name": data.get("display_name"),
            "aliases": data.get("aliases", []),
            "status": data.get("status", "ready"),
            "blocked_reason": data.get("blocked_reason", ""),
        }
        for key, data in catalog.items()
        if isinstance(data, dict)
    }
    slim_robots = {
        key: {"display_name": data.get("display_name"), "aliases": data.get("aliases", [])}
        for key, data in robot_catalog.items()
        if isinstance(key, str) and isinstance(data, dict)
    }
    content = {
        "user_latest_message": user_text,
        "previous_draft_intent": draft_intent or {},
        "robot_catalog": slim_robots,
        "object_catalog": slim_catalog,
    }
    return [
        {"role": "system", "content": resolver_prompt},
        {"role": "user", "content": json.dumps(content, indent=2, ensure_ascii=False)},
    ]


def resolve_intent(
    *,
    user_text: str,
    draft_intent: dict[str, Any] | None,
    cfg: OrchestratorConfig,
    api_key: str | None,
    enable_api: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve the next draft intent, optionally using LLM for conversational updates."""
    notes: list[str] = []
    if enable_api and api_key:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "pick_place_vr_clarification_resolver.md"
        resolver_prompt = read_text(str(prompt_path))
        sys.stderr.write("[Stage A] Clarification Resolver: 正在调用模型更新 draft_intent...\n")
        sys.stderr.flush()
        try:
            messages = build_clarification_resolver_messages(user_text, draft_intent, resolver_prompt)
            payload_text = call_llm_chat_completion(cfg.llm, api_key=api_key, messages=messages)
            payload = extract_structured_object(payload_text)
            notes.append("llm clarification resolver")
            return build_intent_from_llm_payload(user_text, payload, draft_intent), notes
        except Exception as e:
            sys.stderr.write(f"[Stage A] Clarification Resolver: 模型更新失败，回退到规则解析：{e}\n")
            sys.stderr.flush()
            notes.append("rule clarification resolver after llm failure")
            return build_or_update_intent(user_text, draft_intent), notes

    notes.append("rule clarification resolver")
    return build_or_update_intent(user_text, draft_intent), notes


def run_spec_builder_workflow(
    skill: SkillDef,
    user_text: str,
    schema: dict[str, Any],
    cfg: OrchestratorConfig,
    api_key: str | None,
    enable_api: bool,
    draft_intent: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str], list[str], int, list[str], dict[str, Any] | None]:
    """Run Stage A: natural language -> spec -> preset enrichment -> schema gate."""
    resolver_notes: list[str] = []
    if skill.name == "pick_place_vr":
        intent, resolver_notes = resolve_intent(
            user_text=user_text,
            draft_intent=draft_intent,
            cfg=cfg,
            api_key=api_key,
            enable_api=enable_api,
        )
        intent_errors, intent_questions = validate_intent(intent)
        if intent_errors:
            return (
                {"spec_version": "0.1", "task": {"family": "manager_based/manipulation/pick_place"}},
                intent_errors,
                intent_questions,
                0,
                resolver_notes + ["intent safety gate"],
                intent,
            )
        spec = intent_to_spec(intent)
    else:
        spec = local_spec_builder(skill, user_text, schema)

    enrichment_notes: list[str] = []
    if skill.name == "pick_place_vr":
        spec, enrichment_notes = enrich_pick_place_vr_spec(spec)
    errors = validate_spec(skill, spec, schema)
    if not errors:
        enrichment_notes = (resolver_notes if skill.name == "pick_place_vr" else ["local spec parser"]) + enrichment_notes
        return spec, [], [], 0, enrichment_notes, intent if skill.name == "pick_place_vr" else None

    if not enable_api:
        return spec, errors, derive_questions_from_validation_errors(errors), 0, enrichment_notes, (
            intent if skill.name == "pick_place_vr" else None
        )
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {cfg.llm.api_key_env}")

    spec_builder_prompt = read_text(skill.spec_builder_prompt_path) if skill.spec_builder_prompt_path else None
    draft_spec: dict[str, Any] | None = spec
    max_iterations = max(1, cfg.workflow.max_spec_iterations)
    for i in range(1, max_iterations + 1):
        sys.stderr.write(f"[Stage A] Spec Builder {i}/{max_iterations}: 正在调用模型生成/更新 spec...\n")
        sys.stderr.flush()
        messages = build_spec_builder_messages(user_text, schema, draft_spec, errors, spec_builder_prompt)
        spec_text = call_llm_chat_completion(cfg.llm, api_key=api_key, messages=messages)
        sys.stderr.write(f"[Stage A] Spec Builder {i}/{max_iterations}: 已返回，正在解析与校验...\n")
        sys.stderr.flush()
        draft_spec = extract_structured_object(spec_text)
        if skill.name == "pick_place_vr":
            draft_spec, enrichment_notes = enrich_pick_place_vr_spec(draft_spec)
            if enrichment_notes:
                sys.stderr.write(f"[Stage A] Preset enrichment: {', '.join(enrichment_notes)}\n")
                sys.stderr.flush()
        errors = validate_spec(skill, draft_spec, schema)
        if not errors:
            return draft_spec, [], [], i, enrichment_notes, intent if skill.name == "pick_place_vr" else None
        sys.stderr.write(f"[Stage A] Spec Builder {i}/{max_iterations}: schema 未通过（{len(errors)} 条），继续补齐...\n")
        sys.stderr.flush()

    questions = derive_questions_from_validation_errors(errors)
    return draft_spec or {}, errors, questions, max_iterations, enrichment_notes, intent if skill.name == "pick_place_vr" else None
