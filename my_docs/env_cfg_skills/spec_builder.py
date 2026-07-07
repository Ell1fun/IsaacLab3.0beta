from __future__ import annotations

import json
import sys
from typing import Any

from env_utils import read_text
from llm_client import call_llm_chat_completion, extract_structured_object
from presets import enrich_pick_place_vr_spec
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
    robot_preset = ""
    if any(k in text for k in ("傅里叶", "fourier", "gr1t2", "gr1")):
        robot_preset = "fourier_gr1t2_high_pd"
    elif any(k in text for k in ("r11", "赛力斯", "seres")):
        robot_preset = "seres_r11_a2_high_pd"
    elif any(k in text for k in ("h1", "unitree h1", "宇树h1")):
        robot_preset = "unitree_h1"
    elif any(k in text for k in ("宇树", "unitree", "g1")):
        robot_preset = "unitree_g1_inspire_ftp"

    object_preset = ""
    if any(k in text for k in ("水瓶", "瓶装水", "bottle", "bottled water")):
        object_preset = "bottled_water_c01"
    elif any(k in text for k in ("方向盘", "steering")):
        object_preset = "steering_wheel"
    elif any(k in text for k in ("螺母", "nut")):
        object_preset = "factory_m16_nut_green"
    elif any(k in text for k in ("排气管", "exhaust")):
        object_preset = "exhaust_pipe"

    return {
        "spec_version": "0.1",
        "task": {"family": "manager_based/manipulation/pick_place"},
        "robot": {"preset": robot_preset},
        "scene": {},
        "object": {"preset": object_preset},
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


def run_spec_builder_workflow(
    skill: SkillDef,
    user_text: str,
    schema: dict[str, Any],
    cfg: OrchestratorConfig,
    api_key: str | None,
    enable_api: bool,
) -> tuple[dict[str, Any], list[str], list[str], int, list[str]]:
    """Run Stage A: natural language -> spec -> preset enrichment -> schema gate."""
    spec = local_spec_builder(skill, user_text, schema)
    enrichment_notes: list[str] = []
    if skill.name == "pick_place_vr":
        spec, enrichment_notes = enrich_pick_place_vr_spec(spec)
    errors = validate_spec(skill, spec, schema)
    if not errors:
        enrichment_notes = ["local spec parser"] + enrichment_notes
        return spec, [], [], 0, enrichment_notes

    if not enable_api:
        return spec, errors, derive_questions_from_validation_errors(errors), 0, enrichment_notes
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
            return draft_spec, [], [], i, enrichment_notes
        sys.stderr.write(f"[Stage A] Spec Builder {i}/{max_iterations}: schema 未通过（{len(errors)} 条），继续补齐...\n")
        sys.stderr.flush()

    questions = derive_questions_from_validation_errors(errors)
    return draft_spec or {}, errors, questions, max_iterations, enrichment_notes
