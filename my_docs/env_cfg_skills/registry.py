from __future__ import annotations

import os

from env_utils import abs_from_repo_root
from skill_types import SkillDef


def skills_registry() -> list[SkillDef]:
    """Return the local skill registry."""
    pick_place_vr_schema = abs_from_repo_root("my_docs/env_cfg_skills/schemas/pick_place_vr_spec.schema.yml")
    spec_builder_prompt = abs_from_repo_root("my_docs/env_cfg_skills/prompts/pick_place_vr_spec_builder.md")
    generator_prompt = abs_from_repo_root("my_docs/env_cfg_skills/prompts/pick_place_vr_generator.md")
    return [
        SkillDef(
            name="pick_place_vr",
            description="Manager-based manipulation pick_place VR/teleop env_cfg generator (humanoid presets)",
            triggers=(
                "pick_place",
                "pickplace",
                "vr",
                "teleop",
                "openxr",
                "pink_ik",
                "抓取",
                "遥操",
                "方向盘",
                "宇树",
                "傅里叶",
                "unitree",
                "fourier",
            ),
            schema_path=pick_place_vr_schema,
            generator_prompt_path=generator_prompt if os.path.exists(generator_prompt) else None,
            spec_builder_prompt_path=spec_builder_prompt if os.path.exists(spec_builder_prompt) else None,
        )
    ]


def route_skill(user_text: str) -> tuple[SkillDef | None, dict[str, int]]:
    """Route user text to the most likely skill using keyword hit counts."""
    text = user_text.lower()
    scores: dict[str, int] = {}
    best: SkillDef | None = None
    best_score = 0
    for skill in skills_registry():
        score = sum(1 for t in skill.triggers if t in text)
        scores[skill.name] = score
        if score > best_score:
            best_score = score
            best = skill
    if best_score == 0:
        return None, scores
    return best, scores


def resolve_skill(registry: dict[str, SkillDef], user_text: str, skill_name: str | None) -> SkillDef | None:
    """Resolve skill from explicit name or user text routing."""
    if skill_name:
        return registry.get(skill_name)
    skill, _scores = route_skill(user_text)
    return skill
