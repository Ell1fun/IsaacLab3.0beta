from __future__ import annotations

from typing import Any

from env_utils import load_yaml_or_json
from generator_workflow import run_generator_workflow, write_generated_files
from post_generation_tests import run_post_generation_tests
from skill_types import OrchestratorConfig, SkillDef
from spec_builder import run_spec_builder_workflow


def run_generate_pipeline(
    *,
    skill: SkillDef,
    user_text: str,
    cfg: OrchestratorConfig,
    api_key: str | None,
    enable_api: bool,
    draft_intent: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Run the two-workflow generate pipeline and return output plus exit code."""
    schema_obj = load_yaml_or_json(skill.schema_path)
    if not isinstance(schema_obj, dict):
        raise ValueError("Schema must be a JSON/YAML object")

    spec_obj, spec_errors, questions, iterations, enrichment_notes, updated_intent = run_spec_builder_workflow(
        skill=skill,
        user_text=user_text,
        schema=schema_obj,
        cfg=cfg,
        api_key=api_key,
        enable_api=enable_api,
        draft_intent=draft_intent,
    )

    output: dict[str, Any] = {
        "skill": skill.name,
        "llm": {
            "api_base_url": cfg.llm.api_base_url,
            "api_key_env": cfg.llm.api_key_env,
            "model": cfg.llm.model,
            "api_enabled": bool(enable_api),
        },
        "workflow": {"max_spec_iterations": cfg.workflow.max_spec_iterations, "dry_run": cfg.workflow.dry_run},
        "stage_a_spec_builder": {
            "ok": not spec_errors,
            "iterations": iterations,
            "preset_enrichment": enrichment_notes,
            "errors": spec_errors,
            "questions": questions,
            "spec": spec_obj,
            "draft_intent": updated_intent,
        },
    }

    if spec_errors:
        output["stage_b_generator"] = {
            "skipped": True,
            "reason": "Spec schema gate failed. 补齐 stage_a_spec_builder.questions 后再进入 generator。",
        }
        return output, 1

    generated = run_generator_workflow(
        skill=skill,
        spec=spec_obj,
        cfg=cfg,
        api_key=api_key,
        enable_api=enable_api,
    )
    if generated.get("ok") is False:
        output["stage_b_generator"] = {
            "skipped": True,
            "reason": generated.get("summary", "Generator returned a blocked candidate."),
            "generated": generated,
        }
        return output, 1

    post_generation_errors = run_post_generation_tests(generated, spec_obj)
    if post_generation_errors:
        output["stage_b_generator"] = {
            "skipped": True,
            "reason": "Post-generation automatic tests failed. 不写文件。",
            "post_generation_tests": {"ok": False, "errors": post_generation_errors},
            "generated": generated,
        }
        return output, 1

    written = write_generated_files(generated, dry_run=cfg.workflow.dry_run)
    output["stage_b_generator"] = {
        "skipped": False,
        "generated": generated,
        "post_generation_tests": {"ok": True, "errors": []},
        "written_files": written,
    }
    return output, 0


def concise_success_output(output: dict[str, Any]) -> dict[str, Any]:
    """Return the minimal user-facing success payload."""
    stage_b = output.get("stage_b_generator", {})
    generated = stage_b.get("generated", {}) if isinstance(stage_b, dict) else {}
    commands = generated.get("commands", []) if isinstance(generated, dict) else []
    return {
        "written_files": stage_b.get("written_files", []) if isinstance(stage_b, dict) else [],
        "commands": commands if isinstance(commands, list) else [],
    }


def concise_failure_output(output: dict[str, Any]) -> dict[str, Any]:
    """Return a compact failure payload for common blocked workflow cases."""
    stage_a = output.get("stage_a_spec_builder", {})
    if isinstance(stage_a, dict) and stage_a.get("ok") is False:
        draft_intent = stage_a.get("draft_intent", {})
        summary: dict[str, Any] = {}
        if isinstance(draft_intent, dict):
            primary = draft_intent.get("primary_object")
            scene_objects = draft_intent.get("scene_objects", [])
            unknown_objects = draft_intent.get("unknown_objects", [])
            if isinstance(primary, dict):
                summary["primary_object"] = primary.get("preset") or primary.get("label")
            if isinstance(scene_objects, list):
                summary["scene_objects"] = [
                    item.get("preset") or item.get("label") for item in scene_objects if isinstance(item, dict)
                ]
            if isinstance(unknown_objects, list):
                summary["unknown_objects"] = [
                    item.get("label") or item.get("key") for item in unknown_objects if isinstance(item, dict)
                ]
        return {
            "blocked": True,
            "stage": "stage_a_spec_builder",
            "errors": stage_a.get("errors", []),
            "questions": stage_a.get("questions", []),
            "current_intent_summary": summary,
        }

    stage_b = output.get("stage_b_generator", {})
    generated = stage_b.get("generated", {}) if isinstance(stage_b, dict) else {}
    if isinstance(generated, dict) and generated.get("generator_mode") == "adaptation_candidate":
        candidate = generated.get("adaptation_candidate", {})
        return {
            "blocked": True,
            "reason": stage_b.get("reason", generated.get("summary", "")) if isinstance(stage_b, dict) else "",
            "generator_mode": "adaptation_candidate",
            "adaptation_candidate": candidate if isinstance(candidate, dict) else {},
        }
    if isinstance(generated, dict) and str(generated.get("generator_mode", "")).startswith("adaptation_"):
        return {
            "blocked": True,
            "reason": stage_b.get("reason", generated.get("summary", "")) if isinstance(stage_b, dict) else "",
            "generator_mode": generated.get("generator_mode"),
            "post_generation_tests": stage_b.get("post_generation_tests", {}) if isinstance(stage_b, dict) else {},
            "adaptation_candidate": generated.get("adaptation_candidate", {}),
        }
    return output
