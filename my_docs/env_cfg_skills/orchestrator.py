from __future__ import annotations

import argparse
import json
import sys

from config_loader import runtime_config_from_args
from env_utils import dump_yaml_or_json, load_yaml_or_json
from llm_client import get_api_key
from pipeline import concise_failure_output, concise_success_output, run_generate_pipeline
from registry import resolve_skill, route_skill, skills_registry
from schema_validation import validate_spec
from skill_types import OrchestratorConfig, SkillDef
from starter_specs import template_pick_place_vr_spec


def run_chat_loop(
    *,
    registry: dict[str, SkillDef],
    cfg: OrchestratorConfig,
    skill_name: str | None,
    enable_api: bool,
    fmt: str,
) -> int:
    """Interactive demo loop over the same two-workflow pipeline as `generate`."""
    api_key = get_api_key(cfg.llm)
    if enable_api and not api_key:
        sys.stderr.write(f"Missing API key env var: {cfg.llm.api_key_env}\n")
        return 2

    sys.stdout.write("env_cfg skill orchestrator chat demo\n")
    sys.stdout.write("输入自然语言需求；输入 /quit 退出。\n")
    sys.stdout.write("提示：当前 demo 会在 spec 通过 schema gate 后自动进入 generator。\n\n")

    messages: list[str] = []
    selected_skill: SkillDef | None = registry.get(skill_name) if skill_name else None
    if skill_name and selected_skill is None:
        sys.stderr.write(f"Unknown skill: {skill_name}\n")
        sys.stderr.write(f"Available: {', '.join(sorted(registry.keys()))}\n")
        return 2

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

        messages.append(user_text)
        combined_text = "\n".join(messages)
        if selected_skill is None:
            selected_skill = resolve_skill(registry, combined_text, None)
            if selected_skill is None:
                sys.stdout.write("还没有匹配到 skill。你可以继续补充任务类型，例如 pick_place / VR / teleop。\n")
                continue
            sys.stdout.write(f"已路由到 skill: {selected_skill.name}\n")

        try:
            output, exit_code = run_generate_pipeline(
                skill=selected_skill,
                user_text=combined_text,
                cfg=cfg,
                api_key=api_key,
                enable_api=enable_api,
            )
        except Exception as e:
            sys.stderr.write(f"Workflow failed: {e}\n")
            return 1

        if exit_code == 0:
            sys.stdout.write(dump_yaml_or_json(concise_success_output(output), fmt) + "\n")
            messages = []
            selected_skill = registry.get(skill_name) if skill_name else None
        else:
            sys.stdout.write(dump_yaml_or_json(concise_failure_output(output), fmt) + "\n")
            questions = output.get("stage_a_spec_builder", {}).get("questions", [])
            if questions:
                sys.stdout.write("请继续回答上面的补齐问题。\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the orchestrator CLI parser."""
    parser = argparse.ArgumentParser(prog="env_cfg_skills.orchestrator", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_route = sub.add_parser("route")
    p_route.add_argument("text", nargs="+")

    p_validate = sub.add_parser("validate")
    p_validate.add_argument("--skill", required=True)
    p_validate.add_argument("--spec", required=True)

    p_template = sub.add_parser("template")
    p_template.add_argument("--skill", required=True)
    p_template.add_argument("--format", default="yml", choices=("yml", "yaml", "json"))

    p_generate = sub.add_parser("generate")
    p_generate.add_argument("text", nargs="+")
    p_generate.add_argument("--skill", default=None)
    p_generate.add_argument("--format", default="yml", choices=("yml", "yaml", "json"))
    p_generate.add_argument("--config", default=None, help="YAML/JSON config with LLM and workflow settings")
    p_generate.add_argument("--enable_api", action="store_true")
    p_generate.add_argument("--api_base_url", default=None)
    p_generate.add_argument("--model", default=None)
    p_generate.add_argument("--write", action="store_true", help="Actually write generated files. Default is dry-run.")

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--skill", default=None)
    p_chat.add_argument("--format", default="yml", choices=("yml", "yaml", "json"))
    p_chat.add_argument("--config", default=None, help="YAML/JSON config with LLM and workflow settings")
    p_chat.add_argument("--enable_api", action="store_true")
    p_chat.add_argument("--api_base_url", default=None)
    p_chat.add_argument("--model", default=None)
    p_chat.add_argument("--write", action="store_true", help="Actually write generated files. Default is dry-run.")
    return parser


def main(argv: list[str]) -> int:
    """CLI entrypoint: route, template, validate, generate, or chat."""
    parser = build_parser()
    args = parser.parse_args(argv)
    registry = {s.name: s for s in skills_registry()}

    if args.cmd == "route":
        user_text = " ".join(args.text)
        best, scores = route_skill(user_text)
        payload = {
            "selected": best.name if best else None,
            "scores": scores,
            "schema_path": best.schema_path if best else None,
            "spec_builder_prompt_path": best.spec_builder_prompt_path if best else None,
            "generator_prompt_path": best.generator_prompt_path if best else None,
        }
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return 0

    if args.cmd == "generate":
        user_text = " ".join(args.text)
        try:
            cfg = runtime_config_from_args(args)
        except Exception as e:
            sys.stderr.write(f"Failed to load orchestrator config: {e}\n")
            return 2

        skill = resolve_skill(registry, user_text, args.skill)
        if skill is None:
            sys.stderr.write("Unable to route skill from user text. Provide --skill explicitly.\n")
            return 2

        api_key = get_api_key(cfg.llm)
        if args.enable_api and not api_key:
            sys.stderr.write(f"Missing API key env var: {cfg.llm.api_key_env}\n")
            return 2

        try:
            output, exit_code = run_generate_pipeline(
                skill=skill,
                user_text=user_text,
                cfg=cfg,
                api_key=api_key,
                enable_api=args.enable_api,
            )
        except Exception as e:
            sys.stderr.write(f"Generate workflow failed: {e}\n")
            return 1

        if exit_code == 0:
            output = concise_success_output(output)
        else:
            output = concise_failure_output(output)
        sys.stdout.write(dump_yaml_or_json(output, args.format) + "\n")
        return exit_code

    if args.cmd == "chat":
        try:
            cfg = runtime_config_from_args(args)
        except Exception as e:
            sys.stderr.write(f"Failed to load orchestrator config: {e}\n")
            return 2
        return run_chat_loop(
            registry=registry,
            cfg=cfg,
            skill_name=args.skill,
            enable_api=args.enable_api,
            fmt=args.format,
        )

    if args.skill not in registry:
        sys.stderr.write(f"Unknown skill: {args.skill}\n")
        sys.stderr.write(f"Available: {', '.join(sorted(registry.keys()))}\n")
        return 2

    skill = registry[args.skill]

    if args.cmd == "template":
        if skill.name == "pick_place_vr":
            sys.stdout.write(dump_yaml_or_json(template_pick_place_vr_spec(), args.format) + "\n")
            return 0
        sys.stderr.write(f"No template available for skill: {skill.name}\n")
        return 2

    if args.cmd == "validate":
        spec_obj = load_yaml_or_json(args.spec)
        if not isinstance(spec_obj, dict):
            sys.stderr.write("Spec must be a JSON/YAML object\n")
            return 2
        schema_obj = load_yaml_or_json(skill.schema_path)
        if not isinstance(schema_obj, dict):
            sys.stderr.write("Schema must be a JSON/YAML object\n")
            return 2
        errors = validate_spec(skill, spec_obj, schema_obj)
        if errors:
            sys.stdout.write(json.dumps({"ok": False, "errors": errors}, indent=2, ensure_ascii=False) + "\n")
            return 1
        sys.stdout.write(json.dumps({"ok": True}, indent=2, ensure_ascii=False) + "\n")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
