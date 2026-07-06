from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SkillDef:
    """A registered skill definition.

    This is the minimal metadata an orchestrator needs to:
    - route a user request to a skill (via triggers)
    - locate the skill's schema and prompt assets (via file paths)
    """

    name: str
    description: str
    triggers: tuple[str, ...]
    schema_path: str
    generator_prompt_path: str | None = None
    spec_builder_prompt_path: str | None = None


@dataclass(frozen=True)
class LlmConfig:
    """LLM API configuration.

    The orchestrator reads the API key from environment variables to avoid
    storing secrets in the repo. Do not hardcode API keys into files.
    """

    api_base_url: str
    api_key_env: str
    model: str


def _repo_root_from_this_file() -> str:
    """Return the IsaacLab repo root inferred from this file location."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _abs_from_repo_root(path_from_repo_root: str) -> str:
    """Convert a repo-root-relative path to an absolute path."""
    return os.path.join(_repo_root_from_this_file(), path_from_repo_root)


def _skills_registry() -> list[SkillDef]:
    """Return the skill registry.

    In a production system this registry is typically loaded from config files,
    a database, or a plugin system. Here we hardcode the minimal list.
    """
    pick_place_vr_schema = _abs_from_repo_root("my_docs/env_cfg_skills/schemas/pick_place_vr_spec.schema.yml")
    env_cfg_generator_prompt = _abs_from_repo_root("my_docs/env_cfg_skills/prompts/env_cfg_generator.md")
    return [
        SkillDef(
            name="pick_place_vr",
            description="Manager-based manipulation pick_place VR/teleop env_cfg generator (humanoid presets)",
            triggers=("pick_place", "pickplace", "vr", "teleop", "openxr", "pink_ik"),
            schema_path=pick_place_vr_schema,
            generator_prompt_path=env_cfg_generator_prompt if os.path.exists(env_cfg_generator_prompt) else None,
            spec_builder_prompt_path=None,
        )
    ]


def _read_text(path: str) -> str:
    """Read a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_yaml_or_json(path: str) -> Any:
    """Load YAML/JSON into Python objects based on file extension."""
    suffix = os.path.splitext(path)[1].lower()
    text = _read_text(path)
    if suffix in (".json",):
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("YAML parsing requires PyYAML (import yaml failed)") from e
    return yaml.safe_load(text)


def _dump_yaml_or_json(data: Any, fmt: str) -> str:
    """Dump Python objects to YAML/JSON strings (for CLI output)."""
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False)
    if fmt == "yml" or fmt == "yaml":
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML dumping requires PyYAML (import yaml failed)") from e
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    raise ValueError(f"Unsupported format: {fmt}")


def route_skill(user_text: str) -> tuple[SkillDef | None, dict[str, int]]:
    """Route user text to the most likely skill.

    The current routing strategy is a simple keyword hit-count:
    - each trigger keyword contained in the user text contributes +1 score
    - choose the highest scoring skill
    """
    text = user_text.lower()
    scores: dict[str, int] = {}
    best: SkillDef | None = None
    best_score = 0
    for skill in _skills_registry():
        score = sum(1 for t in skill.triggers if t in text)
        scores[skill.name] = score
        if score > best_score:
            best_score = score
            best = skill
    if best_score == 0:
        return None, scores
    return best, scores


def _basic_pick_place_vr_checks(spec: dict[str, Any]) -> list[str]:
    """A minimal validator used when jsonschema is unavailable.

    This is intentionally lightweight: it checks only key required fields and a few
    if-then constraints for the pick_place_vr spec.
    """
    errors: list[str] = []

    for k in ("spec_version", "task", "robot", "scene", "object", "target", "control"):
        if k not in spec:
            errors.append(f"Missing required top-level field: {k}")

    control = spec.get("control") if isinstance(spec.get("control"), dict) else {}
    teleop = control.get("teleop") if isinstance(control.get("teleop"), dict) else {}
    teleop_enabled = bool(teleop.get("enabled"))
    if teleop_enabled:
        if not teleop.get("input_source"):
            errors.append("control.teleop.enabled=true requires: control.teleop.input_source")
        if not teleop.get("world_T_anchor"):
            errors.append("control.teleop.enabled=true requires: control.teleop.world_T_anchor")
        robot = spec.get("robot") if isinstance(spec.get("robot"), dict) else {}
        idle = robot.get("idle_wrist_pose") if isinstance(robot.get("idle_wrist_pose"), dict) else {}
        if not idle.get("left") or not idle.get("right"):
            errors.append("control.teleop.enabled=true requires: robot.idle_wrist_pose.left and robot.idle_wrist_pose.right")

    hand = teleop.get("hand") if isinstance(teleop.get("hand"), dict) else {}
    if hand.get("mode") == "dexpilot":
        if not hand.get("dexpilot_config_path"):
            errors.append("control.teleop.hand.mode=dexpilot requires: control.teleop.hand.dexpilot_config_path")

    return errors


def validate_spec(skill: SkillDef, spec: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate spec against schema.

    Returns a list of error messages. Empty list means validation passed.
    """
    try:
        import jsonschema  # type: ignore
    except Exception:
        if skill.name == "pick_place_vr":
            return _basic_pick_place_vr_checks(spec)
        return ["jsonschema is not installed; no validator available for this skill"]

    try:
        jsonschema.validate(instance=spec, schema=schema)
    except jsonschema.ValidationError as e:  # type: ignore[attr-defined]
        return [e.message]
    return []


def _template_pick_place_vr_spec() -> dict[str, Any]:
    """Return a starter spec template for quick iteration/debugging."""
    return {
        "spec_version": "0.1",
        "task": {"family": "manager_based/manipulation/pick_place"},
        "robot": {
            "preset": "unitree_g1_inspire_ftp",
            "eef_link_names": {"left": "", "right": ""},
            "idle_wrist_pose": {
                "left": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
                "right": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
        },
        "scene": {
            "ground": {"kind": "ground_plane"},
            "table": {
                "usd_path": "",
                "pose": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
        },
        "object": {"usd_path": "", "pose": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "target": {"pose": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "control": {
            "mode": "pink_ik",
            "teleop": {
                "enabled": True,
                "input_source": "openxr_hand_tracking",
                "world_T_anchor": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
        },
    }


def _default_llm_config() -> LlmConfig:
    """Build default LLM config from environment variables."""
    api_base_url = os.environ.get("ENV_CFG_LLM_API_BASE_URL", "").strip()
    if not api_base_url:
        api_base_url = "https://example.invalid"
    return LlmConfig(api_base_url=api_base_url, api_key_env="ENV_CFG_LLM_API_KEY", model=os.environ.get("ENV_CFG_LLM_MODEL", "gpt-4.1"))


def _get_api_key(cfg: LlmConfig) -> str | None:
    """Read API key from env var. Returns None if not set."""
    key = os.environ.get(cfg.api_key_env, "")
    key = key.strip()
    return key or None


def _call_llm_chat_completion(cfg: LlmConfig, api_key: str, messages: list[dict[str, str]]) -> str:
    """Call an OpenAI-compatible chat completion endpoint and return assistant text.

    This is an optional integration. It is only used when the user explicitly enables
    API calls via CLI flags.
    """
    url = cfg.api_base_url.rstrip("/") + "/v1/chat/completions"
    payload = {"model": cfg.model, "messages": messages, "temperature": 0.2}
    body = json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"LLM API call failed: {e}") from e

    try:
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError("Unexpected LLM API response format") from e


def _spec_builder_loop_placeholder(skill: SkillDef, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Placeholder spec builder.

    In a production orchestrator this would:
    - call LLM to propose/patch the spec
    - validate against schema
    - produce follow-up questions for missing fields
    Here we return a template to keep the orchestrator runnable without any LLM.
    """
    _ = user_prompt, schema
    if skill.name == "pick_place_vr":
        return _template_pick_place_vr_spec()
    return {"spec_version": "0.1"}


def _derive_questions_from_validation_errors(errors: list[str]) -> list[str]:
    """Convert validator errors into user-facing questions (very lightweight)."""
    questions: list[str] = []
    for e in errors:
        questions.append(f"需要补齐/修正：{e}")
    return questions


def main(argv: list[str]) -> int:
    """CLI entrypoint.

    Commands:
    - route: choose a skill from user text and print resolved asset paths
    - template: print a starter spec template for a given skill
    - validate: validate a spec file against the skill schema
    - generate: run a two-stage workflow (spec builder loop -> validation gate) and print a spec draft
    """
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
    p_generate.add_argument("--enable_api", action="store_true")
    p_generate.add_argument("--api_base_url", default=None)
    p_generate.add_argument("--model", default=None)

    args = parser.parse_args(argv)
    registry = {s.name: s for s in _skills_registry()}

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
        if args.skill:
            if args.skill not in registry:
                sys.stderr.write(f"Unknown skill: {args.skill}\n")
                sys.stderr.write(f"Available: {', '.join(sorted(registry.keys()))}\n")
                return 2
            skill = registry[args.skill]
        else:
            skill, _scores = route_skill(user_text)
            if skill is None:
                sys.stderr.write("Unable to route skill from user text. Provide --skill explicitly.\n")
                return 2

        schema_obj = _load_yaml_or_json(skill.schema_path)
        if not isinstance(schema_obj, dict):
            sys.stderr.write("Schema must be a JSON/YAML object\n")
            return 2

        llm_cfg = _default_llm_config()
        if args.api_base_url is not None:
            llm_cfg = LlmConfig(api_base_url=args.api_base_url, api_key_env=llm_cfg.api_key_env, model=llm_cfg.model)
        if args.model is not None:
            llm_cfg = LlmConfig(api_base_url=llm_cfg.api_base_url, api_key_env=llm_cfg.api_key_env, model=args.model)

        api_key = _get_api_key(llm_cfg)
        if args.enable_api and not api_key:
            sys.stderr.write(f"Missing API key env var: {llm_cfg.api_key_env}\n")
            return 2

        if args.enable_api:
            messages = [
                {"role": "system", "content": "You are a spec builder. Output only the spec content, no extra text."},
                {"role": "user", "content": user_text},
            ]
            spec_text = _call_llm_chat_completion(llm_cfg, api_key=api_key or "", messages=messages)
            spec_obj = {"raw": spec_text}
            sys.stdout.write(_dump_yaml_or_json(spec_obj, args.format) + "\n")
            return 0

        spec_obj = _spec_builder_loop_placeholder(skill, user_text, schema_obj)
        if not isinstance(spec_obj, dict):
            sys.stderr.write("Spec builder returned non-object\n")
            return 2
        errors = validate_spec(skill, spec_obj, schema_obj)
        questions = _derive_questions_from_validation_errors(errors) if errors else []
        output = {"skill": skill.name, "ok": not errors, "questions": questions, "spec": spec_obj}
        sys.stdout.write(_dump_yaml_or_json(output, args.format) + "\n")
        return 0

    if args.skill not in registry:
        sys.stderr.write(f"Unknown skill: {args.skill}\n")
        sys.stderr.write(f"Available: {', '.join(sorted(registry.keys()))}\n")
        return 2

    skill = registry[args.skill]

    if args.cmd == "template":
        if skill.name == "pick_place_vr":
            sys.stdout.write(_dump_yaml_or_json(_template_pick_place_vr_spec(), args.format) + "\n")
            return 0
        sys.stderr.write(f"No template available for skill: {skill.name}\n")
        return 2

    if args.cmd == "validate":
        spec_obj = _load_yaml_or_json(args.spec)
        if not isinstance(spec_obj, dict):
            sys.stderr.write("Spec must be a JSON/YAML object\n")
            return 2
        schema_obj = _load_yaml_or_json(skill.schema_path)
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
    """Invoke the CLI with process argv.

    This script is meant to be a minimal, explicit orchestrator entrypoint:
    it does not run a server and it does not call any LLM by default.
    """
    raise SystemExit(main(sys.argv[1:]))
