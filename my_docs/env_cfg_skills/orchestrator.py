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

    For local demos, the orchestrator may read api_key from an ignored local
    config file. For safer shared environments, prefer api_key_env.
    """

    api_base_url: str
    api_key_env: str
    model: str
    api_key: str | None = None


@dataclass(frozen=True)
class WorkflowConfig:
    """Workflow runtime knobs loaded from YAML/JSON config or CLI flags."""

    max_spec_iterations: int = 3
    dry_run: bool = True


@dataclass(frozen=True)
class OrchestratorConfig:
    """Top-level orchestrator config.

    Users can provide this via --config as YAML/JSON. A typical config looks like:

    llm:
      api_base_url: "https://your-model-api.example.com"
      api_key: "your-local-api-key"
      api_key_env: "ENV_CFG_LLM_API_KEY"
      model: "your-model-name"
    workflow:
      max_spec_iterations: 3
      dry_run: true
    """

    llm: LlmConfig
    workflow: WorkflowConfig


def _repo_root_from_this_file() -> str:
    """Return the IsaacLab repo root inferred from this file location."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _abs_from_repo_root(path_from_repo_root: str) -> str:
    """Convert a repo-root-relative path to an absolute path."""
    return os.path.join(_repo_root_from_this_file(), path_from_repo_root)


def _default_local_config_path() -> str:
    """Return the local ignored orchestrator config path."""
    return os.path.join(os.path.dirname(__file__), "orchestrator_config.local.yml")


def _skills_registry() -> list[SkillDef]:
    """Return the skill registry.

    In a production system this registry is typically loaded from config files,
    a database, or a plugin system. Here we hardcode the minimal list.
    """
    pick_place_vr_schema = _abs_from_repo_root("my_docs/env_cfg_skills/schemas/pick_place_vr_spec.schema.yml")
    spec_builder_prompt = _abs_from_repo_root("my_docs/env_cfg_skills/prompts/pick_place_vr_spec_builder.md")
    generator_prompt = _abs_from_repo_root("my_docs/env_cfg_skills/prompts/pick_place_vr_generator.md")
    return [
        SkillDef(
            name="pick_place_vr",
            description="Manager-based manipulation pick_place VR/teleop env_cfg generator (humanoid presets)",
            triggers=("pick_place", "pickplace", "vr", "teleop", "openxr", "pink_ik"),
            schema_path=pick_place_vr_schema,
            generator_prompt_path=generator_prompt if os.path.exists(generator_prompt) else None,
            spec_builder_prompt_path=spec_builder_prompt if os.path.exists(spec_builder_prompt) else None,
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


def _default_orchestrator_config() -> OrchestratorConfig:
    """Build default orchestrator config from env vars and safe defaults."""
    return OrchestratorConfig(llm=_default_llm_config(), workflow=WorkflowConfig())


def _load_orchestrator_config(path: str | None) -> OrchestratorConfig:
    """Load orchestrator config from YAML/JSON.

    Config is optional. If omitted, the orchestrator first tries:
    my_docs/env_cfg_skills/orchestrator_config.local.yml.
    If that file does not exist, environment variables and safe defaults are used.
    """
    default = _default_orchestrator_config()
    if not path:
        local_path = _default_local_config_path()
        if os.path.exists(local_path):
            path = local_path
        else:
            return default

    raw = _load_yaml_or_json(path)
    if not isinstance(raw, dict):
        raise ValueError("Orchestrator config must be a YAML/JSON object")

    llm_raw = raw.get("llm", {})
    if llm_raw is None:
        llm_raw = {}
    if not isinstance(llm_raw, dict):
        raise ValueError("Config field 'llm' must be an object")

    workflow_raw = raw.get("workflow", {})
    if workflow_raw is None:
        workflow_raw = {}
    if not isinstance(workflow_raw, dict):
        raise ValueError("Config field 'workflow' must be an object")

    llm = LlmConfig(
        api_base_url=str(llm_raw.get("api_base_url", default.llm.api_base_url)),
        api_key_env=str(llm_raw.get("api_key_env", default.llm.api_key_env)),
        model=str(llm_raw.get("model", default.llm.model)),
        api_key=str(llm_raw["api_key"]).strip() if llm_raw.get("api_key") else None,
    )
    workflow = WorkflowConfig(
        max_spec_iterations=int(workflow_raw.get("max_spec_iterations", default.workflow.max_spec_iterations)),
        dry_run=bool(workflow_raw.get("dry_run", default.workflow.dry_run)),
    )
    return OrchestratorConfig(llm=llm, workflow=workflow)


def _get_api_key(cfg: LlmConfig) -> str | None:
    """Read API key from local config first, then env var."""
    if cfg.api_key:
        return cfg.api_key
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


def _extract_structured_object(text: str) -> dict[str, Any]:
    """Parse a JSON/YAML object from model output.

    The model should ideally output only YAML/JSON. This helper also accepts fenced
    code blocks to make early experimentation less brittle.
    """
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


def _schema_text(schema: dict[str, Any]) -> str:
    """Render schema into compact JSON text for model prompts."""
    return json.dumps(schema, indent=2, ensure_ascii=False)


def _build_spec_builder_messages(
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


def _build_generator_messages(
    spec: dict[str, Any],
    generator_prompt: str | None,
) -> list[dict[str, str]]:
    """Build chat messages for the Generator workflow."""
    system = (
        "你是 Isaac Lab env_cfg 代码生成器。"
        "根据已经通过 schema 校验的 PickPlaceVRSpec 生成文件。"
        "输出必须是 JSON/YAML 对象，格式为："
        "{'files': [{'path': '相对仓库根目录的路径', 'content': '文件内容'}], "
        "'summary': '生成说明', 'commands': ['可运行命令']}。"
        "不要输出解释文字，不要使用 markdown。"
    )
    if generator_prompt:
        system += "\n\n生成规则：\n" + generator_prompt
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps({"spec": spec}, indent=2, ensure_ascii=False)},
    ]


def _run_spec_builder_workflow(
    skill: SkillDef,
    user_text: str,
    schema: dict[str, Any],
    cfg: OrchestratorConfig,
    api_key: str | None,
    enable_api: bool,
) -> tuple[dict[str, Any], list[str], list[str], int]:
    """Run Stage A: natural language -> spec -> schema gate.

    Returns (spec, validation_errors, user_questions, iterations_used).
    """
    if not enable_api:
        spec = _spec_builder_loop_placeholder(skill, user_text, schema)
        errors = validate_spec(skill, spec, schema)
        return spec, errors, _derive_questions_from_validation_errors(errors), 0

    if not api_key:
        raise RuntimeError(f"Missing API key env var: {cfg.llm.api_key_env}")

    spec_builder_prompt = _read_text(skill.spec_builder_prompt_path) if skill.spec_builder_prompt_path else None
    draft_spec: dict[str, Any] | None = None
    errors: list[str] = []
    max_iterations = max(1, cfg.workflow.max_spec_iterations)
    for i in range(1, max_iterations + 1):
        messages = _build_spec_builder_messages(user_text, schema, draft_spec, errors, spec_builder_prompt)
        spec_text = _call_llm_chat_completion(cfg.llm, api_key=api_key, messages=messages)
        draft_spec = _extract_structured_object(spec_text)
        errors = validate_spec(skill, draft_spec, schema)
        if not errors:
            return draft_spec, [], [], i

    questions = _derive_questions_from_validation_errors(errors)
    return draft_spec or {}, errors, questions, max_iterations


def _run_generator_workflow(
    skill: SkillDef,
    spec: dict[str, Any],
    cfg: OrchestratorConfig,
    api_key: str | None,
    enable_api: bool,
) -> dict[str, Any]:
    """Run Stage B: validated spec -> env_cfg files and task registration output."""
    generator_prompt = _read_text(skill.generator_prompt_path) if skill.generator_prompt_path else None

    if not enable_api:
        return {
            "files": [],
            "summary": "dry-run placeholder: spec 已通过 schema gate，但未调用 LLM API 生成 env_cfg。",
            "commands": [],
        }

    if not api_key:
        raise RuntimeError(f"Missing API key env var: {cfg.llm.api_key_env}")

    messages = _build_generator_messages(spec, generator_prompt)
    generated_text = _call_llm_chat_completion(cfg.llm, api_key=api_key, messages=messages)
    generated = _extract_structured_object(generated_text)
    if "files" not in generated:
        generated["files"] = []
    if "summary" not in generated:
        generated["summary"] = "LLM returned generated payload without summary."
    if "commands" not in generated:
        generated["commands"] = []
    return generated


def _write_generated_files(generated: dict[str, Any], dry_run: bool) -> list[str]:
    """Write generated files under repo root when dry_run is False."""
    written: list[str] = []
    files = generated.get("files", [])
    if not isinstance(files, list):
        raise ValueError("generated.files must be a list")

    repo_root = _repo_root_from_this_file()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each generated file entry must be an object")
        rel_path = item.get("path")
        content = item.get("content")
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError("Each generated file needs a non-empty string path")
        if not isinstance(content, str):
            raise ValueError(f"Generated file {rel_path} needs string content")

        abs_path = os.path.abspath(os.path.join(repo_root, rel_path))
        if not abs_path.startswith(repo_root + os.sep):
            raise ValueError(f"Refusing to write outside repo root: {rel_path}")
        if dry_run:
            written.append(f"DRY_RUN:{abs_path}")
            continue
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(abs_path)
    return written


def _runtime_config_from_args(args: argparse.Namespace) -> OrchestratorConfig:
    """Create runtime config from --config plus CLI overrides.

    For safety, file writes are enabled only when --write is provided. The YAML
    dry_run field is documented as a default, but CLI --write is the explicit gate
    for this demo entrypoint.
    """
    cfg = _load_orchestrator_config(getattr(args, "config", None))
    llm_cfg = cfg.llm
    if getattr(args, "api_base_url", None) is not None:
        llm_cfg = LlmConfig(api_base_url=args.api_base_url, api_key_env=llm_cfg.api_key_env, model=llm_cfg.model, api_key=llm_cfg.api_key)
    if getattr(args, "model", None) is not None:
        llm_cfg = LlmConfig(api_base_url=llm_cfg.api_base_url, api_key_env=llm_cfg.api_key_env, model=args.model, api_key=llm_cfg.api_key)
    return OrchestratorConfig(
        llm=llm_cfg,
        workflow=WorkflowConfig(max_spec_iterations=cfg.workflow.max_spec_iterations, dry_run=not bool(getattr(args, "write", False))),
    )


def _resolve_skill(registry: dict[str, SkillDef], user_text: str, skill_name: str | None) -> SkillDef | None:
    """Resolve skill from explicit name or user text routing."""
    if skill_name:
        return registry.get(skill_name)
    skill, _scores = route_skill(user_text)
    return skill


def _run_generate_pipeline(
    *,
    skill: SkillDef,
    user_text: str,
    cfg: OrchestratorConfig,
    api_key: str | None,
    enable_api: bool,
) -> tuple[dict[str, Any], int]:
    """Run the two-workflow generate pipeline and return output plus exit code."""
    schema_obj = _load_yaml_or_json(skill.schema_path)
    if not isinstance(schema_obj, dict):
        raise ValueError("Schema must be a JSON/YAML object")

    spec_obj, spec_errors, questions, iterations = _run_spec_builder_workflow(
        skill=skill,
        user_text=user_text,
        schema=schema_obj,
        cfg=cfg,
        api_key=api_key,
        enable_api=enable_api,
    )

    output: dict[str, Any] = {
        "skill": skill.name,
        "llm": {
            "api_base_url": cfg.llm.api_base_url,
            "api_key_env": cfg.llm.api_key_env,
            "model": cfg.llm.model,
            "api_enabled": bool(enable_api),
        },
        "workflow": {
            "max_spec_iterations": cfg.workflow.max_spec_iterations,
            "dry_run": cfg.workflow.dry_run,
        },
        "stage_a_spec_builder": {
            "ok": not spec_errors,
            "iterations": iterations,
            "errors": spec_errors,
            "questions": questions,
            "spec": spec_obj,
        },
    }

    if spec_errors:
        output["stage_b_generator"] = {
            "skipped": True,
            "reason": "Spec schema gate failed. 补齐 stage_a_spec_builder.questions 后再进入 generator。",
        }
        return output, 1

    generated = _run_generator_workflow(
        skill=skill,
        spec=spec_obj,
        cfg=cfg,
        api_key=api_key,
        enable_api=enable_api,
    )
    written = _write_generated_files(generated, dry_run=cfg.workflow.dry_run)
    output["stage_b_generator"] = {
        "skipped": False,
        "generated": generated,
        "written_files": written,
    }
    return output, 0


def _run_chat_loop(
    *,
    registry: dict[str, SkillDef],
    cfg: OrchestratorConfig,
    skill_name: str | None,
    enable_api: bool,
    fmt: str,
) -> int:
    """Interactive demo loop.

    This is the user-friendly local demo entrypoint: run once, then type prompts
    repeatedly. It shares the same two-workflow pipeline as `generate`.
    """
    api_key = _get_api_key(cfg.llm)
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
            selected_skill = _resolve_skill(registry, combined_text, None)
            if selected_skill is None:
                sys.stdout.write("还没有匹配到 skill。你可以继续补充任务类型，例如 pick_place / VR / teleop。\n")
                continue
            sys.stdout.write(f"已路由到 skill: {selected_skill.name}\n")

        try:
            output, exit_code = _run_generate_pipeline(
                skill=selected_skill,
                user_text=combined_text,
                cfg=cfg,
                api_key=api_key,
                enable_api=enable_api,
            )
        except Exception as e:
            sys.stderr.write(f"Workflow failed: {e}\n")
            return 1

        sys.stdout.write(_dump_yaml_or_json(output, fmt) + "\n")
        if exit_code == 0:
            sys.stdout.write("流程结束。输入新的需求可以重新开始；输入 /quit 退出。\n")
            messages = []
            selected_skill = registry.get(skill_name) if skill_name else None
        else:
            questions = output.get("stage_a_spec_builder", {}).get("questions", [])
            if questions:
                sys.stdout.write("请继续回答上面的补齐问题。\n")


def main(argv: list[str]) -> int:
    """CLI entrypoint.

    Commands:
    - route: choose a skill from user text and print resolved asset paths
    - template: print a starter spec template for a given skill
    - validate: validate a spec file against the skill schema
    - generate: run two workflows:
      1) Spec Builder loop -> schema gate
      2) Generator -> env_cfg files/task registration payload -> optional file writes
    - chat: interactive local demo loop over the same two workflows
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
        try:
            cfg = _runtime_config_from_args(args)
        except Exception as e:
            sys.stderr.write(f"Failed to load orchestrator config: {e}\n")
            return 2

        skill = _resolve_skill(registry, user_text, args.skill)
        if skill is None:
            sys.stderr.write("Unable to route skill from user text. Provide --skill explicitly.\n")
            return 2

        api_key = _get_api_key(cfg.llm)
        if args.enable_api and not api_key:
            sys.stderr.write(f"Missing API key env var: {cfg.llm.api_key_env}\n")
            return 2

        try:
            output, exit_code = _run_generate_pipeline(
                skill=skill,
                user_text=user_text,
                cfg=cfg,
                api_key=api_key,
                enable_api=args.enable_api,
            )
        except Exception as e:
            sys.stderr.write(f"Generate workflow failed: {e}\n")
            return 1

        sys.stdout.write(_dump_yaml_or_json(output, args.format) + "\n")
        return exit_code

    if args.cmd == "chat":
        try:
            cfg = _runtime_config_from_args(args)
        except Exception as e:
            sys.stderr.write(f"Failed to load orchestrator config: {e}\n")
            return 2
        return _run_chat_loop(
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
