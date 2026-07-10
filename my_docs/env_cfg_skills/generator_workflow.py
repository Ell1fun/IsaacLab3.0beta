from __future__ import annotations

import ast
import json
import os
import sys
from typing import Any

from env_utils import abs_from_repo_root, is_missing_value, read_text, repo_root_from_this_file
from llm_client import call_llm_chat_completion, extract_structured_object
from presets import load_skill_presets
from skill_types import OrchestratorConfig, SkillDef


ADAPTATION_PROMPT_PATH = abs_from_repo_root(
    "my_docs/env_cfg_skills/prompts/pick_place_vr_template_adaptation.md"
)


def teleop_command_for_spec(spec: dict[str, Any]) -> str:
    """Build the final teleop command from the registered task id."""
    task = spec.get("task", {})
    gym_id = task.get("gym_id", "<task.gym_id>")
    return (
        "./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py "
        f"--task {gym_id} "
        "--visualizer kit "
        "--xr "
        "--cloudxr_env cloudxrjs"
    )


def python_asset_expr(path: str) -> str:
    """Convert preset asset path syntax like ${ISAAC_NUCLEUS_DIR}/x.usd to Python code."""
    for var in ("ISAAC_NUCLEUS_DIR", "ISAACLAB_NUCLEUS_DIR"):
        prefix = "${" + var + "}/"
        if path.startswith(prefix):
            return f'f"{{{var}}}/{path[len(prefix):]}"'
    return repr(path)


def scene_objects_from_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated-looking additional scene objects from the spec."""
    scene_objects = spec.get("scene_objects", [])
    return [item for item in scene_objects if isinstance(item, dict)] if isinstance(scene_objects, list) else []


def render_scene_object_cfg_lines(scene_object: dict[str, Any]) -> list[str]:
    """Render a RigidObjectCfg assignment for an additional scene object."""
    name = str(scene_object.get("name", "scene_object"))
    usd_path = str(scene_object.get("usd_path", ""))
    if is_missing_value(usd_path):
        raise RuntimeError(f"scene_objects.{name}.usd_path 仍为空或 TODO，不能模板化生成 env_cfg。")
    pose = scene_object.get("pose", {}) if isinstance(scene_object.get("pose"), dict) else {}
    pos = pose.get("pos", [0.0, 0.0, 0.0])
    quat = pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    scale = scene_object.get("scale")
    mass = scene_object.get("mass", 0.05)
    single_rigid_body = bool(scene_object.get("single_rigid_body"))
    prim_suffix = "".join(part[:1].upper() + part[1:] for part in name.split("_") if part) or "SceneObject"
    lines = [
        f"        self.scene.{name} = RigidObjectCfg(",
        f'            prim_path="{{ENV_REGEX_NS}}/SceneObjects/{prim_suffix}",',
        f"            init_state=RigidObjectCfg.InitialStateCfg(pos={tuple(pos)!r}, rot={tuple(quat)!r}),",
    ]
    if single_rigid_body:
        lines.extend(
            [
                "            spawn=SingleRigidBodyUsdFileCfg(",
                f"                usd_path={python_asset_expr(usd_path)},",
                f"                scale={tuple(scale)!r}," if scale else "                scale=None,",
                "                rigid_props=PhysxRigidBodyPropertiesCfg(),",
                f"                mass_props=MassPropertiesCfg(mass={float(mass)!r}),",
                "            ),",
            ]
        )
    else:
        lines.extend(
            [
                "            spawn=UsdFileCfg(",
                f"                usd_path={python_asset_expr(usd_path)},",
                f"                scale={tuple(scale)!r}," if scale else "",
                "            ),",
            ]
        )
    lines.append("        )")
    return [line for line in lines if line]


def auto_env_class_name(spec: dict[str, Any]) -> str:
    """Build a deterministic AutoPickPlace...EnvCfg class name from task.gym_id."""
    gym_id = str(spec.get("task", {}).get("gym_id", "Isaac-Auto-PickPlace-Task-Teleop-v0"))
    middle = gym_id.removeprefix("Isaac-Auto-PickPlace-").removesuffix("-v0")
    parts = [p for p in middle.replace("-", "_").split("_") if p and p.lower() != "teleop"]
    return "AutoPickPlace" + "".join(part[:1].upper() + part[1:] for part in parts) + "TeleopEnvCfg"


def input_source_from_spec(spec: dict[str, Any]) -> str:
    """Return the requested teleop input source from the spec."""
    return str(spec.get("control", {}).get("teleop", {}).get("input_source", "openxr_hand_tracking"))


def template_base_for_spec(spec: dict[str, Any]) -> dict[str, str] | None:
    """Return the exact env_cfg template for (robot preset, input source)."""
    robot_preset = str(spec.get("robot", {}).get("preset", ""))
    input_source = input_source_from_spec(spec)
    presets = load_skill_presets()
    robot = presets.get("robots", {}).get(robot_preset, {})
    templates = robot.get("supported_teleop_templates", {}) if isinstance(robot, dict) else {}
    template = templates.get(input_source) if isinstance(templates, dict) else None
    return template if isinstance(template, dict) else None


def find_reference_template_for_input_source(spec: dict[str, Any]) -> dict[str, str] | None:
    """Find another robot preset that supports the requested input source."""
    requested_robot = str(spec.get("robot", {}).get("preset", ""))
    input_source = input_source_from_spec(spec)
    presets = load_skill_presets()
    for robot_name, robot in presets.get("robots", {}).items():
        if robot_name == requested_robot or not isinstance(robot, dict):
            continue
        templates = robot.get("supported_teleop_templates", {})
        if isinstance(templates, dict) and input_source in templates:
            template = templates[input_source]
            if isinstance(template, dict):
                return {
                    "robot_preset": str(robot_name),
                    "input_source": input_source,
                    "module": str(template.get("adaptation_reference_module", template.get("module", ""))),
                    "class": str(template.get("adaptation_reference_class", template.get("class", ""))),
                    "reference_env_cfg": str(
                        template.get("adaptation_reference_env_cfg", robot.get("reference_env_cfg", ""))
                    ),
                    "runtime_template_module": str(template.get("module", "")),
                    "runtime_template_class": str(template.get("class", "")),
                }
    return None


def robot_preset_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the resolved robot preset dict for the requested robot."""
    robot_preset = str(spec.get("robot", {}).get("preset", ""))
    presets = load_skill_presets()
    robot = presets.get("robots", {}).get(robot_preset, {})
    return robot if isinstance(robot, dict) else {}


def output_paths_for_spec(spec: dict[str, Any]) -> dict[str, str]:
    """Return generator output paths configured by the robot preset."""
    robot = robot_preset_for_spec(spec)
    default_output_dir = "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place"
    output_dir = str(robot.get("output_dir", default_output_dir) or default_output_dir)
    registration_module = str(robot.get("registration_module", f"{output_dir}/__init__.py") or f"{output_dir}/__init__.py")
    return {"output_dir": output_dir, "registration_module": registration_module}


def read_repo_file_if_exists(path_from_repo_root: str) -> str:
    """Read a repository file if the preset points to an existing path."""
    if not path_from_repo_root:
        return ""
    abs_path = abs_from_repo_root(path_from_repo_root)
    if not os.path.exists(abs_path):
        return ""
    return read_text(abs_path)


def adaptation_focus_template_for_robot(robot: dict[str, Any]) -> dict[str, str] | None:
    """Return the robot's closest existing teleop template for adaptation context."""
    templates = robot.get("supported_teleop_templates", {}) if isinstance(robot, dict) else {}
    if not isinstance(templates, dict) or not templates:
        return None
    preferred = templates.get("openxr_hand_tracking")
    if isinstance(preferred, dict):
        return preferred
    for template in templates.values():
        if isinstance(template, dict):
            return template
    return None


def _node_source(lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    return "\n".join(lines[start - 1 : end])


def _compact_code_snippet(source: str) -> str:
    """Remove blank and pure comment lines while preserving executable code."""
    compact_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        compact_lines.append(line.rstrip())
    return "\n".join(compact_lines)


def _assigned_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            names.extend(elt.id for elt in target.elts if isinstance(elt, ast.Name))
    return names


def _is_relevant_symbol(name: str) -> bool:
    lowered = name.lower()
    keywords = (
        "envcfg",
        "scene",
        "action",
        "teleop",
        "pipeline",
        "controller",
        "source",
        "retarget",
        "reorder",
        "combiner",
        "output",
        "device",
        "hand",
        "grip",
        "trigger",
        "motion",
        "openxr",
        "dex",
        "ik",
        "wrist",
        "pose",
        "eef",
    )
    return any(k in lowered for k in keywords)


def extract_env_cfg_adaptation_context(
    source: str,
    *,
    role: str,
    source_path: str,
    focus_class: str | None = None,
) -> dict[str, Any]:
    """Extract adaptation-relevant snippets from an env_cfg source file."""
    lines = source.splitlines()
    context: dict[str, Any] = {
        "role": role,
        "source_path": source_path,
        "source_chars": len(source),
        "source_lines": len(lines),
        "focus_class": focus_class,
        "imports": [],
        "symbols": [],
    }
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        context["parse_error"] = str(e)
        context["raw_source_fallback"] = source
        return context

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            snippet = _compact_code_snippet(_node_source(lines, node))
            if snippet:
                context["imports"].append(snippet)
            continue

        symbol_name = ""
        symbol_type = type(node).__name__
        include = False
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            symbol_name = node.name
            include = _is_relevant_symbol(symbol_name) or symbol_name == focus_class
            lowered_name = symbol_name.lower()
            if (
                role == "same_input_source_reference"
                and "motion" in str(focus_class or "").lower()
                and "pipeline" in lowered_name
                and "motion" not in lowered_name
                and "controller" not in lowered_name
            ):
                include = False
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = _assigned_names(node)
            symbol_name = ", ".join(names)
            include = any(_is_relevant_symbol(name) for name in names)

        if not include:
            continue
        snippet = _compact_code_snippet(_node_source(lines, node))
        if not snippet:
            continue
        context["symbols"].append(
            {
                "name": symbol_name,
                "type": symbol_type,
                "start_line": getattr(node, "lineno", None),
                "end_line": getattr(node, "end_lineno", None),
                "source": snippet,
            }
        )
    context["extracted_chars"] = sum(
        len(str(s.get("source", ""))) for s in context["symbols"] if isinstance(s, dict)
    ) + sum(len(str(i)) for i in context["imports"])
    return context


def render_template_env_cfg(spec: dict[str, Any]) -> tuple[str, str]:
    """Render a thin env_cfg subclass from an existing repository template."""
    robot_preset = str(spec.get("robot", {}).get("preset", ""))
    template = template_base_for_spec(spec)
    if template is None:
        raise RuntimeError(
            f"当前组合暂无确定性 env_cfg 模板：robot={robot_preset}, input_source={input_source_from_spec(spec)}。"
            "需要进入 adaptation candidate workflow，不能直接套用不匹配模板。"
        )

    task = spec.get("task", {})
    obj = spec.get("object", {})
    scene = spec.get("scene", {})
    target = spec.get("target", {})
    class_name = auto_env_class_name(spec)
    file_stem = str(task.get("generated_file_stem"))
    usd_path = str(obj.get("usd_path", ""))
    if is_missing_value(usd_path):
        raise RuntimeError("object.usd_path 仍为空或 TODO，不能模板化生成 env_cfg。")

    scale = obj.get("scale")
    single_rigid_body = bool(obj.get("single_rigid_body"))
    mass = obj.get("mass", 0.05)
    object_pose = obj.get("pose", {})
    object_pos = object_pose.get("pos") if isinstance(object_pose, dict) else None
    object_quat = object_pose.get("quat_xyzw") if isinstance(object_pose, dict) else None
    table_pose = scene.get("table", {}).get("pose", {}) if isinstance(scene.get("table"), dict) else {}
    target_pose = target.get("pose", {}) if isinstance(target, dict) else {}
    scene_objects = scene_objects_from_spec(spec)
    needs_single_rigid_body = single_rigid_body or any(bool(item.get("single_rigid_body")) for item in scene_objects)

    lines = [
        "# Copyright (c) 2022-2026, The Isaac Lab Project Developers.",
        "# SPDX-License-Identifier: BSD-3-Clause",
        "",
        '"""Auto-generated pick_place teleop env_cfg.',
        "",
        "This file is generated by my_docs/env_cfg_skills/orchestrator.py.",
        "It subclasses an existing repository env_cfg template and only overrides",
        "task-specific object/table/target parameters derived from the validated spec.",
        '"""',
        "",
        "from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR",
    ]
    if scene_objects:
        lines.extend(
            [
                "from isaaclab.assets import RigidObjectCfg",
                "from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg",
            ]
        )
    if needs_single_rigid_body:
        lines.extend(
            [
                "from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg",
                "from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg",
            ]
        )
    lines.extend(["", f"from .{template['module']} import {template['class']}"])
    if needs_single_rigid_body:
        lines.append("from .pickplace_seres_r11_env_cfg import SingleRigidBodyUsdFileCfg")
    lines.extend(
        [
            "",
            "",
            f"class {class_name}({template['class']}):",
            f'    """Auto task for {robot_preset} + {obj.get("preset", "object")}."""',
            "",
            "    def __post_init__(self):",
            "        super().__post_init__()",
            "",
            "        # Deterministic overrides from PickPlaceVRSpec presets.",
        ]
    )
    if single_rigid_body:
        lines.extend(
            [
                "        self.scene.object.spawn = SingleRigidBodyUsdFileCfg(",
                f"            usd_path={python_asset_expr(usd_path)},",
                f"            scale={tuple(scale)!r}," if scale else "            scale=None,",
                "            rigid_props=PhysxRigidBodyPropertiesCfg(),",
                f"            mass_props=MassPropertiesCfg(mass={float(mass)!r}),",
                "        )",
            ]
        )
    else:
        lines.append(f"        self.scene.object.spawn.usd_path = {python_asset_expr(usd_path)}")
        if scale:
            lines.append(f"        self.scene.object.spawn.scale = {tuple(scale)!r}")
    if object_pos:
        lines.append(f"        self.scene.object.init_state.pos = {tuple(object_pos)!r}")
    if object_quat:
        lines.append(f"        self.scene.object.init_state.rot = {tuple(object_quat)!r}")
    if isinstance(table_pose, dict) and table_pose.get("pos"):
        lines.append(f"        self.scene.packing_table.init_state.pos = {tuple(table_pose['pos'])!r}")
    if isinstance(table_pose, dict) and table_pose.get("quat_xyzw"):
        lines.append(f"        self.scene.packing_table.init_state.rot = {tuple(table_pose['quat_xyzw'])!r}")
    if isinstance(target_pose, dict) and target_pose.get("pos"):
        lines.append(f"        self.target_pose = {target_pose!r}")
    if scene_objects:
        lines.extend(["", "        # Additional scene objects are props/distractors; success still uses self.scene.object."])
        for scene_object in scene_objects:
            lines.extend(render_scene_object_cfg_lines(scene_object))
    lines.append("")
    return file_stem, "\n".join(lines)


def render_registered_init(
    spec: dict[str, Any], module_name: str, class_name: str, registration_module: str
) -> str:
    """Render pick_place __init__.py with an appended auto-generated gym registration."""
    init_abs_path = abs_from_repo_root(registration_module)
    content = read_text(init_abs_path)
    gym_id = str(spec.get("task", {}).get("gym_id"))
    if gym_id in content:
        return content
    block = f'''

# Auto-generated task registration. Keep the Auto prefix to distinguish it from hand-written tasks.
gym.register(
    id="{gym_id}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.{module_name}:{class_name}",
        "robomimic_bc_cfg_entry_point": f"{{agents.__name__}}:robomimic/bc_rnn_low_dim.json",
    }},
    disable_env_checker=True,
)
'''
    return content.rstrip() + block


def run_template_generator_workflow(spec: dict[str, Any]) -> dict[str, Any]:
    """Generate env_cfg and registration deterministically for repository presets."""
    file_stem, env_cfg_content = render_template_env_cfg(spec)
    class_name = auto_env_class_name(spec)
    paths = output_paths_for_spec(spec)
    output_dir = paths["output_dir"]
    env_cfg_path = f"{output_dir}/{file_stem}.py"
    init_path = paths["registration_module"]
    init_content = render_registered_init(
        spec, module_name=file_stem, class_name=class_name, registration_module=init_path
    )
    return {
        "files": [
            {"path": env_cfg_path, "content": env_cfg_content},
            {"path": init_path, "content": init_content},
        ],
        "summary": "template generator: 已根据仓库现有机器人模板确定性生成 env_cfg 薄封装和 task 注册，未调用大模型生成代码。",
        "commands": [teleop_command_for_spec(spec)],
        "generator_mode": "template",
    }


def run_adaptation_candidate_workflow(spec: dict[str, Any], enable_api: bool) -> dict[str, Any]:
    """Return a blocked adaptation candidate when exact template is missing."""
    robot_preset = str(spec.get("robot", {}).get("preset", ""))
    input_source = input_source_from_spec(spec)
    reference = find_reference_template_for_input_source(spec)
    reason = (
        f"没有 exact template: robot={robot_preset}, input_source={input_source}。"
        "当前不会把不匹配的参考模板直接写成正式 env_cfg。"
    )
    candidate = {
        "requested_robot": robot_preset,
        "requested_input_source": input_source,
        "reference_template": reference,
        "ai_usage": (
            "enable_api=true 时，后续可让 AI 只做 gap analysis / adaptation_spec；"
            "最终代码仍必须通过自动 tests gate 后才能写文件。"
            if enable_api
            else "未启用 API；只输出 adaptation candidate 信息。"
        ),
        "required_automatic_tests": [
            "generated Python AST parse",
            "no TODO placeholders",
            "task registration contains generated gym_id",
            "teleop command is single-line copyable",
            "optional IsaacLab smoke test: create env with --num_envs 1 and no XR session",
        ],
    }
    return {
        "ok": False,
        "files": [],
        "summary": reason,
        "commands": [],
        "generator_mode": "adaptation_candidate",
        "adaptation_candidate": candidate,
    }


def build_adaptation_messages(
    spec: dict[str, Any],
    requested_robot_reference_context: dict[str, Any],
    requested_robot_motion_controller_reference_context: dict[str, Any] | None,
    input_source_reference: dict[str, str] | None,
    input_source_reference_context: dict[str, Any],
    class_name: str,
) -> list[dict[str, str]]:
    """Build LLM messages for controlled template adaptation."""
    system = read_text(ADAPTATION_PROMPT_PATH)
    user_payload = {
        "required_class_name": class_name,
        "spec": spec,
        "target_robot_adaptation_preset": spec.get("robot", {}).get("motion_controller_adaptation", {}),
        "requested_robot_existing_reference_env_cfg_context": requested_robot_reference_context,
        "requested_robot_motion_controller_reference_context": requested_robot_motion_controller_reference_context,
        "same_input_source_reference": input_source_reference,
        "same_input_source_reference_env_cfg_context": input_source_reference_context,
        "context_policy": {
            "mode": "structured_snippets",
            "description": (
                "The workflow sends AST-extracted imports and relevant symbols instead of full files. "
                "This is not truncation; it preserves adaptation-relevant class/function/constant snippets."
            ),
        },
        "rules": [
            "优先复用 requested robot 的 robot/action/object/table 结构。",
            "只迁移 input_source / controller pipeline 相关逻辑。",
            "不要改写 task 注册，注册由程序生成。",
            "不要输出 TODO。",
            "不要假设不存在的 link/joint 名，必须使用 spec 中已有字段。",
            "输出必须是 JSON 对象，不能是 markdown。",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, indent=2, ensure_ascii=False)},
    ]


def _float_list(value: Any, length: int, default: list[float]) -> list[float]:
    if isinstance(value, list) and len(value) >= length:
        try:
            return [float(value[i]) for i in range(length)]
        except (TypeError, ValueError):
            return default
    return default


def validate_g1_motion_controller_adaptation_spec(
    spec: dict[str, Any], adaptation_spec: dict[str, Any]
) -> list[str]:
    """Validate AI-proposed G1 motion-controller adaptation parameters."""
    errors: list[str] = []
    if str(spec.get("robot", {}).get("preset", "")) != "unitree_g1_inspire_ftp":
        errors.append("G1 motion-controller deterministic adapter only supports unitree_g1_inspire_ftp")
    if input_source_from_spec(spec) != "openxr_controller":
        errors.append("G1 motion-controller deterministic adapter requires openxr_controller")
    hand_joint_names = spec.get("robot", {}).get("hand_joint_names", [])
    hand = adaptation_spec.get("hand", {}) if isinstance(adaptation_spec, dict) else {}
    open_pose = hand.get("open_pose", {}) if isinstance(hand, dict) else {}
    closed_pose = hand.get("closed_pose", {}) if isinstance(hand, dict) else {}
    proposed_names = hand.get("hand_joint_names", []) if isinstance(hand, dict) else []
    if not isinstance(hand_joint_names, list) or len(hand_joint_names) != 24:
        errors.append("spec.robot.hand_joint_names must contain 24 G1 Inspire hand joints")
        return errors
    if proposed_names != hand_joint_names:
        errors.append("adaptation_spec.hand.hand_joint_names must exactly match spec.robot.hand_joint_names")
    for pose_name, pose in (("open_pose", open_pose), ("closed_pose", closed_pose)):
        if not isinstance(pose, dict):
            errors.append(f"adaptation_spec.hand.{pose_name} must be an object")
            continue
        missing = [name for name in hand_joint_names if name not in pose]
        if missing:
            errors.append(f"adaptation_spec.hand.{pose_name} missing joints: {missing[:6]}")
        for name in hand_joint_names:
            if name in pose:
                try:
                    float(pose[name])
                except (TypeError, ValueError):
                    errors.append(f"adaptation_spec.hand.{pose_name}.{name} must be numeric")
                    break
    return errors


def render_g1_motion_controller_adapted_env_cfg(
    spec: dict[str, Any],
    adaptation_spec: dict[str, Any],
) -> tuple[str, str]:
    """Render a candidate G1 Inspire FTP motion-controller env_cfg from adaptation_spec."""
    class_name = auto_env_class_name(spec)
    file_stem = str(spec.get("task", {}).get("generated_file_stem"))
    obj = spec.get("object", {})
    scene = spec.get("scene", {})
    target = spec.get("target", {})
    robot = spec.get("robot", {})
    mc_preset = robot.get("motion_controller_adaptation", {}) if isinstance(robot, dict) else {}
    wrist = adaptation_spec.get("wrist", {}) if isinstance(adaptation_spec, dict) else {}
    hand = adaptation_spec.get("hand", {}) if isinstance(adaptation_spec, dict) else {}

    left_offset = _float_list(
        wrist.get("left_offset_rpy_deg"),
        3,
        _float_list(mc_preset.get("left_controller_offset_rpy_deg"), 3, [45.0, 180.0, -90.0]),
    )
    right_offset = _float_list(
        wrist.get("right_offset_rpy_deg"),
        3,
        _float_list(mc_preset.get("right_controller_offset_rpy_deg"), 3, [-135.0, 0.0, 90.0]),
    )
    use_wrist_rotation = bool(wrist.get("use_wrist_rotation", mc_preset.get("use_wrist_rotation", False)))
    use_wrist_position = bool(wrist.get("use_wrist_position", mc_preset.get("use_wrist_position", False)))
    hand_joint_names = list(robot.get("hand_joint_names", []))
    left_hand_joint_names = [name for name in hand_joint_names if str(name).startswith("L_")]
    right_hand_joint_names = [name for name in hand_joint_names if str(name).startswith("R_")]
    open_pose = {str(k): float(v) for k, v in hand.get("open_pose", {}).items()}
    closed_pose = {str(k): float(v) for k, v in hand.get("closed_pose", {}).items()}
    scale = obj.get("scale")
    usd_path = str(obj.get("usd_path", ""))
    single_rigid_body = bool(obj.get("single_rigid_body"))
    mass = obj.get("mass", 0.05)
    object_pose = obj.get("pose", {})
    table_pose = scene.get("table", {}).get("pose", {}) if isinstance(scene.get("table"), dict) else {}
    target_pose = target.get("pose", {}) if isinstance(target, dict) else {}
    scene_objects = scene_objects_from_spec(spec)
    needs_single_rigid_body = single_rigid_body or any(bool(item.get("single_rigid_body")) for item in scene_objects)

    lines = [
        "# Copyright (c) 2022-2026, The Isaac Lab Project Developers.",
        "# SPDX-License-Identifier: BSD-3-Clause",
        "",
        '"""Auto-generated G1 motion-controller pick_place teleop env_cfg.',
        "",
        "This file is generated from a validated PickPlaceVRSpec plus a model-proposed",
        "G1MotionControllerAdaptationSpec. The model proposes parameters only; Python",
        "code is rendered deterministically by my_docs/env_cfg_skills/generator_workflow.py.",
        '"""',
        "",
        "from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR",
    ]
    if scene_objects:
        lines.extend(
            [
                "from isaaclab.assets import RigidObjectCfg",
                "from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg",
            ]
        )
    if needs_single_rigid_body:
        lines.extend(
            [
                "from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg",
                "from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg",
            ]
        )
    lines.extend(
        [
            "",
            "from .pickplace_unitree_g1_inspire_hand_env_cfg import PickPlaceG1InspireFTPEnvCfg",
        ]
    )
    if needs_single_rigid_body:
        lines.append("from .pickplace_seres_r11_env_cfg import SingleRigidBodyUsdFileCfg")
    lines.extend(
        [
            "",
            "",
            f"G1_HAND_OPEN_POSE = {open_pose!r}",
            f"G1_HAND_CLOSED_POSE = {closed_pose!r}",
            f"G1_LEFT_HAND_RETARGET_JOINTS = {left_hand_joint_names!r}",
            f"G1_RIGHT_HAND_RETARGET_JOINTS = {right_hand_joint_names!r}",
            f"G1_PINK_HAND_JOINT_ORDER = {hand_joint_names!r}",
            "",
            "",
            "class G1ControllerTriggerGraspRetargeter:",
            '    """Map controller trigger/squeeze to G1 Inspire hand joint targets."""',
            "",
            "    def __init__(self, hand_joint_names, controller_side, name, open_pose, closed_pose):",
            "        from isaacteleop.retargeting_engine.interface import BaseRetargeter",
            "        from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType",
            "        from isaacteleop.retargeting_engine.tensor_types import ControllerInput, ControllerInputIndex, RobotHandJoints",
            "",
            "        open_lookup = {joint: float(open_pose.get(joint, 0.0)) for joint in hand_joint_names}",
            "        span_lookup = {joint: float(closed_pose.get(joint, 0.0)) - open_lookup[joint] for joint in hand_joint_names}",
            "",
            "        class _Retargeter(BaseRetargeter):",
            "            def __init__(self, hand_joint_names, controller_side, name):",
            "                self._hand_joint_names = list(hand_joint_names)",
            "                self._side = controller_side.lower()",
            "                super().__init__(name=name)",
            "",
            "            def input_spec(self):",
            "                return {f'controller_{self._side}': OptionalType(ControllerInput())}",
            "",
            "            def output_spec(self):",
            "                return {'hand_joints': RobotHandJoints(f'hand_joints_{self._side}', self._hand_joint_names)}",
            "",
            "            def _compute_fn(self, inputs, outputs, context) -> None:",
            "                output_group = outputs['hand_joints']",
            "                controller_group = inputs[f'controller_{self._side}']",
            "                if controller_group.is_none:",
            "                    grasp = 0.0",
            "                else:",
            "                    trigger = float(controller_group[ControllerInputIndex.TRIGGER_VALUE])",
            "                    squeeze = float(controller_group[ControllerInputIndex.SQUEEZE_VALUE])",
            "                    grasp = max(trigger, squeeze)",
            "                for i, joint_name in enumerate(self._hand_joint_names):",
            "                    output_group[i] = open_lookup.get(joint_name, 0.0) + grasp * span_lookup.get(joint_name, 0.0)",
            "",
            "        self._retargeter = _Retargeter(hand_joint_names, controller_side, name)",
            "",
            "    def __getattr__(self, name):",
            "        return getattr(self._retargeter, name)",
            "",
            "",
            "def _build_g1_inspire_motion_controller_pickplace_pipeline():",
            "    from isaacteleop.retargeters import Se3AbsRetargeter, Se3RetargeterConfig, TensorReorderer",
            "    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource",
            "    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput",
            "    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix",
            "",
            "    controllers = ControllersSource(name='controllers')",
            "    transform_input = ValueInput('world_T_anchor', TransformMatrix())",
            "    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))",
            "",
            "    left_se3 = Se3AbsRetargeter(",
            "        Se3RetargeterConfig(",
            "            input_device=ControllersSource.LEFT,",
            "            zero_out_xy_rotation=False,",
            f"            use_wrist_rotation={use_wrist_rotation!r},",
            f"            use_wrist_position={use_wrist_position!r},",
            f"            target_offset_roll={left_offset[0]!r},",
            f"            target_offset_pitch={left_offset[1]!r},",
            f"            target_offset_yaw={left_offset[2]!r},",
            "        ),",
            "        name='left_ee_pose',",
            "    )",
            "    connected_left_se3 = left_se3.connect({ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)})",
            "",
            "    right_se3 = Se3AbsRetargeter(",
            "        Se3RetargeterConfig(",
            "            input_device=ControllersSource.RIGHT,",
            "            zero_out_xy_rotation=False,",
            f"            use_wrist_rotation={use_wrist_rotation!r},",
            f"            use_wrist_position={use_wrist_position!r},",
            f"            target_offset_roll={right_offset[0]!r},",
            f"            target_offset_pitch={right_offset[1]!r},",
            f"            target_offset_yaw={right_offset[2]!r},",
            "        ),",
            "        name='right_ee_pose',",
            "    )",
            "    connected_right_se3 = right_se3.connect({ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)})",
            "",
            "    left_hand = G1ControllerTriggerGraspRetargeter(",
            "        G1_LEFT_HAND_RETARGET_JOINTS, controller_side='left', name='left_hand',",
            "        open_pose=G1_HAND_OPEN_POSE, closed_pose=G1_HAND_CLOSED_POSE,",
            "    )",
            "    connected_left_hand = left_hand.connect({ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)})",
            "    right_hand = G1ControllerTriggerGraspRetargeter(",
            "        G1_RIGHT_HAND_RETARGET_JOINTS, controller_side='right', name='right_hand',",
            "        open_pose=G1_HAND_OPEN_POSE, closed_pose=G1_HAND_CLOSED_POSE,",
            "    )",
            "    connected_right_hand = right_hand.connect({ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)})",
            "",
            "    left_ee_elements = ['l_pos_x', 'l_pos_y', 'l_pos_z', 'l_quat_x', 'l_quat_y', 'l_quat_z', 'l_quat_w']",
            "    right_ee_elements = ['r_pos_x', 'r_pos_y', 'r_pos_z', 'r_quat_x', 'r_quat_y', 'r_quat_z', 'r_quat_w']",
            "    output_order = left_ee_elements + right_ee_elements + G1_PINK_HAND_JOINT_ORDER",
            "    reorderer = TensorReorderer(",
            "        input_config={",
            "            'left_ee_pose': left_ee_elements,",
            "            'right_ee_pose': right_ee_elements,",
            "            'left_hand_joints': G1_LEFT_HAND_RETARGET_JOINTS,",
            "            'right_hand_joints': G1_RIGHT_HAND_RETARGET_JOINTS,",
            "        },",
            "        output_order=output_order,",
            "        name='action_reorderer',",
            "        input_types={",
            "            'left_ee_pose': 'array',",
            "            'right_ee_pose': 'array',",
            "            'left_hand_joints': 'scalar',",
            "            'right_hand_joints': 'scalar',",
            "        },",
            "    )",
            "    connected_reorderer = reorderer.connect(",
            "        {",
            "            'left_ee_pose': connected_left_se3.output('ee_pose'),",
            "            'right_ee_pose': connected_right_se3.output('ee_pose'),",
            "            'left_hand_joints': connected_left_hand.output('hand_joints'),",
            "            'right_hand_joints': connected_right_hand.output('hand_joints'),",
            "        }",
            "    )",
            "    pipeline = OutputCombiner({'action': connected_reorderer.output('output')})",
            "    return pipeline, [left_se3, right_se3]",
            "",
            "",
            f"class {class_name}(PickPlaceG1InspireFTPEnvCfg):",
            '    """Auto task for Unitree G1 Inspire FTP + motion controller."""',
            "",
            "    def __post_init__(self):",
            "        super().__post_init__()",
            "",
            "        # Deterministic object/table/target overrides from PickPlaceVRSpec.",
        ]
    )
    if single_rigid_body:
        lines.extend(
            [
                "        self.scene.object.spawn = SingleRigidBodyUsdFileCfg(",
                f"            usd_path={python_asset_expr(usd_path)},",
                f"            scale={tuple(scale)!r}," if scale else "            scale=None,",
                "            rigid_props=PhysxRigidBodyPropertiesCfg(),",
                f"            mass_props=MassPropertiesCfg(mass={float(mass)!r}),",
                "        )",
            ]
        )
    else:
        lines.append(f"        self.scene.object.spawn.usd_path = {python_asset_expr(usd_path)}")
        if scale:
            lines.append(f"        self.scene.object.spawn.scale = {tuple(scale)!r}")
    if isinstance(object_pose, dict) and object_pose.get("pos"):
        lines.append(f"        self.scene.object.init_state.pos = {tuple(object_pose['pos'])!r}")
    if isinstance(object_pose, dict) and object_pose.get("quat_xyzw"):
        lines.append(f"        self.scene.object.init_state.rot = {tuple(object_pose['quat_xyzw'])!r}")
    if isinstance(table_pose, dict) and table_pose.get("pos"):
        lines.append(f"        self.scene.packing_table.init_state.pos = {tuple(table_pose['pos'])!r}")
    if isinstance(table_pose, dict) and table_pose.get("quat_xyzw"):
        lines.append(f"        self.scene.packing_table.init_state.rot = {tuple(table_pose['quat_xyzw'])!r}")
    if isinstance(target_pose, dict) and target_pose.get("pos"):
        lines.append(f"        self.target_pose = {target_pose!r}")
    if scene_objects:
        lines.extend(["", "        # Additional scene objects are props/distractors; success still uses self.scene.object."])
        for scene_object in scene_objects:
            lines.extend(render_scene_object_cfg_lines(scene_object))
    lines.extend(
        [
            "",
            "        # Replace hand-tracking teleop pipeline with controller trigger-open/close pipeline.",
            "        self.isaac_teleop.pipeline_builder = lambda: _build_g1_inspire_motion_controller_pickplace_pipeline()[0]",
            "",
        ]
    )
    return file_stem, "\n".join(lines)


def run_adaptation_generator_workflow(
    spec: dict[str, Any],
    cfg: OrchestratorConfig,
    api_key: str | None,
) -> dict[str, Any]:
    """Use AI to propose adaptation_spec, then render candidate env_cfg deterministically."""
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {cfg.llm.api_key_env}")

    reference = find_reference_template_for_input_source(spec)
    if reference is None:
        return run_adaptation_candidate_workflow(spec, enable_api=True)

    robot = robot_preset_for_spec(spec)
    requested_robot_reference_path = str(robot.get("reference_env_cfg", ""))
    requested_robot_reference = read_repo_file_if_exists(requested_robot_reference_path)
    requested_robot_template = adaptation_focus_template_for_robot(robot)
    requested_robot_reference_context = extract_env_cfg_adaptation_context(
        requested_robot_reference,
        role="requested_robot_existing_reference",
        source_path=requested_robot_reference_path,
        focus_class=str(requested_robot_template.get("class", "")) if requested_robot_template else None,
    )
    mc_reference_context = None
    mc_reference_path = str(robot.get("motion_controller_adaptation", {}).get("reference_env_cfg", ""))
    if mc_reference_path:
        mc_reference_content = read_repo_file_if_exists(mc_reference_path)
        mc_reference_context = extract_env_cfg_adaptation_context(
            mc_reference_content,
            role="requested_robot_motion_controller_reference",
            source_path=mc_reference_path,
            focus_class=None,
        )
    input_source_reference_path = reference.get("reference_env_cfg", "")
    input_source_reference_content = read_repo_file_if_exists(input_source_reference_path)
    input_source_reference_context = extract_env_cfg_adaptation_context(
        input_source_reference_content,
        role="same_input_source_reference",
        source_path=input_source_reference_path,
        focus_class=reference.get("class"),
    )
    class_name = auto_env_class_name(spec)
    file_stem = str(spec.get("task", {}).get("generated_file_stem"))
    paths = output_paths_for_spec(spec)
    output_dir = paths["output_dir"]
    env_cfg_path = f"{output_dir}/{file_stem}.py"

    sys.stderr.write("[Stage B] Adaptation Generator: exact template 缺失，正在调用模型生成 adaptation_spec...\n")
    sys.stderr.flush()
    messages = build_adaptation_messages(
        spec,
        requested_robot_reference_context=requested_robot_reference_context,
        requested_robot_motion_controller_reference_context=mc_reference_context,
        input_source_reference=reference,
        input_source_reference_context=input_source_reference_context,
        class_name=class_name,
    )
    generated_text = call_llm_chat_completion(cfg.llm, api_key=api_key, messages=messages)
    sys.stderr.write("[Stage B] Adaptation Generator: 已返回，正在解析 adaptation_spec...\n")
    sys.stderr.flush()
    obj = extract_structured_object(generated_text)
    if obj.get("blocked_reason"):
        return {
            "ok": False,
            "files": [],
            "summary": str(obj["blocked_reason"]),
            "commands": [],
            "generator_mode": "adaptation_candidate_blocked_by_ai",
            "adaptation_candidate": {
                "requested_robot": str(spec.get("robot", {}).get("preset", "")),
                "requested_input_source": input_source_from_spec(spec),
                "reference_template": reference,
                "blocked_reason": str(obj["blocked_reason"]),
            },
        }
    adaptation_spec = obj.get("adaptation_spec")
    if not isinstance(adaptation_spec, dict):
        return {
            "ok": False,
            "files": [],
            "summary": "AI adaptation response missing adaptation_spec.",
            "commands": [],
            "generator_mode": "adaptation_candidate_invalid_ai_output",
            "adaptation_candidate": {"reference_template": reference},
        }
    adaptation_errors = validate_g1_motion_controller_adaptation_spec(spec, adaptation_spec)
    if adaptation_errors:
        return {
            "ok": False,
            "files": [],
            "summary": "AI adaptation_spec did not pass deterministic validation.",
            "commands": [],
            "generator_mode": "adaptation_spec_validation_failed",
            "adaptation_candidate": {
                "requested_robot": str(spec.get("robot", {}).get("preset", "")),
                "requested_input_source": input_source_from_spec(spec),
                "reference_template": reference,
                "errors": adaptation_errors,
                "adaptation_spec": adaptation_spec,
            },
        }

    file_stem, env_cfg_content = render_g1_motion_controller_adapted_env_cfg(spec, adaptation_spec)
    env_cfg_path = f"{output_dir}/{file_stem}.py"

    init_path = paths["registration_module"]
    init_content = render_registered_init(
        spec, module_name=file_stem, class_name=class_name, registration_module=init_path
    )
    return {
        "files": [
            {"path": env_cfg_path, "content": env_cfg_content},
            {"path": init_path, "content": init_content},
        ],
        "summary": str(obj.get("summary", "AI adaptation_spec accepted; env_cfg rendered deterministically.")),
        "commands": [teleop_command_for_spec(spec)],
        "generator_mode": "adaptation_spec_deterministic_candidate",
        "adaptation_candidate": {
            "requested_robot": str(spec.get("robot", {}).get("preset", "")),
            "requested_input_source": input_source_from_spec(spec),
            "reference_template": reference,
            "adaptation_spec": adaptation_spec,
        },
    }


def build_generator_messages(spec: dict[str, Any], generator_prompt: str | None) -> list[dict[str, str]]:
    """Build chat messages for the fallback LLM Generator workflow."""
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


def run_generator_workflow(
    skill: SkillDef,
    spec: dict[str, Any],
    cfg: OrchestratorConfig,
    api_key: str | None,
    enable_api: bool,
) -> dict[str, Any]:
    """Run Stage B: validated spec -> env_cfg files and task registration output."""
    if skill.name == "pick_place_vr":
        if template_base_for_spec(spec) is not None:
            return run_template_generator_workflow(spec)
        if enable_api:
            return run_adaptation_generator_workflow(spec, cfg=cfg, api_key=api_key)
        return run_adaptation_candidate_workflow(spec, enable_api=enable_api)

    generator_prompt = read_text(skill.generator_prompt_path) if skill.generator_prompt_path else None
    if not enable_api:
        return {
            "files": [],
            "summary": "dry-run placeholder: spec 已通过 schema gate，但未调用 LLM API 生成 env_cfg。",
            "commands": [teleop_command_for_spec(spec)],
        }
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {cfg.llm.api_key_env}")

    sys.stderr.write("[Stage B] Generator: 正在调用模型生成 env_cfg / 注册改动 / 启动命令...\n")
    sys.stderr.flush()
    messages = build_generator_messages(spec, generator_prompt)
    generated_text = call_llm_chat_completion(cfg.llm, api_key=api_key, messages=messages)
    sys.stderr.write("[Stage B] Generator: 已返回，正在解析输出...\n")
    sys.stderr.flush()
    generated = extract_structured_object(generated_text)
    generated.setdefault("files", [])
    generated.setdefault("summary", "LLM returned generated payload without summary.")
    generated.setdefault("commands", [])
    if not generated["commands"]:
        generated["commands"] = [teleop_command_for_spec(spec)]
    return generated


def write_generated_files(generated: dict[str, Any], dry_run: bool) -> list[str]:
    """Write generated files under repo root when dry_run is False."""
    written: list[str] = []
    files = generated.get("files", [])
    if not isinstance(files, list):
        raise ValueError("generated.files must be a list")

    repo_root = repo_root_from_this_file()
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
