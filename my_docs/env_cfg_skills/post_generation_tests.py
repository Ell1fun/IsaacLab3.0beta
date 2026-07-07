from __future__ import annotations

import ast
from typing import Any


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return None
    return None


def run_post_generation_tests(generated: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Run fast deterministic checks before generated files are written."""
    errors: list[str] = []
    files = generated.get("files", [])
    if not isinstance(files, list):
        return ["generated.files must be a list"]

    gym_id = str(spec.get("task", {}).get("gym_id", ""))
    has_env_cfg = False
    has_registration = False

    for item in files:
        if not isinstance(item, dict):
            errors.append("Each generated file entry must be an object")
            continue
        path = item.get("path")
        content = item.get("content")
        if not isinstance(path, str) or not path:
            errors.append("Each generated file needs a non-empty string path")
            continue
        if not isinstance(content, str):
            errors.append(f"{path}: content must be a string")
            continue
        if "TODO" in content or "todo" in content:
            errors.append(f"{path}: contains TODO placeholder")

        if path.endswith(".py"):
            tree = None
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError as exc:
                errors.append(f"{path}: Python syntax error: {exc}")
            if tree is not None and "G1ControllerTriggerGraspRetargeter" in content:
                hand_order = _literal_assignment(tree, "G1_PINK_HAND_JOINT_ORDER")
                left_joints = _literal_assignment(tree, "G1_LEFT_HAND_RETARGET_JOINTS")
                right_joints = _literal_assignment(tree, "G1_RIGHT_HAND_RETARGET_JOINTS")
                open_pose = _literal_assignment(tree, "G1_HAND_OPEN_POSE")
                closed_pose = _literal_assignment(tree, "G1_HAND_CLOSED_POSE")
                expected = spec.get("robot", {}).get("hand_joint_names", [])
                if hand_order != expected:
                    errors.append(f"{path}: G1_PINK_HAND_JOINT_ORDER must match spec.robot.hand_joint_names")
                if not isinstance(left_joints, list) or len(left_joints) != 12:
                    errors.append(f"{path}: G1_LEFT_HAND_RETARGET_JOINTS must contain 12 joints")
                if not isinstance(right_joints, list) or len(right_joints) != 12:
                    errors.append(f"{path}: G1_RIGHT_HAND_RETARGET_JOINTS must contain 12 joints")
                for pose_name, pose in (("G1_HAND_OPEN_POSE", open_pose), ("G1_HAND_CLOSED_POSE", closed_pose)):
                    if not isinstance(pose, dict):
                        errors.append(f"{path}: {pose_name} must be a dict")
                        continue
                    missing = [joint for joint in expected if joint not in pose]
                    if missing:
                        errors.append(f"{path}: {pose_name} missing joints: {missing[:6]}")

        if path.endswith("_env_cfg.py"):
            has_env_cfg = True
            if "class AutoPickPlace" not in content:
                errors.append(f"{path}: expected AutoPickPlace env cfg class")
            if "def __post_init__(self):" not in content:
                errors.append(f"{path}: missing __post_init__ override")

        if path.endswith("__init__.py"):
            has_registration = True
            if gym_id and gym_id not in content:
                errors.append(f"{path}: missing generated gym_id registration")

    commands = generated.get("commands", [])
    if not isinstance(commands, list) or not commands:
        errors.append("generated.commands must contain at least one command")
    else:
        for command in commands:
            if not isinstance(command, str) or not command.strip():
                errors.append("generated.commands contains a non-string or empty command")
                continue
            if "\n" in command:
                errors.append("teleop command must be a single-line copyable command")
            if "--task" not in command:
                errors.append("teleop command must contain --task")

    if not has_env_cfg:
        errors.append("missing generated auto env_cfg file")
    if not has_registration:
        errors.append("missing generated task registration file")
    return errors
