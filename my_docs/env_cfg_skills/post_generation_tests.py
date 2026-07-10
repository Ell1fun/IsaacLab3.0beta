from __future__ import annotations

import ast
import math
import os
from typing import Any

from env_utils import abs_from_repo_root
from presets import load_skill_presets


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


def _float_list(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) < length:
        return None
    out: list[float] = []
    for i in range(length):
        try:
            out.append(float(value[i]))
        except (TypeError, ValueError):
            return None
    return out


def _robot_preset_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    robot_preset_key = str(spec.get("robot", {}).get("preset", "") or "")
    presets = load_skill_presets()
    robots = presets.get("robots", {})
    if not isinstance(robots, dict):
        return {}
    preset = robots.get(robot_preset_key, {})
    return preset if isinstance(preset, dict) else {}


def _object_radius_m(obj: dict[str, Any], default_radius_m: float) -> float:
    radius_m = obj.get("radius_m")
    if isinstance(radius_m, (int, float)) and float(radius_m) > 0.0:
        return float(radius_m)
    scale = _float_list(obj.get("scale"), 3)
    if scale:
        return default_radius_m * max(scale)
    return default_radius_m


def _collect_spawned_objects(spec: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    primary = spec.get("object", {})
    if isinstance(primary, dict):
        out.append({"kind": "primary", **primary})
    scene_objects = spec.get("scene_objects", [])
    if isinstance(scene_objects, list):
        for item in scene_objects:
            if isinstance(item, dict):
                out.append({"kind": "scene_object", **item})
    return out


def _semantic_hint(obj: dict[str, Any]) -> str:
    placement = obj.get("placement", {})
    if isinstance(placement, dict) and isinstance(placement.get("semantic"), str):
        return str(placement["semantic"])
    return ""


def _idle_wrist_pose(robot: dict[str, Any], side: str) -> list[float] | None:
    idle = robot.get("idle_wrist_pose", {})
    if not isinstance(idle, dict):
        return None
    pose = idle.get(side, {})
    if not isinstance(pose, dict):
        return None
    pos = _float_list(pose.get("pos"), 3)
    if not pos:
        return None
    if all(abs(v) < 1e-9 for v in pos):
        return None
    return pos


def _table_center_from_spec(spec: dict[str, Any]) -> list[float] | None:
    scene = spec.get("scene", {})
    if not isinstance(scene, dict):
        return None
    table = scene.get("table", {})
    if not isinstance(table, dict):
        return None
    pose = table.get("pose", {})
    if not isinstance(pose, dict):
        return None
    return _float_list(pose.get("pos"), 3)


def _tabletop_xy_half_extents(robot: dict[str, Any]) -> tuple[float, float]:
    default_half = (0.35, 0.25)
    table = robot.get("default_table", {})
    table_dict = table if isinstance(table, dict) else {}
    extents = _float_list(table_dict.get("tabletop_xy_half_extents"), 2)
    if extents and extents[0] > 0.0 and extents[1] > 0.0:
        return float(extents[0]), float(extents[1])
    extents2 = _float_list(robot.get("tabletop_xy_half_extents"), 2)
    if extents2 and extents2[0] > 0.0 and extents2[1] > 0.0:
        return float(extents2[0]), float(extents2[1])
    return default_half


def _tabletop_margin_m(robot: dict[str, Any]) -> float:
    table = robot.get("default_table", {})
    table_dict = table if isinstance(table, dict) else {}
    margin = table_dict.get("tabletop_margin_m", robot.get("tabletop_margin_m", 0.05))
    try:
        margin_f = float(margin)
    except (TypeError, ValueError):
        return 0.05
    return max(0.0, margin_f)


def _min_object_spacing_m(robot: dict[str, Any]) -> float:
    value = robot.get("min_object_spacing_m", 0.12)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.12


def _max_idle_wrist_to_object_distance_m(robot: dict[str, Any]) -> float:
    reach = robot.get("reachability", {}) if isinstance(robot.get("reachability"), dict) else {}
    value = reach.get("max_idle_wrist_to_object_distance_m", 0.85)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.85


def _pose_xy_in_tabletop_bounds(
    *,
    table_center: list[float],
    pos: list[float],
    half_extents: tuple[float, float],
    margin_m: float,
) -> bool:
    dx = float(pos[0]) - float(table_center[0])
    dy = float(pos[1]) - float(table_center[1])
    limit_x = max(0.0, float(half_extents[0]) - float(margin_m))
    limit_y = max(0.0, float(half_extents[1]) - float(margin_m))
    return abs(dx) <= limit_x and abs(dy) <= limit_y


def _xy_distance(a: list[float], b: list[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _xyz_distance(a: list[float], b: list[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _resolve_repo_path(path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(expanded):
        return expanded
    return abs_from_repo_root(expanded)


def _reachability_cfg(robot: dict[str, Any]) -> dict[str, Any]:
    reachability = robot.get("reachability", {})
    return reachability if isinstance(reachability, dict) else {}


def _strict_ik_cfg(robot: dict[str, Any]) -> dict[str, Any]:
    strict_ik = _reachability_cfg(robot).get("strict_ik", {})
    return strict_ik if isinstance(strict_ik, dict) else {}


def _strict_ik_enabled(robot: dict[str, Any]) -> bool:
    return bool(_strict_ik_cfg(robot).get("enabled", False))


def _strict_ik_required(robot: dict[str, Any]) -> bool:
    return bool(_strict_ik_cfg(robot).get("required", False))


def _side_for_object(obj: dict[str, Any], pos: list[float], robot: dict[str, Any]) -> str:
    semantic = _semantic_hint(obj)
    if semantic == "near_left_hand":
        return "left"
    if semantic == "near_right_hand":
        return "right"
    left_idle = _idle_wrist_pose(robot, "left")
    right_idle = _idle_wrist_pose(robot, "right")
    if left_idle and right_idle:
        return "left" if _xyz_distance(pos, left_idle) <= _xyz_distance(pos, right_idle) else "right"
    if left_idle:
        return "left"
    return "right"


def _controlled_velocity_indices(model: Any, controlled_joint_names: list[str]) -> list[int]:
    indices: list[int] = []
    for joint_name in controlled_joint_names:
        if not model.existJointName(joint_name):
            continue
        joint_id = model.getJointId(joint_name)
        joint = model.joints[joint_id]
        start = int(joint.idx_v)
        for offset in range(int(joint.nv)):
            indices.append(start + offset)
    return sorted(set(indices))


def _apply_initial_joint_positions(model: Any, q: Any, joint_pos: dict[str, Any]) -> Any:
    for joint_name, value in joint_pos.items():
        if not model.existJointName(str(joint_name)):
            continue
        joint_id = model.getJointId(str(joint_name))
        joint = model.joints[joint_id]
        if int(joint.nq) != 1:
            continue
        try:
            q[int(joint.idx_q)] = float(value)
        except (TypeError, ValueError):
            continue
    return q


def _strict_position_ik_reachable(
    *,
    robot: dict[str, Any],
    side: str,
    target_pos: list[float],
) -> tuple[bool, str]:
    """Run a lightweight Pinocchio position IK check without launching simulation."""
    urdf_path = str(robot.get("urdf_path", "") or "")
    if not urdf_path:
        return False, "robot preset has no urdf_path"
    urdf_abs = _resolve_repo_path(urdf_path)
    if not os.path.exists(urdf_abs):
        return False, f"urdf_path does not exist: {urdf_path}"
    eef_link_names = robot.get("eef_link_names", {})
    if not isinstance(eef_link_names, dict):
        return False, "robot preset has no eef_link_names"
    eef_frame = str(eef_link_names.get(side, "") or "")
    if not eef_frame:
        return False, f"robot preset has no {side} eef_link_names entry"
    controlled_joint_names = [str(name) for name in robot.get("controlled_joint_names", []) if name]
    if not controlled_joint_names:
        return False, "robot preset has no controlled_joint_names"

    try:
        import numpy as np  # type: ignore
        import pinocchio as pin  # type: ignore
        from pinocchio.robot_wrapper import RobotWrapper  # type: ignore
    except Exception as exc:
        return False, f"Pinocchio IK dependencies are unavailable: {exc}"

    try:
        wrapper = RobotWrapper.BuildFromURDF(urdf_abs)
        model = wrapper.model
        data = model.createData()
        if not model.existFrame(eef_frame):
            return False, f"EEF frame not found in URDF: {eef_frame}"
        controlled_v_indices = _controlled_velocity_indices(model, controlled_joint_names)
        if not controlled_v_indices:
            return False, "none of controlled_joint_names exist in the URDF model"

        q = wrapper.q0.copy()
        init_state = robot.get("init_state", {}) if isinstance(robot.get("init_state"), dict) else {}
        joint_pos = init_state.get("joint_pos", {}) if isinstance(init_state.get("joint_pos"), dict) else {}
        q = _apply_initial_joint_positions(model, q, joint_pos)

        cfg = _strict_ik_cfg(robot)
        max_iterations = int(cfg.get("max_iterations", 80))
        tolerance_m = float(cfg.get("position_tolerance_m", 0.04))
        damping = float(cfg.get("damping", 1e-4))
        step_size = float(cfg.get("step_size", 0.6))
        target = np.array(target_pos, dtype=float)
        frame_id = model.getFrameId(eef_frame)
        best_error = float("inf")

        for _ in range(max_iterations):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            current = data.oMf[frame_id].translation
            error = target - current
            error_norm = float(np.linalg.norm(error))
            best_error = min(best_error, error_norm)
            if error_norm <= tolerance_m:
                return True, f"IK converged with position error {round(error_norm, 5)} m"
            jacobian = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            jacobian_controlled = jacobian[:, controlled_v_indices]
            lhs = jacobian_controlled @ jacobian_controlled.T + damping * np.eye(3)
            delta_controlled = jacobian_controlled.T @ np.linalg.solve(lhs, error)
            velocity = np.zeros(model.nv)
            velocity[controlled_v_indices] = delta_controlled
            q = pin.integrate(model, q, velocity * step_size)
        return False, f"IK did not converge; best position error {round(best_error, 5)} m"
    except Exception as exc:
        return False, f"IK check failed: {exc}"


def _validate_strict_ik_reachability(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    robot = _robot_preset_from_spec(spec)
    if not _strict_ik_enabled(robot):
        return errors
    cfg = _strict_ik_cfg(robot)
    z_offset = float(cfg.get("target_z_offset_m", 0.05))
    required = _strict_ik_required(robot)
    for obj in _collect_spawned_objects(spec):
        pose = obj.get("pose", {})
        if not isinstance(pose, dict):
            continue
        pos = _float_list(pose.get("pos"), 3)
        if not pos:
            continue
        name = str(obj.get("name") or obj.get("preset") or obj.get("kind") or "object")
        side = _side_for_object(obj, pos, robot)
        target = [pos[0], pos[1], pos[2] + z_offset]
        reachable, reason = _strict_position_ik_reachable(robot=robot, side=side, target_pos=target)
        if reachable:
            continue
        unavailable = (
            "no urdf_path" in reason
            or "does not exist" in reason
            or "dependencies are unavailable" in reason
            or "no controlled_joint_names" in reason
            or "none of controlled_joint_names" in reason
        )
        if unavailable and not required:
            continue
        errors.append(f"strict IK reachability failed for '{name}' using {side} EEF: {reason}")
    return errors


def _validate_tabletop_bounds_and_spacing(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    robot = _robot_preset_from_spec(spec)
    table_center = _table_center_from_spec(spec)
    if not table_center:
        return ["spec.scene.table.pose.pos is missing; cannot validate tabletop bounds"]
    half_extents = _tabletop_xy_half_extents(robot)
    margin_m = _tabletop_margin_m(robot)
    min_spacing_m = _min_object_spacing_m(robot)
    objects = _collect_spawned_objects(spec)
    positions: list[tuple[str, list[float], float]] = []
    for obj in objects:
        name = str(obj.get("name") or obj.get("preset") or obj.get("kind") or "object")
        pose = obj.get("pose", {})
        if not isinstance(pose, dict):
            continue
        pos = _float_list(pose.get("pos"), 3)
        if not pos:
            continue
        radius = _object_radius_m(obj, default_radius_m=0.06)
        positions.append((name, pos, radius))
        if not _pose_xy_in_tabletop_bounds(
            table_center=table_center, pos=pos, half_extents=half_extents, margin_m=margin_m
        ):
            errors.append(
                f"object '{name}' is out of tabletop XY bounds: pos={pos}, table_center={table_center}, "
                f"half_extents={list(half_extents)}, margin_m={margin_m}"
            )
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            name_a, pos_a, rad_a = positions[i]
            name_b, pos_b, rad_b = positions[j]
            dist = _xy_distance(pos_a, pos_b)
            required = max(min_spacing_m, rad_a + rad_b)
            if dist < required:
                errors.append(
                    f"objects '{name_a}' and '{name_b}' are too close: dist_xy={round(dist, 4)}, "
                    f"required>={round(required, 4)}"
                )
    return errors


def _validate_reachability_heuristic(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    robot = _robot_preset_from_spec(spec)
    left_idle = _idle_wrist_pose(robot, "left")
    right_idle = _idle_wrist_pose(robot, "right")
    if not left_idle and not right_idle:
        return errors
    threshold_m = _max_idle_wrist_to_object_distance_m(robot)
    objects = _collect_spawned_objects(spec)
    for obj in objects:
        pose = obj.get("pose", {})
        if not isinstance(pose, dict):
            continue
        pos = _float_list(pose.get("pos"), 3)
        if not pos:
            continue
        name = str(obj.get("name") or obj.get("preset") or obj.get("kind") or "object")
        semantic = _semantic_hint(obj)
        if semantic == "near_left_hand" and left_idle:
            dist = _xyz_distance(pos, left_idle)
            if dist > threshold_m:
                errors.append(
                    f"reachability check failed for '{name}' (near_left_hand): dist={round(dist, 4)} > {threshold_m}"
                )
            continue
        if semantic == "near_right_hand" and right_idle:
            dist = _xyz_distance(pos, right_idle)
            if dist > threshold_m:
                errors.append(
                    f"reachability check failed for '{name}' (near_right_hand): dist={round(dist, 4)} > {threshold_m}"
                )
            continue
        candidates: list[float] = []
        if left_idle:
            candidates.append(_xyz_distance(pos, left_idle))
        if right_idle:
            candidates.append(_xyz_distance(pos, right_idle))
        best = min(candidates) if candidates else float("inf")
        if best > threshold_m:
            errors.append(f"reachability check failed for '{name}': dist={round(best, 4)} > {threshold_m}")
    return errors


def run_post_generation_tests(generated: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Run fast deterministic checks before generated files are written."""
    errors: list[str] = []
    files = generated.get("files", [])
    if not isinstance(files, list):
        return ["generated.files must be a list"]

    gym_id = str(spec.get("task", {}).get("gym_id", ""))
    scene_objects_raw = spec.get("scene_objects", [])
    scene_objects = scene_objects_raw if isinstance(scene_objects_raw, list) else []
    scene_object_names = [str(item.get("name")) for item in scene_objects if isinstance(item, dict)]
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
            for name in scene_object_names:
                if f"self.scene.{name} = RigidObjectCfg(" not in content:
                    errors.append(f"{path}: missing generated scene object: {name}")

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

    errors.extend(_validate_tabletop_bounds_and_spacing(spec))
    errors.extend(_validate_reachability_heuristic(spec))
    errors.extend(_validate_strict_ik_reachability(spec))
    return errors
