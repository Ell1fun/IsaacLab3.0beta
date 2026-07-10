from __future__ import annotations

import copy
import math
import os
from typing import Any

from env_utils import deep_fill, is_missing_value, load_yaml_or_json, pascal_from_slug, presets_dir


def load_skill_presets() -> dict[str, dict[str, Any]]:
    """Load robot/object preset facts used for deterministic spec enrichment."""
    base = presets_dir()
    robot_path = os.path.join(base, "robot_presets.yml")
    object_path = os.path.join(base, "object_presets.yml")
    placement_path = os.path.join(base, "placement_presets.yml")
    robots = load_yaml_or_json(robot_path) if os.path.exists(robot_path) else {}
    objects = load_yaml_or_json(object_path) if os.path.exists(object_path) else {}
    placements = load_yaml_or_json(placement_path) if os.path.exists(placement_path) else {}
    if not isinstance(robots, dict):
        raise ValueError("robot_presets.yml must contain a YAML object")
    if not isinstance(objects, dict):
        raise ValueError("object_presets.yml must contain a YAML object")
    if not isinstance(placements, dict):
        raise ValueError("placement_presets.yml must contain a YAML object")
    return {
        "robots": resolve_preset_inheritance(robots),
        "objects": objects,
        "placements": resolve_preset_inheritance(placements),
    }


def resolve_preset_inheritance(presets: dict[str, Any]) -> dict[str, Any]:
    """Resolve simple preset inheritance declared with `inherits`."""
    resolved: dict[str, Any] = {}

    def resolve(name: str) -> dict[str, Any]:
        if name in resolved:
            return resolved[name]
        raw = presets.get(name, {})
        if not isinstance(raw, dict):
            raise ValueError(f"Preset {name} must be an object")
        parent_name = raw.get("inherits")
        if parent_name:
            parent = copy.deepcopy(resolve(str(parent_name)))
            child = {k: v for k, v in raw.items() if k != "inherits"}
            deep_fill(parent, child, overwrite=True)
            resolved[name] = parent
        else:
            resolved[name] = copy.deepcopy(raw)
        return resolved[name]

    for preset_name in presets:
        resolve(preset_name)
    return resolved


def task_names_from_presets(
    robot_preset: dict[str, Any], object_preset: dict[str, Any], input_source: str = "openxr_hand_tracking"
) -> dict[str, str]:
    """Derive auto file stem and Gym id from robot/object preset metadata."""
    robot_slug = str(robot_preset.get("robot_slug", "robot"))
    object_slug = str(object_preset.get("object_slug", "object"))
    robot_display = str(robot_preset.get("display_name", pascal_from_slug(robot_slug)))
    object_display = str(object_preset.get("display_name", pascal_from_slug(object_slug)))
    mode_slug = {
        "openxr_hand_tracking": "hand_tracking",
        "openxr_controller": "motion_controller",
        "keyboard_debug": "keyboard_debug",
    }.get(input_source, input_source.replace("-", "_"))
    mode_display = pascal_from_slug(mode_slug)
    return {
        "variant": f"{robot_slug}_{object_slug}_{mode_slug}_teleop",
        "generated_file_stem": f"auto_pickplace_{robot_slug}_{object_slug}_{mode_slug}_teleop_env_cfg",
        "gym_id": f"Isaac-Auto-PickPlace-{robot_display}-{object_display}-{mode_display}-Teleop-v0",
    }


def compute_default_pick_place_placement(
    spec: dict[str, Any],
    robot_preset: dict[str, Any],
    object_preset: dict[str, Any],
    *,
    object_pose_was_user_provided: bool,
    target_pose_was_user_provided: bool,
) -> list[str]:
    """Compute object/target poses from robot workspace facts when users omit poses."""
    notes: list[str] = []
    placement = robot_preset.get("default_placement")
    table_pose = spec.get("scene", {}).get("table", {}).get("pose", {})
    table_pos = table_pose.get("pos") if isinstance(table_pose, dict) else None
    if not isinstance(placement, dict) or not isinstance(table_pos, list) or len(table_pos) < 2:
        return notes

    z = float(placement.get("z", object_preset.get("tabletop_z", 0.9996)))
    object_offset = placement.get("object_xy_from_table_center", [0.0, -0.10])
    target_offset = placement.get("target_xy_from_table_center", [0.25, -0.10])
    if not (isinstance(object_offset, list) and len(object_offset) >= 2):
        object_offset = [0.0, -0.10]
    if not (isinstance(target_offset, list) and len(target_offset) >= 2):
        target_offset = [0.25, -0.10]

    def add_offset(base: float, offset: float) -> float:
        return round(float(base) + float(offset), 6)

    obj = spec.setdefault("object", {})
    target = spec.setdefault("target", {})
    if not object_pose_was_user_provided:
        obj["pose"] = {
            "pos": [add_offset(table_pos[0], object_offset[0]), add_offset(table_pos[1], object_offset[1]), z],
            "quat_xyzw": object_preset.get("default_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
        }
        notes.append("placement computed: object.pose")
    if not target_pose_was_user_provided:
        target["pose"] = {
            "pos": [add_offset(table_pos[0], target_offset[0]), add_offset(table_pos[1], target_offset[1]), z],
            "quat_xyzw": object_preset.get("default_target_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
        }
        notes.append("placement computed: target.pose")
    return notes


def quat_xyzw_from_yaw(yaw: float) -> list[float]:
    """Return an XYZW quaternion for a z-axis yaw angle [rad]."""
    half = yaw * 0.5
    return [0.0, 0.0, round(math.sin(half), 8), round(math.cos(half), 8)]


def semantic_pose_from_placement(
    spec: dict[str, Any],
    robot_preset: dict[str, Any],
    object_preset: dict[str, Any],
    placement_preset: dict[str, Any],
    semantic: str,
) -> dict[str, Any] | None:
    """Compute an object pose from a named placement rule."""
    table_pose = spec.get("scene", {}).get("table", {}).get("pose", {})
    table_pos = table_pose.get("pos") if isinstance(table_pose, dict) else None
    named = placement_preset.get("named_placements", {}) if isinstance(placement_preset, dict) else {}
    rule = named.get(semantic) if isinstance(named, dict) else None
    if not isinstance(rule, dict) or not isinstance(table_pos, list) or len(table_pos) < 2:
        return None
    offset = rule.get("xy_from_table_center", [0.0, 0.0])
    if not (isinstance(offset, list) and len(offset) >= 2):
        offset = [0.0, 0.0]
    default_placement = robot_preset.get("default_placement", {}) if isinstance(robot_preset, dict) else {}
    z = float(default_placement.get("z", object_preset.get("tabletop_z", 0.9996)))
    if rule.get("z_mode") != "tabletop" and isinstance(rule.get("z"), (int, float)):
        z = float(rule["z"])
    yaw = float(rule.get("yaw", 0.0))
    return {
        "pos": [
            round(float(table_pos[0]) + float(offset[0]), 6),
            round(float(table_pos[1]) + float(offset[1]), 6),
            z,
        ],
        "quat_xyzw": object_preset.get("default_quat_xyzw", quat_xyzw_from_yaw(yaw)),
    }


def enrich_object_from_preset(obj: dict[str, Any], object_preset: dict[str, Any]) -> None:
    """Fill object asset facts from an object preset."""
    object_fill_keys = ("usd_path", "scale", "mass", "single_rigid_body")
    deep_fill(obj, {k: object_preset[k] for k in object_fill_keys if k in object_preset})


def enrich_pick_place_vr_spec(spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Enrich PickPlaceVRSpec with deterministic robot/object preset facts."""
    enriched = copy.deepcopy(spec)
    notes: list[str] = []
    presets = load_skill_presets()
    robot = enriched.setdefault("robot", {})
    obj = enriched.setdefault("object", {})
    scene = enriched.setdefault("scene", {})
    task = enriched.setdefault("task", {})
    target = enriched.setdefault("target", {})

    robot_name = robot.get("preset") if isinstance(robot, dict) else None
    object_name = obj.get("preset") if isinstance(obj, dict) else None
    robot_preset = presets["robots"].get(robot_name) if robot_name else None
    object_preset = presets["objects"].get(object_name) if object_name else None
    placement_preset = presets["placements"].get(str(robot_name), presets["placements"].get("default", {}))
    object_pose_was_user_provided = isinstance(obj.get("pose"), dict) and not is_missing_value(obj.get("pose"))
    target_pose_was_user_provided = isinstance(target.get("pose"), dict) and not is_missing_value(target.get("pose"))

    if isinstance(robot_preset, dict):
        robot_fill_keys = (
            "robot_cfg_import",
            "usd_path",
            "urdf_path",
            "base_link_name",
            "eef_link_names",
            "controlled_joint_names",
            "hand_joint_names",
            "fixed_base",
            "init_state",
            "idle_wrist_pose",
            "motion_controller_adaptation",
        )
        deep_fill(robot, {k: robot_preset[k] for k in robot_fill_keys if k in robot_preset})
        notes.append(f"robot preset enriched: {robot_name}")
        if isinstance(robot_preset.get("default_table"), dict):
            table = scene.setdefault("table", {})
            deep_fill(table, robot_preset["default_table"])

    if isinstance(object_preset, dict):
        enrich_object_from_preset(obj, object_preset)
        notes.append(f"object preset enriched: {object_name}")

    scene.setdefault("ground", {"kind": "ground_plane"})

    if isinstance(robot_preset, dict) and isinstance(object_preset, dict):
        placement = obj.get("placement") if isinstance(obj.get("placement"), dict) else {}
        semantic = placement.get("semantic") if isinstance(placement, dict) else None
        if semantic and not object_pose_was_user_provided:
            pose = semantic_pose_from_placement(enriched, robot_preset, object_preset, placement_preset, str(semantic))
            if pose:
                obj["pose"] = pose
                object_pose_was_user_provided = True
                notes.append(f"placement computed: object.pose from semantic {semantic}")
        input_source = str(enriched.get("control", {}).get("teleop", {}).get("input_source", "openxr_hand_tracking"))
        deep_fill(task, task_names_from_presets(robot_preset, object_preset, input_source=input_source))
        notes.extend(
            compute_default_pick_place_placement(
                enriched,
                robot_preset,
                object_preset,
                object_pose_was_user_provided=object_pose_was_user_provided,
                target_pose_was_user_provided=target_pose_was_user_provided,
            )
        )
    scene_objects = enriched.get("scene_objects", [])
    if isinstance(scene_objects, list) and isinstance(robot_preset, dict):
        for item in scene_objects:
            if not isinstance(item, dict):
                continue
            item_preset_name = item.get("preset")
            item_preset = presets["objects"].get(item_preset_name) if item_preset_name else None
            if not isinstance(item_preset, dict):
                continue
            enrich_object_from_preset(item, item_preset)
            notes.append(f"scene object preset enriched: {item_preset_name}")
            item_pose_was_user_provided = isinstance(item.get("pose"), dict) and not is_missing_value(item.get("pose"))
            placement = item.get("placement") if isinstance(item.get("placement"), dict) else {}
            semantic = placement.get("semantic") if isinstance(placement, dict) else None
            if semantic and not item_pose_was_user_provided:
                pose = semantic_pose_from_placement(enriched, robot_preset, item_preset, placement_preset, str(semantic))
                if pose:
                    item["pose"] = pose
                    notes.append(f"placement computed: scene_objects.{item.get('name', item_preset_name)} from {semantic}")
    return enriched, notes
