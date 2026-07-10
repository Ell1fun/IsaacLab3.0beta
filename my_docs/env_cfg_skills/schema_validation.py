from __future__ import annotations

import copy
from typing import Any

from presets import load_skill_presets
from skill_types import SkillDef


def _schema_node(schema: dict[str, Any], *path: str) -> dict[str, Any] | None:
    node: Any = schema
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    """Return value when it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def schema_with_pick_place_vr_preset_enums(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a runtime schema whose preset enums are derived from preset files."""
    resolved = copy.deepcopy(schema)
    presets = load_skill_presets()
    robot_node = _schema_node(resolved, "properties", "robot", "properties", "preset")
    object_node = _schema_node(resolved, "properties", "object", "properties", "preset")
    scene_object_node = _schema_node(resolved, "properties", "scene_objects", "items", "properties", "preset")
    if robot_node is not None:
        robot_node["enum"] = sorted(str(key) for key in presets.get("robots", {}).keys())
    if object_node is not None:
        object_node["enum"] = sorted(str(key) for key in presets.get("objects", {}).keys())
    if scene_object_node is not None:
        scene_object_node["enum"] = sorted(str(key) for key in presets.get("objects", {}).keys())
    return resolved


def validate_pick_place_vr_preset_keys(spec: dict[str, Any]) -> list[str]:
    """Validate preset keys even when jsonschema is unavailable."""
    presets = load_skill_presets()
    robot = _dict_or_empty(spec.get("robot"))
    obj = _dict_or_empty(spec.get("object"))
    scene_objects_raw = spec.get("scene_objects")
    scene_objects: list[Any] = scene_objects_raw if isinstance(scene_objects_raw, list) else []
    robot_preset = robot.get("preset")
    object_preset = obj.get("preset")
    errors: list[str] = []
    if robot_preset and robot_preset not in presets.get("robots", {}):
        errors.append(f"robot.preset must be one of presets/robot_presets.yml keys: {robot_preset!r}")
    if object_preset and object_preset not in presets.get("objects", {}):
        errors.append(f"object.preset must be one of presets/object_presets.yml keys: {object_preset!r}")
    object_keys = presets.get("objects", {})
    reserved_names = {"object", "robot", "packing_table", "terrain", "ground"}
    seen_names: set[str] = set()
    for index, item in enumerate(scene_objects):
        if not isinstance(item, dict):
            errors.append(f"scene_objects[{index}] must be an object")
            continue
        name = item.get("name")
        preset = item.get("preset")
        if preset and preset not in object_keys:
            errors.append(f"scene_objects[{index}].preset must be one of presets/object_presets.yml keys: {preset!r}")
        if isinstance(name, str):
            if name in reserved_names:
                errors.append(f"scene_objects[{index}].name is reserved: {name!r}")
            if name in seen_names:
                errors.append(f"scene_objects[{index}].name is duplicated: {name!r}")
            seen_names.add(name)
    errors.extend(validate_pick_place_vr_placement_keys(spec, presets))
    return errors


def validate_pick_place_vr_placement_keys(spec: dict[str, Any], presets: dict[str, dict[str, Any]]) -> list[str]:
    """Validate semantic placement keys against placement_presets.yml."""
    errors: list[str] = []
    robot = _dict_or_empty(spec.get("robot"))
    robot_preset = str(robot.get("preset", "default"))
    placements = _dict_or_empty(presets.get("placements"))
    placement_preset = _dict_or_empty(placements.get(robot_preset, placements.get("default", {})))
    named = placement_preset.get("named_placements", {}) if isinstance(placement_preset, dict) else {}
    allowed = set(named.keys()) if isinstance(named, dict) else set()

    def check(path: str, node: Any) -> None:
        placement = node.get("placement") if isinstance(node, dict) else None
        semantic = placement.get("semantic") if isinstance(placement, dict) else None
        if semantic and semantic not in allowed:
            errors.append(f"{path}.placement.semantic must be one of placement_presets.yml keys: {semantic!r}")

    check("object", spec.get("object"))
    scene_objects_raw = spec.get("scene_objects")
    scene_objects: list[Any] = scene_objects_raw if isinstance(scene_objects_raw, list) else []
    for index, item in enumerate(scene_objects):
        check(f"scene_objects[{index}]", item)
    return errors


def basic_pick_place_vr_checks(spec: dict[str, Any]) -> list[str]:
    """Minimal validator used when jsonschema is unavailable."""
    errors: list[str] = []
    for k in ("spec_version", "task", "robot", "scene", "object", "target", "control"):
        if k not in spec:
            errors.append(f"Missing required top-level field: {k}")

    control = _dict_or_empty(spec.get("control"))
    teleop = _dict_or_empty(control.get("teleop"))
    if bool(teleop.get("enabled")):
        if not teleop.get("input_source"):
            errors.append("control.teleop.enabled=true requires: control.teleop.input_source")
        if not teleop.get("world_T_anchor"):
            errors.append("control.teleop.enabled=true requires: control.teleop.world_T_anchor")
        robot = _dict_or_empty(spec.get("robot"))
        idle = _dict_or_empty(robot.get("idle_wrist_pose"))
        if not idle.get("left") or not idle.get("right"):
            errors.append("control.teleop.enabled=true requires: robot.idle_wrist_pose.left and robot.idle_wrist_pose.right")

    hand = _dict_or_empty(teleop.get("hand"))
    if hand.get("mode") == "dexpilot" and not hand.get("dexpilot_config_path"):
        errors.append("control.teleop.hand.mode=dexpilot requires: control.teleop.hand.dexpilot_config_path")
    errors.extend(validate_pick_place_vr_preset_keys(spec))
    return errors


def validate_spec(skill: SkillDef, spec: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate spec against schema. Empty list means validation passed."""
    if skill.name == "pick_place_vr":
        schema = schema_with_pick_place_vr_preset_enums(schema)
        preset_errors = validate_pick_place_vr_preset_keys(spec)
        if preset_errors:
            return preset_errors

    try:
        import jsonschema  # type: ignore
    except Exception:
        if skill.name == "pick_place_vr":
            return basic_pick_place_vr_checks(spec)
        return ["jsonschema is not installed; no validator available for this skill"]

    try:
        jsonschema.validate(instance=spec, schema=schema)
    except jsonschema.ValidationError as e:  # type: ignore[attr-defined]
        return [e.message]
    return []
