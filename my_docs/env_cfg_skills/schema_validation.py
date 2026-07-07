from __future__ import annotations

from typing import Any

from skill_types import SkillDef


def basic_pick_place_vr_checks(spec: dict[str, Any]) -> list[str]:
    """Minimal validator used when jsonschema is unavailable."""
    errors: list[str] = []
    for k in ("spec_version", "task", "robot", "scene", "object", "target", "control"):
        if k not in spec:
            errors.append(f"Missing required top-level field: {k}")

    control = spec.get("control") if isinstance(spec.get("control"), dict) else {}
    teleop = control.get("teleop") if isinstance(control.get("teleop"), dict) else {}
    if bool(teleop.get("enabled")):
        if not teleop.get("input_source"):
            errors.append("control.teleop.enabled=true requires: control.teleop.input_source")
        if not teleop.get("world_T_anchor"):
            errors.append("control.teleop.enabled=true requires: control.teleop.world_T_anchor")
        robot = spec.get("robot") if isinstance(spec.get("robot"), dict) else {}
        idle = robot.get("idle_wrist_pose") if isinstance(robot.get("idle_wrist_pose"), dict) else {}
        if not idle.get("left") or not idle.get("right"):
            errors.append("control.teleop.enabled=true requires: robot.idle_wrist_pose.left and robot.idle_wrist_pose.right")

    hand = teleop.get("hand") if isinstance(teleop.get("hand"), dict) else {}
    if hand.get("mode") == "dexpilot" and not hand.get("dexpilot_config_path"):
        errors.append("control.teleop.hand.mode=dexpilot requires: control.teleop.hand.dexpilot_config_path")
    return errors


def validate_spec(skill: SkillDef, spec: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate spec against schema. Empty list means validation passed."""
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
