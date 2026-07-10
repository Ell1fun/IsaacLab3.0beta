from __future__ import annotations

import re
from typing import Any

from object_intent import object_catalog, object_recommendation_lines
from presets import load_skill_presets


UNSUPPORTED_OBJECT_ALIASES = {
    "water_pipe": {
        "label": "水管",
        "aliases": ["水管", "管子", "pipe", "water pipe"],
    }
}

PRIMARY_VERBS = ("抓取", "抓", "拿取", "拿", "操作", "搬运", "pick", "grasp", "grab", "manipulate")
REMOVE_WORDS = ("不要", "不用", "不添加", "不放", "去掉", "移除", "删掉", "删除", "remove", "drop")


def _ready_object_options() -> str:
    catalog = object_catalog()
    ready = [
        f"{key}({data.get('display_name', key)})"
        for key, data in catalog.items()
        if isinstance(data, dict) and str(data.get("status", "ready")) == "ready"
    ]
    return "、".join(ready)


def _ready_robot_options() -> str:
    presets = load_skill_presets()
    robots = presets.get("robots", {})
    if not isinstance(robots, dict):
        return ""
    ready = [
        f"{key}({data.get('display_name', key)})"
        for key, data in robots.items()
        if isinstance(key, str) and isinstance(data, dict)
    ]
    return "、".join(sorted(ready))


def _object_alias_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    catalog = object_catalog()
    for preset, data in catalog.items():
        if not isinstance(data, dict):
            continue
        aliases = [str(alias) for alias in data.get("aliases", []) if alias]
        aliases.extend([str(preset), str(data.get("object_slug", preset))])
        entries.append(
            {
                "kind": "known",
                "preset": str(preset),
                "key": str(preset),
                "label": str(data.get("display_name", preset)),
                "name": str(data.get("object_slug", preset)),
                "status": str(data.get("status", "ready")),
                "blocked_reason": str(data.get("blocked_reason", "")),
                "aliases": sorted(set(aliases), key=len, reverse=True),
            }
        )
    for key, data in UNSUPPORTED_OBJECT_ALIASES.items():
        entries.append(
            {
                "kind": "unknown",
                "preset": None,
                "key": key,
                "label": str(data["label"]),
                "name": key,
                "status": "unknown",
                "blocked_reason": "object is not in object_presets.yml",
                "aliases": sorted(set(data["aliases"]), key=len, reverse=True),
            }
        )
    entries.sort(key=lambda item: max(len(alias) for alias in item["aliases"]), reverse=True)
    return entries


def _robot_alias_entries() -> list[tuple[str, str]]:
    presets = load_skill_presets()
    robots = presets.get("robots", {})
    if not isinstance(robots, dict):
        return []
    entries: list[tuple[str, str]] = []
    for preset, data in robots.items():
        if not isinstance(preset, str) or not isinstance(data, dict):
            continue
        raw_aliases = data.get("aliases", [])
        if isinstance(raw_aliases, list):
            entries.extend((str(alias), preset) for alias in raw_aliases if alias)
        entries.append((preset, preset))
        if data.get("robot_slug"):
            entries.append((str(data["robot_slug"]), preset))
        if data.get("display_name"):
            entries.append((str(data["display_name"]), preset))
    entries = [(alias.lower(), preset) for alias, preset in entries if alias]
    entries.sort(key=lambda item: len(item[0]), reverse=True)
    return entries


def _contains_alias(text: str, aliases: list[str]) -> str:
    lowered = text.lower()
    for alias in aliases:
        if alias.lower() in lowered:
            return alias
    return ""


def _has_remove_intent(text: str) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in REMOVE_WORDS)


def _has_primary_context(text: str, alias: str) -> bool:
    escaped = re.escape(alias)
    for verb in PRIMARY_VERBS:
        if re.search(f"{re.escape(verb)}[^，,。；;]*{escaped}", text, flags=re.IGNORECASE):
            return True
    return False


def _has_primary_selection_context(text: str, alias: str) -> bool:
    escaped = re.escape(alias)
    span = "[^，,。；;]{0,12}"
    return bool(
        re.search(f"{escaped}{span}主|主{span}{escaped}|{escaped}{span}作为主", text, flags=re.IGNORECASE)
    )


def _semantic_for_alias(text: str, alias: str) -> str:
    escaped = re.escape(alias)
    span = "[^，,。；;]{0,16}"
    table_span = "[^，,。；;]{0,24}"
    if re.search(f"{escaped}{span}左手|左手{span}{escaped}", text):
        return "near_left_hand"
    if re.search(f"{escaped}{span}右手|右手{span}{escaped}", text):
        return "near_right_hand"
    if re.search(f"桌{table_span}{escaped}|{escaped}{table_span}桌", text):
        return "table_back"
    return "table_center"


def _mentioned_objects(text: str) -> list[dict[str, Any]]:
    mentions: dict[str, dict[str, Any]] = {}
    for entry in _object_alias_entries():
        alias = _contains_alias(text, entry["aliases"])
        if not alias:
            continue
        key = str(entry["key"])
        mention = {
            "kind": entry["kind"],
            "preset": entry["preset"],
            "key": key,
            "label": entry["label"],
            "name": entry["name"],
            "status": entry["status"],
            "blocked_reason": entry["blocked_reason"],
            "alias": alias,
            "placement": {"semantic": _semantic_for_alias(text, alias)},
            "is_primary_candidate": _has_primary_context(text, alias),
        }
        existing = mentions.get(key)
        if existing is None or (mention["is_primary_candidate"] and not existing["is_primary_candidate"]):
            mentions[key] = mention
    return list(mentions.values())


def _detect_robot_preset(text: str, fallback: str = "") -> str:
    lowered = text.lower()
    for alias, preset in _robot_alias_entries():
        if alias and alias in lowered:
            return preset
    return fallback


def _detect_control(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = fallback or {}
    lowered = text.lower()
    wants_motion_controller = any(
        k in lowered
        for k in ("motion controller", "motion-controller", "controller", "手柄", "控制器", "pico手柄", "pico controller")
    )
    input_source = "openxr_controller" if wants_motion_controller else str(fallback.get("input_source", "openxr_hand_tracking"))
    mode = "motion_controller" if wants_motion_controller else str(fallback.get("mode", "pink_ik"))
    return {"mode": mode, "input_source": input_source}


def _empty_intent() -> dict[str, Any]:
    return {
        "robot_preset": "",
        "control": {"mode": "pink_ik", "input_source": "openxr_hand_tracking"},
        "primary_object": None,
        "scene_objects": [],
        "unknown_objects": [],
        "removed_objects": [],
    }


def _same_object(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return str(a.get("key") or a.get("preset") or a.get("name")) == str(b.get("key") or b.get("preset") or b.get("name"))


def _entry_by_key(key: str) -> dict[str, Any] | None:
    for entry in _object_alias_entries():
        if str(entry.get("key")) == key or str(entry.get("preset")) == key or str(entry.get("name")) == key:
            return entry
    return None


def _mention_from_key(key: str, placement: dict[str, Any] | None = None, label: str | None = None) -> dict[str, Any]:
    entry = _entry_by_key(key)
    if entry is None:
        return {
            "kind": "unknown",
            "preset": None,
            "key": key,
            "label": label or key,
            "name": key,
            "status": "unknown",
            "blocked_reason": "object is not in object_presets.yml",
            "alias": label or key,
            "placement": placement or {"semantic": "table_center"},
            "is_primary_candidate": False,
        }
    return {
        "kind": entry["kind"],
        "preset": entry["preset"],
        "key": entry["key"],
        "label": entry["label"],
        "name": entry["name"],
        "status": entry["status"],
        "blocked_reason": entry["blocked_reason"],
        "alias": label or str(entry["aliases"][0]),
        "placement": placement or {"semantic": "table_center"},
        "is_primary_candidate": False,
    }


def _remove_mentions(intent: dict[str, Any], mentions: list[dict[str, Any]]) -> None:
    removed_keys = {str(m.get("key")) for m in mentions}
    primary = intent.get("primary_object")
    if isinstance(primary, dict) and str(primary.get("key")) in removed_keys:
        intent["primary_object"] = None
    intent["scene_objects"] = [
        item for item in intent.get("scene_objects", []) if isinstance(item, dict) and str(item.get("key")) not in removed_keys
    ]
    intent["unknown_objects"] = [
        item for item in intent.get("unknown_objects", []) if isinstance(item, dict) and str(item.get("key")) not in removed_keys
    ]
    existing_removed = list(intent.get("removed_objects", []))
    for mention in mentions:
        label = str(mention.get("label", mention.get("key", "")))
        if label and label not in existing_removed:
            existing_removed.append(label)
    intent["removed_objects"] = existing_removed


def _remove_keys(intent: dict[str, Any], keys: list[str]) -> None:
    mentions = [_mention_from_key(key) for key in keys]
    _remove_mentions(intent, mentions)


def _add_scene_object(intent: dict[str, Any], mention: dict[str, Any]) -> None:
    if mention["kind"] == "unknown":
        unknown = intent.setdefault("unknown_objects", [])
        if not any(_same_object(item, mention) for item in unknown if isinstance(item, dict)):
            unknown.append(mention)
        return
    scene_objects = intent.setdefault("scene_objects", [])
    primary = intent.get("primary_object")
    if isinstance(primary, dict) and _same_object(primary, mention):
        return
    if not any(_same_object(item, mention) for item in scene_objects if isinstance(item, dict)):
        role = "distractor" if mention.get("preset") == "bottled_water_c01" else "prop"
        scene_objects.append({**mention, "role": role})


def build_intent_from_llm_payload(
    user_text: str,
    payload: dict[str, Any],
    draft_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a model-produced intent patch/full intent to the current draft intent."""
    intent = {**_empty_intent(), **(draft_intent or {})}
    robot_preset = payload.get("robot_preset")
    if isinstance(robot_preset, str) and robot_preset:
        intent["robot_preset"] = robot_preset
    else:
        intent["robot_preset"] = _detect_robot_preset(user_text, str(intent.get("robot_preset", "")))

    control = payload.get("control")
    if isinstance(control, dict):
        intent["control"] = {
            "mode": str(control.get("mode", intent.get("control", {}).get("mode", "pink_ik"))),
            "input_source": str(
                control.get("input_source", intent.get("control", {}).get("input_source", "openxr_hand_tracking"))
            ),
        }
    else:
        intent["control"] = _detect_control(user_text, intent.get("control") if isinstance(intent.get("control"), dict) else None)

    remove_keys = [str(key) for key in payload.get("remove_object_keys", []) if key]
    if remove_keys:
        _remove_keys(intent, remove_keys)

    primary_key = payload.get("primary_object_key")
    if isinstance(primary_key, str) and primary_key:
        primary = _mention_from_key(primary_key, _semantic_from_payload(payload.get("primary_placement")))
        if primary["kind"] == "known":
            intent["primary_object"] = primary
            intent["scene_objects"] = [
                item for item in intent.get("scene_objects", []) if isinstance(item, dict) and not _same_object(item, primary)
            ]
        else:
            _add_scene_object(intent, primary)

    for item in payload.get("scene_object_keys", []):
        if isinstance(item, str):
            mention = _mention_from_key(item)
            _add_scene_object(intent, mention)
            continue
        if isinstance(item, dict):
            key = str(item.get("key", ""))
            if not key:
                continue
            mention = _mention_from_key(key, _semantic_from_payload(item.get("placement")), str(item.get("label", "")) or None)
            role = str(item.get("role", "prop"))
            _add_scene_object(intent, {**mention, "role": role})

    for item in payload.get("unknown_object_keys", []):
        if isinstance(item, str):
            mention = _mention_from_key(item)
        elif isinstance(item, dict):
            key = str(item.get("key", ""))
            if not key:
                continue
            mention = _mention_from_key(key, _semantic_from_payload(item.get("placement")), str(item.get("label", "")) or None)
        else:
            continue
        unknown = intent.setdefault("unknown_objects", [])
        if not any(_same_object(existing, mention) for existing in unknown if isinstance(existing, dict)):
            unknown.append(mention)

    intent.pop("ambiguous_primary_candidates", None)
    return intent


def _semantic_from_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and isinstance(value.get("semantic"), str):
        return {"semantic": value["semantic"]}
    return None


def build_or_update_intent(user_text: str, draft_intent: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a new task intent or update an existing one from the latest user message."""
    intent = {**_empty_intent(), **(draft_intent or {})}
    intent["robot_preset"] = _detect_robot_preset(user_text, str(intent.get("robot_preset", "")))
    intent["control"] = _detect_control(user_text, intent.get("control") if isinstance(intent.get("control"), dict) else None)
    mentions = _mentioned_objects(user_text)

    if draft_intent and _has_remove_intent(user_text):
        _remove_mentions(intent, mentions)
        return intent

    primary_mentions = [m for m in mentions if m["is_primary_candidate"]]
    if len(primary_mentions) == 1:
        primary = primary_mentions[0]
        if primary["kind"] == "unknown":
            _add_scene_object(intent, primary)
        else:
            intent["primary_object"] = primary
        for mention in mentions:
            if not _same_object(mention, primary):
                _add_scene_object(intent, mention)
        return intent

    if len(primary_mentions) > 1:
        intent["ambiguous_primary_candidates"] = primary_mentions
        for mention in mentions:
            _add_scene_object(intent, mention)
        return intent

    if draft_intent and any(word in user_text for word in ("主", "主物体", "作为主", "主操作")) and mentions:
        ready_mentions = [
            m for m in mentions if m["kind"] == "known" and _has_primary_selection_context(user_text, str(m.get("alias", "")))
        ]
        if len(ready_mentions) == 1:
            intent["primary_object"] = ready_mentions[0]
            intent["scene_objects"] = [
                item for item in intent.get("scene_objects", []) if not _same_object(item, ready_mentions[0])
            ]
            intent.pop("ambiguous_primary_candidates", None)
            for mention in mentions:
                if not _same_object(mention, ready_mentions[0]):
                    _add_scene_object(intent, mention)
            return intent

    if not draft_intent and len(mentions) == 1 and mentions[0]["kind"] == "known":
        intent["primary_object"] = mentions[0]
        return intent

    for mention in mentions:
        _add_scene_object(intent, mention)
    if not draft_intent and len([m for m in mentions if m["kind"] == "known"]) > 1:
        intent["ambiguous_primary_candidates"] = mentions
    return intent


def validate_intent(intent: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return deterministic errors/questions for an intent that is not safe to generate."""
    errors: list[str] = []
    questions: list[str] = []
    presets = load_skill_presets()
    robots = presets.get("robots", {})
    if not isinstance(robots, dict):
        robots = {}
    robot_preset = str(intent.get("robot_preset", "") or "")
    if not robot_preset or robot_preset not in robots:
        errors.append("robot preset is missing or unknown")
        options = _ready_robot_options()
        questions.append(
            "当前还没有可用的机器人 preset，或我无法识别你说的机器人名称。"
            + ("请从以下机器人中选择一个：\n" + options if options else "请提供 robot_preset（例如 unitree_g1_inspire_ftp）。")
        )
    unknown_objects = [item for item in intent.get("unknown_objects", []) if isinstance(item, dict)]
    scene_objects = [item for item in intent.get("scene_objects", []) if isinstance(item, dict)]
    primary = intent.get("primary_object")
    all_known = [item for item in scene_objects if item.get("kind") == "known"]
    if isinstance(primary, dict):
        all_known.append(primary)

    if unknown_objects:
        labels = "、".join(sorted({str(item.get("label", item.get("key"))) for item in unknown_objects}))
        errors.append(f"unknown object mention(s): {labels}")
        questions.append(
            f"{labels} 当前不在 object_presets.yml 中，不能直接生成。请选择："
            f"1. 改用已有 ready 物体（{_ready_object_options()}）；"
            "2. 提供 USD 路径、scale、single_rigid_body 先新增 preset；"
            f"3. 本次不添加 {labels}。"
        )

    blocked = [item for item in all_known if str(item.get("status", "ready")) != "ready"]
    if blocked:
        labels = "、".join(str(item.get("label", item.get("preset"))) for item in blocked)
        errors.append(f"object preset is not ready: {labels}")
        details = "\n".join(f"- {item.get('preset')}: {item.get('blocked_reason', 'not ready')}" for item in blocked)
        questions.append(
            "这些物体 preset 还不是 ready，不能直接生成："
            f"{details}。请选择：1. 改用 ready 物体；2. 先补齐资产信息；3. 本次不添加。"
        )

    ambiguous = [item for item in intent.get("ambiguous_primary_candidates", []) if isinstance(item, dict)]
    if not isinstance(primary, dict):
        candidates = [item for item in all_known if str(item.get("status", "ready")) == "ready"] or ambiguous
        if len(candidates) > 1:
            labels = "、".join(str(item.get("label", item.get("preset"))) for item in candidates)
            errors.append(f"primary object is unclear among mentioned objects: {labels}")
            questions.append(
                "我检测到多个已知物体，但没有明确哪个是主 pick-place object："
                f"{labels}。"
                "请明确一个主操作物体；其余物体可以作为 scene_objects 附加摆放。"
            )
        elif not candidates:
            errors.append("no ready primary object")
            questions.append(
                "当前还没有可用的主 pick-place object。请从 ready object presets 中选择一个主物体：\n"
                + "\n".join(line for line in object_recommendation_lines() if "当前状态：blocked" not in line)
            )
    return errors, questions


def intent_to_spec(intent: dict[str, Any]) -> dict[str, Any]:
    """Convert a validated intent to a minimal PickPlaceVRSpec."""
    primary_raw = intent.get("primary_object")
    control_raw = intent.get("control")
    primary: dict[str, Any] = primary_raw if isinstance(primary_raw, dict) else {}
    control: dict[str, Any] = control_raw if isinstance(control_raw, dict) else {}
    scene_objects = [
        {
            "name": str(item.get("name", item.get("preset"))),
            "preset": str(item.get("preset")),
            "role": str(item.get("role", "prop")),
            "placement": item.get("placement", {"semantic": "table_center"}),
        }
        for item in intent.get("scene_objects", [])
        if isinstance(item, dict) and item.get("kind") == "known" and str(item.get("status", "ready")) == "ready"
    ]
    spec: dict[str, Any] = {
        "spec_version": "0.1",
        "task": {"family": "manager_based/manipulation/pick_place"},
        "robot": {"preset": str(intent.get("robot_preset", ""))},
        "scene": {},
        "object": {
            "preset": str(primary.get("preset", "")),
            "placement": primary.get("placement", {"semantic": "table_center"}),
        },
        "target": {},
        "control": {
            "mode": str(control.get("mode", "pink_ik")),
            "teleop": {
                "enabled": True,
                "input_source": str(control.get("input_source", "openxr_hand_tracking")),
                "world_T_anchor": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
        },
    }
    if spec["control"]["mode"] == "motion_controller":
        spec["control"]["teleop"]["hand"] = {"mode": "trigger_open_close"}
    if scene_objects:
        spec["scene_objects"] = scene_objects
    return spec
