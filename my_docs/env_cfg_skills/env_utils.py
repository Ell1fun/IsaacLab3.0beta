from __future__ import annotations

import copy
import json
import os
from typing import Any


def repo_root_from_this_file() -> str:
    """Return the IsaacLab repo root inferred from this module location."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def abs_from_repo_root(path_from_repo_root: str) -> str:
    """Convert a repo-root-relative path to an absolute path."""
    return os.path.join(repo_root_from_this_file(), path_from_repo_root)


def default_local_config_path() -> str:
    """Return the local ignored orchestrator config path."""
    return os.path.join(os.path.dirname(__file__), "orchestrator_config.local.yml")


def presets_dir() -> str:
    """Return the skill presets directory."""
    return os.path.join(os.path.dirname(__file__), "presets")


def read_text(path: str) -> str:
    """Read a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_yaml_or_json(path: str) -> Any:
    """Load YAML/JSON into Python objects based on file extension."""
    suffix = os.path.splitext(path)[1].lower()
    text = read_text(path)
    if suffix == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("YAML parsing requires PyYAML (import yaml failed)") from e
    return yaml.safe_load(text)


def dump_yaml_or_json(data: Any, fmt: str) -> str:
    """Dump Python objects to YAML/JSON strings for CLI output."""
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False)
    if fmt in ("yml", "yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML dumping requires PyYAML (import yaml failed)") from e
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=4096)
    raise ValueError(f"Unsupported format: {fmt}")


def is_missing_value(value: Any) -> bool:
    """Return true when a model-produced value should be overwritten by preset facts."""
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return stripped == "" or stripped.upper().startswith("TODO")
    if isinstance(value, list):
        return len(value) == 0
    if isinstance(value, dict):
        return len(value) == 0
    return False


def deep_fill(dst: dict[str, Any], src: dict[str, Any], *, overwrite: bool = False) -> None:
    """Fill missing values in dst from src, recursively."""
    for key, value in src.items():
        if key not in dst or overwrite or is_missing_value(dst.get(key)):
            dst[key] = copy.deepcopy(value)
            continue
        if isinstance(dst.get(key), dict) and isinstance(value, dict):
            deep_fill(dst[key], value, overwrite=overwrite)


def pascal_from_slug(slug: str) -> str:
    """Convert snake_case-ish slugs to PascalCase for Gym ids/classes."""
    return "".join(part[:1].upper() + part[1:] for part in slug.replace("-", "_").split("_") if part)
