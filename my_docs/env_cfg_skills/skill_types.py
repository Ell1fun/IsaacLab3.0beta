from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillDef:
    """A registered skill definition used by router and workflows."""

    name: str
    description: str
    triggers: tuple[str, ...]
    schema_path: str
    generator_prompt_path: str | None = None
    spec_builder_prompt_path: str | None = None


@dataclass(frozen=True)
class LlmConfig:
    """LLM API configuration loaded from local config, env vars, or CLI overrides."""

    api_base_url: str
    api_key_env: str
    model: str
    api_key: str | None = None
    api_style: str = "chat_completions"
    timeout_s: int = 300
    max_retries: int = 2
    retry_backoff_s: float = 1.0


@dataclass(frozen=True)
class WorkflowConfig:
    """Workflow runtime knobs."""

    max_spec_iterations: int = 3
    dry_run: bool = True


@dataclass(frozen=True)
class OrchestratorConfig:
    """Top-level runtime config used by the CLI orchestrator."""

    llm: LlmConfig
    workflow: WorkflowConfig
