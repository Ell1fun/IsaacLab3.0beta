from __future__ import annotations

import argparse
import os

from env_utils import default_local_config_path, load_yaml_or_json
from skill_types import LlmConfig, OrchestratorConfig, WorkflowConfig


def default_llm_config() -> LlmConfig:
    """Build default LLM config from environment variables."""
    api_base_url = os.environ.get("ENV_CFG_LLM_API_BASE_URL", "").strip() or "https://example.invalid"
    return LlmConfig(
        api_base_url=api_base_url,
        api_key_env="ENV_CFG_LLM_API_KEY",
        model=os.environ.get("ENV_CFG_LLM_MODEL", "gpt-4.1"),
        api_style=os.environ.get("ENV_CFG_LLM_API_STYLE", "chat_completions"),
        timeout_s=int(os.environ.get("ENV_CFG_LLM_TIMEOUT_S", "300")),
    )


def default_orchestrator_config() -> OrchestratorConfig:
    """Build default orchestrator config from env vars and safe defaults."""
    return OrchestratorConfig(llm=default_llm_config(), workflow=WorkflowConfig())


def load_orchestrator_config(path: str | None) -> OrchestratorConfig:
    """Load orchestrator config from YAML/JSON or fallback to local config/env vars."""
    default = default_orchestrator_config()
    if not path:
        local_path = default_local_config_path()
        if os.path.exists(local_path):
            path = local_path
        else:
            return default

    raw = load_yaml_or_json(path)
    if not isinstance(raw, dict):
        raise ValueError("Orchestrator config must be a YAML/JSON object")

    llm_raw = raw.get("llm", {}) or {}
    workflow_raw = raw.get("workflow", {}) or {}
    if not isinstance(llm_raw, dict):
        raise ValueError("Config field 'llm' must be an object")
    if not isinstance(workflow_raw, dict):
        raise ValueError("Config field 'workflow' must be an object")

    llm = LlmConfig(
        api_base_url=str(llm_raw.get("api_base_url", default.llm.api_base_url)),
        api_key_env=str(llm_raw.get("api_key_env", default.llm.api_key_env)),
        model=str(llm_raw.get("model", default.llm.model)),
        api_key=str(llm_raw["api_key"]).strip() if llm_raw.get("api_key") else None,
        api_style=str(llm_raw.get("api_style", default.llm.api_style)),
        timeout_s=int(llm_raw.get("timeout_s", default.llm.timeout_s)),
        max_retries=int(llm_raw.get("max_retries", default.llm.max_retries)),
        retry_backoff_s=float(llm_raw.get("retry_backoff_s", default.llm.retry_backoff_s)),
    )
    workflow = WorkflowConfig(
        max_spec_iterations=int(workflow_raw.get("max_spec_iterations", default.workflow.max_spec_iterations)),
        dry_run=bool(workflow_raw.get("dry_run", default.workflow.dry_run)),
    )
    return OrchestratorConfig(llm=llm, workflow=workflow)


def runtime_config_from_args(args: argparse.Namespace) -> OrchestratorConfig:
    """Create runtime config from --config plus CLI overrides."""
    cfg = load_orchestrator_config(getattr(args, "config", None))
    llm_cfg = cfg.llm
    if getattr(args, "api_base_url", None) is not None:
        llm_cfg = LlmConfig(
            api_base_url=args.api_base_url,
            api_key_env=llm_cfg.api_key_env,
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            api_style=llm_cfg.api_style,
            timeout_s=llm_cfg.timeout_s,
            max_retries=llm_cfg.max_retries,
            retry_backoff_s=llm_cfg.retry_backoff_s,
        )
    if getattr(args, "model", None) is not None:
        llm_cfg = LlmConfig(
            api_base_url=llm_cfg.api_base_url,
            api_key_env=llm_cfg.api_key_env,
            model=args.model,
            api_key=llm_cfg.api_key,
            api_style=llm_cfg.api_style,
            timeout_s=llm_cfg.timeout_s,
            max_retries=llm_cfg.max_retries,
            retry_backoff_s=llm_cfg.retry_backoff_s,
        )
    return OrchestratorConfig(
        llm=llm_cfg,
        workflow=WorkflowConfig(
            max_spec_iterations=cfg.workflow.max_spec_iterations,
            dry_run=not bool(getattr(args, "write", False)),
        ),
    )
