#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-my_docs/env_cfg_skills/orchestrator_config.local.yml}"

python my_docs/env_cfg_skills/chat_api.py --config "$CONFIG_PATH"
