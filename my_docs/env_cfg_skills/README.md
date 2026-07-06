# env_cfg Skills（AI Agent 自动生成）

这个文件夹存放“AI Agent 自动生成 Isaac Lab env_cfg”的执行材料。完整 skill 不只是 schema，而是 schema + prompts + checklist + examples + orchestrator workflow。

## 当前结构

- `orchestrator.py`：本地编排入口，负责 route / chat / generate / validate / template。
- `orchestrator_config.example.yml`：本地模型 API 配置示例。
- `orchestrator_config.local.yml`：你的本地真实配置（被 `.gitignore` 忽略）。
- `schemas/pick_place_vr_spec.schema.yml`：PickPlaceVRSpec 的结构约束。
- `prompts/pick_place_vr_spec_builder.md`：自然语言 → spec 的补齐规则。
- `prompts/pick_place_vr_generator.md`：spec → env_cfg / task 注册 的生成规则。
- `prompts/env_cfg_generator.md`：早期通用 env_cfg prompt，保留作参考。
- `checklists/post_generation_validation.md`：生成后验收清单。
- `examples/r11_pick_place_hand_tracking/`：R11 示例 spec 与期望摘要。
- `examples/g1_pick_place_hand_tracking/`：G1 Inspire Hand 示例 spec 与期望摘要。

## dry_run

- `dry_run: true`：只演练，不真正写入模型生成的文件。
- 真正写文件需要显式加 `--write`。
- 建议早期一直保持 `dry_run: true`，先看输出文件清单和内容是否合理。

## 本地 API 配置

复制示例配置：

```bash
cp my_docs/env_cfg_skills/orchestrator_config.example.yml   my_docs/env_cfg_skills/orchestrator_config.local.yml
```

编辑 `orchestrator_config.local.yml`：

```yaml
llm:
  api_base_url: "https://你的模型服务地址"
  api_key: "你的本地 API Key"
  api_key_env: "ENV_CFG_LLM_API_KEY"
  model: "你的模型名"
workflow:
  max_spec_iterations: 5
  dry_run: true
```

## 使用方式

交互式 demo：

```bash
python my_docs/env_cfg_skills/orchestrator.py chat --enable_api
```

一次性生成：

```bash
python my_docs/env_cfg_skills/orchestrator.py generate "帮我做一个 pick_place VR teleop env_cfg" --enable_api
```

本地 dry-run（不调用 API）：

```bash
python my_docs/env_cfg_skills/orchestrator.py chat
```
