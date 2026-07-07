# env_cfg Skills（AI Agent 自动生成）

这个文件夹存放“AI Agent 自动生成 Isaac Lab env_cfg”的执行材料。完整 skill 不只是 schema，而是 schema + presets + templates + prompts + checklist + examples + orchestrator workflow。

## 当前结构

- `orchestrator.py`：本地 CLI 编排入口，只负责 route / chat / generate / validate / template 的命令分发和交互循环。
- `pipeline.py`：把 Stage A、schema gate、Stage B、writer 串成完整生成流水线。
- `spec_builder.py`：自然语言到 spec 的本地 parser；必要时才调用 AI API 作为 fallback。
- `presets.py`：加载 robot/object presets、解析继承、补齐仓库事实、计算默认 placement。
- `generator_workflow.py`：exact template 匹配、template generator、adaptation candidate、参考 env_cfg 结构化片段抽取、env_cfg 薄封装、task 注册内容、teleop 命令和写文件逻辑。
- `post_generation_tests.py`：写文件前的自动测试 gate，检查 Python 语法、TODO、task 注册和可复制命令。
- `schema_validation.py`：schema gate 和 jsonschema 缺失时的兜底校验。
- `llm_client.py`：模型 API 调用封装，支持 OpenAI-compatible 和 Ark responses 风格。
- `registry.py`：skill registry 和 prompt 路由。
- `config_loader.py`：读取本地配置、环境变量、CLI override 和 `--write` gate。
- `env_utils.py`：路径、YAML/JSON、deep fill、缺失值判断等通用工具。
- `starter_specs.py`：starter spec 示例。
- `orchestrator_config.example.yml`：本地模型 API 配置示例。
- `orchestrator_config.local.yml`：你的本地真实配置（被 `.gitignore` 忽略）。
- `schemas/pick_place_vr_spec.schema.yml`：PickPlaceVRSpec 的结构约束。
- `presets/robot_presets.yml`：仓库内已知机器人 preset 的确定事实，例如 EEF link、IK joints、hand joints、参考 env_cfg、正确输出目录。
- `presets/object_presets.yml`：仓库内已知物体 preset 的确定事实，例如 USD 路径、scale、质量、是否需要单刚体。
- `prompts/pick_place_vr_spec_builder.md`：自然语言 → spec 的补齐规则。
- `prompts/pick_place_vr_generator.md`：spec → env_cfg / task 注册 的生成规则（当前作为 fallback/参考，主路径优先 deterministic template generator）。
- `prompts/pick_place_vr_template_adaptation.md`：缺 exact template 时，AI 根据目标机器人模板和同输入源参考模板推断 `adaptation_spec` 的迁移规则；不让模型直接写最终 Python。
- `prompts/env_cfg_generator.md`：早期通用 env_cfg prompt，保留作参考。
- `checklists/post_generation_validation.md`：生成后验收清单。
- `examples/r11_pick_place_hand_tracking/`：R11 示例 spec 与期望摘要。
- `examples/g1_pick_place_hand_tracking/`：G1 Inspire Hand 示例 spec 与期望摘要。

## preset enrichment

模型生成 spec 后，orchestrator 会先根据 `robot.preset` 和 `object.preset` 自动补齐仓库已知信息，再进行 schema gate。

这样做的原因：

- schema 只约束字段结构，不知道 GR1T2 的 EEF link 叫什么。
- 模型容易对 `base_link_name`、`hand_joint_names`、`table.usd_path` 产生 TODO。
- 这些值很多在仓库已有参考任务中是确定的，应该由 presets 固化，而不是让模型每次猜。

当前 enrichment 会补齐：

- robot cfg import
- base link / EEF links
- Pink IK controlled joints
- hand joint names
- robot init state
- idle wrist pose
- table USD / pose
- object USD / scale / mass / single rigid body
- object pose / target pose（由 `robot.default_placement` + table center 自动计算）
- auto file stem / Gym task id

## presets 字段约定（建议把它当成“事实表”维护）

这两份 YAML 的关键目标是：把“仓库里已经确定的事实”固化下来，让 workflow 程序消费，而不是让模型每次猜。

### robot_presets.yml（robot preset）

- **标识字段**：`display_name`、`robot_slug`（用于生成 `task.gym_id` 和文件 stem）
- **模板字段**：`supported_teleop_templates.<input_source>.module/class`（决定是否存在 exact template）
- **参考字段**：`reference_env_cfg`（structured snippets 抽取上下文时使用）
- **输出位置**：`output_dir`、`registration_module`（生成文件写入与注册写入位置）
- **机器人事实（enrichment）**：`robot_cfg_import`、`base_link_name`、`eef_link_names`、`controlled_joint_names`、`hand_joint_names`、`fixed_base`、`init_state`、`idle_wrist_pose`
- **摆放事实（placement）**：`default_table`、`default_placement`
- **motion controller 适配事实（可选）**：`motion_controller_adaptation`（wrist offset、reference_env_cfg、use_wrist_rotation/use_wrist_position、grasp_scalar）

### object_presets.yml（object preset）

- **标识字段**：`display_name`、`object_slug`
- **资产事实**：`usd_path`（若是 TODO 必须阻断）、`scale`、`mass`（可选）、`single_rigid_body`
- **参考字段（可选）**：`reference_env_cfg`（当 usd_path 需要从现有任务里抽取时）
- **姿态相关（可选）**：`default_quat_xyzw`、`default_target_quat_xyzw`、`tabletop_z`

## prompts 的职责边界

- `pick_place_vr_spec_builder.md`：Stage A（schema 未通过且 enable_api=true 时）用于补齐 spec，只输出 YAML/JSON spec。
- `pick_place_vr_template_adaptation.md`：Stage B（缺 exact template 且 enable_api=true 时）用于输出 `adaptation_spec`，禁止输出 Python。
- `pick_place_vr_generator.md`：fallback/参考；当前 pick_place_vr 主路径优先 deterministic template generator，不依赖它生成最终代码。
- `env_cfg_generator.md`：早期通用 prompt，保留作参考，不建议作为生产路径。

## template generator

对于仓库里已有完整 pick_place teleop 模板的机器人，Stage B 不再调用大模型写 env_cfg，而是程序确定性生成。现在模板匹配使用 `(robot.preset, control.teleop.input_source)`，避免 handtracking 和 motion controller 共用错误模板：

- `fourier_gr1t2_high_pd + openxr_hand_tracking` → 继承 `PickPlaceGR1T2EnvCfg`
- `unitree_g1_inspire_ftp + openxr_hand_tracking` → 继承 `PickPlaceG1InspireFTPEnvCfg`
- `seres_r11_a2_high_pd + openxr_hand_tracking` → 继承 `PickPlaceR11EnvCfg`
- `seres_r11_a2_high_pd + openxr_controller` → 继承 `PickPlaceR11TriEnvCfg`

模板生成器只生成薄封装文件：复用已有 env_cfg 的控制、teleop pipeline、IK、observations、terminations，只覆盖 spec 中确定的 object/table/target 参数，并追加 Auto task 注册。

如果用户请求的组合没有 exact template，例如 `unitree_g1_inspire_ftp + openxr_controller`，生成器会输出 `adaptation_candidate`，并找到可参考的同输入源模板，例如 `seres_r11_a2_high_pd + openxr_controller`。启用 API 后，adaptation generator 不再让模型直接写 Python，而是用 AST 抽取 imports、EnvCfg、teleop、pipeline、controller、source、retargeter、action、hand、trigger 等相关片段，作为 structured snippets 发给模型，让模型输出 `adaptation_spec`。随后 `generator_workflow.py` 根据 `adaptation_spec` 确定性渲染 candidate env_cfg。候选文件仍必须通过 `post_generation_tests.py` 与更重的 IsaacLab smoke tests 后才允许写入。

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

只测试模型 API（不走 skill/workflow）：

```bash
bash my_docs/env_cfg_skills/chat_api.sh my_docs/env_cfg_skills/orchestrator_config.local.yml
```

如果你担心 tokens 超限，可以限制 chat 上下文长度（按字符近似控制）：

```bash
python my_docs/env_cfg_skills/chat_api.py --config my_docs/env_cfg_skills/orchestrator_config.local.yml --max_history_chars 12000 --max_turns 10
```

一次性生成：

```bash
python my_docs/env_cfg_skills/orchestrator.py generate "帮我做一个 pick_place VR teleop env_cfg" --enable_api
```

本地 dry-run（不调用 API）：

```bash
python my_docs/env_cfg_skills/orchestrator.py chat
```
