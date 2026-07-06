# env_cfg Skills（AI Agent 自动生成）

这个文件夹用于存放“让 AI Agent 自动生成 env_cfg 的技能/模板/提示词（prompts）”，目的是把可复用的生成规范沉淀下来，避免每次从零描述需求。

## 推荐结构

- `prompts/`
  - `env_cfg_generator.md`：主提示词（生成一个新任务 env_cfg 的完整骨架）
  - `teleop_pipeline.md`：可选（生成/修改 IsaacTeleop pipeline builder）
  - `robot_cfg_contract.md`：可选（约束 robot cfg / articulation cfg 的输入输出契约）
- `schemas/`
  - `env_cfg_requirements.yml`：需求字段清单（任务目标、动作、观测、reset、奖励/终止等）
  - `naming_conventions.yml`：命名规范（Gym ID、文件名、class 名、link/joint 名）
- `examples/`
  - `pickplace_like/`：一个“像 pick-place 的任务”输入/输出样例
  - `locomotion_like/`：一个“像 locomotion 的任务”输入/输出样例

## 使用方式（建议）

1. 先在 `schemas/env_cfg_requirements.yml` 里填需求字段（越具体越好）。
2. 选择 `prompts/env_cfg_generator.md` 作为主 prompt，把需求字段作为输入。
3. 生成结果后，把关键决策（action layout、joint/link 列表、teleop pipeline 约定）回填到 `schemas/`，形成可复用约束。

