# Prompt：生成一个 IsaacLab env_cfg（ManagerBased）任务骨架

## 目标

基于用户给定的机器人（robot cfg / articulation）、任务描述与资产清单，生成一个可运行的 `*env_cfg.py`，并保持与仓库现有风格一致。

## 输入（用户需提供）

- 任务类型：`pick_place / reach / push / locomotion / ...`
- 机器人信息：
  - robot cfg import 路径（例如 `isaaclab_assets.robots.seres_r11`）
  - EEF link（左右手/单手）
  - 主要可控关节列表（IK controlled joints）与手指关节列表（如有）
- 资产信息：
  - 目标物体 USD 路径与 scale / mass
  - 桌子/地面等场景元素
- 控制方式：
  - `Pink IK` or `joint position` or `torque`
  - 是否需要 `IsaacTeleop`（hand tracking / controller）
- MDP 需求：
  - reset（初始姿态、物体随机化范围）
  - observations（关节、物体、末端、接触等）
  - terminations（成功/失败）
  - rewards（如需）

## 输出（必须生成）

- 一个新的 env_cfg 文件，包含：
  - `SceneCfg`：机器人、桌子、物体等
  - `ActionsCfg`：与 action layout 对齐（如果是 IK，明确 `PinkInverseKinematicsActionCfg` 的契约）
  - `ObservationsCfg` / `EventsCfg` / `TerminationsCfg`（必要项）
  - `EnvCfg.__post_init__`：必要的 IK/teleop 管线挂载（如需）
- 同时输出：
  - action layout（每一维语义、顺序、维度）
  - 需要用户确认/填补的列表（link/joint 名、USD 路径、URDF 路径等）

## 约束（必须遵守）

- 与仓库现有命名风格一致（参考 `isaaclab_tasks/manager_based/.../*env_cfg.py`）。
- joint/link 名必须来自资产（USD/URDF）；不允许凭空发明名称。
- 如果使用 `TensorReorderer`，必须显式固定顺序并与 `ActionsCfg` 对齐。
- 默认不开启训练奖励/课程，除非用户明确要 RL。

