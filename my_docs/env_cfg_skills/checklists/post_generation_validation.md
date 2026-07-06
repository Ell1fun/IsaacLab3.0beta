# Post Generation Validation Checklist

生成 env_cfg 和注册任务后，必须按以下清单检查。

## 1. 文件与 import

- 新 env_cfg 文件路径是否位于正确任务目录。
- 所有 import 是否能在当前仓库找到。
- `robot_cfg_import` 是否存在。
- task 注册文件是否正确指向新的 env_cfg 类。

## 2. 资产路径

- `object.preset` 是否来自 schema enum。
- `object.preset` 是否能映射到仓库已知 USD。
- robot USD/URDF 路径是否存在或来自 IsaacLab/Nucleus 已知路径。
- object USD 路径是否存在。
- table/scene USD 路径是否存在。
- DexPilot YAML / hand URDF 路径是否存在（如果使用 dexpilot）。

## 3. Link / Joint

- `base_link_name` 是否存在于 URDF/USD。
- 左右 `eef_link_names` 是否存在。
- `controlled_joint_names` 是否存在且顺序稳定。
- `hand_joint_names` 是否存在且数量与 action layout 一致。
- 不允许出现模型凭空发明的 joint/link。

## 4. Action Layout

- teleop pipeline 输出维度必须等于 `ActionsCfg` 期望维度。
- 左右 wrist pose 顺序必须和 `target_eef_link_names` 对齐。
- 手指 action 顺序必须和 `hand_joint_names` 对齐。
- 如果使用 `TensorReorderer`，输入/输出 names 必须显式列出。

## 5. Teleop / IK

- `world_T_anchor` 是否明确。
- wrist offset / idle wrist pose 是否明确。
- Pink IK 的 URDF path、mesh path、base link、frame task 是否完整。
- Nullspace joints 是否只包含受控且合理的关节。

## 6. MDP

- reset event 是否设置 robot/object/target 初始状态。
- observation 是否包含 robot、EEF、object、target 的必要信息。
- termination 是否包含 timeout 和 success。
- VR teleop 第一阶段可以没有 reward，但必须说明原因。

## 7. 运行入口

- Gym task id 是否唯一。
- 自动生成任务的 Gym ID 是否使用 `Isaac-Auto-PickPlace-...` 前缀。
- 自动生成 env_cfg 文件名是否使用 `auto_pickplace_..._env_cfg.py`。
- 自动生成 EnvCfg class 是否使用 `AutoPickPlace...EnvCfg`。
- 启动命令是否包含正确 task id。
- dry-run 时只输出会写入的文件，不实际写入。
- `--write` 写入前必须再次确认生成文件清单。
