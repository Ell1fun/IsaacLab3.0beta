# R11 PickPlace Hand Tracking 示例期望

- 示例输入是当前 `presets/robot_presets.yml` 与 `presets/object_presets.yml` enrichment 后的 spec 快照。
- 使用 `seres_r11_a2_high_pd` preset，EEF 为 `left_wrist_yaw_Link` / `right_wrist_yaw_Link`。
- 使用 `steering_wheel` object preset，USD、scale、single rigid body 均来自 object preset。
- 使用 `openxr_hand_tracking`，应命中 `PickPlaceR11EnvCfg` exact template。
- 预期生成 `auto_pickplace_r11_steering_wheel_hand_tracking_teleop_env_cfg.py` 与 `Isaac-Auto-PickPlace-R11-SteeringWheel-HandTracking-Teleop-v0`。
- `examples/` 不参与运行时生成流程；它只是人工阅读和未来回归测试的样例资产。
