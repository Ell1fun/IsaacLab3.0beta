from __future__ import annotations

from typing import Any


def template_pick_place_vr_spec() -> dict[str, Any]:
    """Return a starter spec template for quick iteration/debugging."""
    return {
        "spec_version": "0.1",
        "task": {
            "family": "manager_based/manipulation/pick_place",
            "variant": "unitree_g1_steering_wheel_teleop",
            "generated_file_stem": "auto_pickplace_unitree_g1_steering_wheel_teleop_env_cfg",
            "gym_id": "Isaac-Auto-PickPlace-UnitreeG1-SteeringWheel-Teleop-v0",
        },
        "robot": {
            "preset": "unitree_g1_inspire_ftp",
            "eef_link_names": {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"},
            "idle_wrist_pose": {
                "left": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
                "right": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
        },
        "scene": {
            "ground": {"kind": "ground_plane"},
            "table": {
                "usd_path": "",
                "pose": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
        },
        "object": {
            "preset": "steering_wheel",
            "usd_path": "${ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            "pose": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
        },
        "target": {"pose": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "control": {
            "mode": "pink_ik",
            "teleop": {
                "enabled": True,
                "input_source": "openxr_hand_tracking",
                "world_T_anchor": {"pos": [0.0, 0.0, 0.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
        },
    }
