# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-PickPlace-GR1T2-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_gr1t2_env_cfg:PickPlaceGR1T2EnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-NutPour-GR1T2-Pink-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nutpour_gr1t2_pink_ik_env_cfg:NutPourGR1T2PinkIKEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_nut_pouring.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exhaustpipe_gr1t2_pink_ik_env_cfg:ExhaustPipeGR1T2PinkIKEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_exhaust_pipe.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_gr1t2_waist_enabled_env_cfg:PickPlaceGR1T2WaistEnabledEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-R11-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_seres_r11_env_cfg:PickPlaceR11EnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-R11-MotionController-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_seres_r11_tri_env_cfg:PickPlaceR11TriEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-G1-InspireFTP-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_unitree_g1_inspire_hand_env_cfg:PickPlaceG1InspireFTPEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

# Auto-generated task registration. Keep the Auto prefix to distinguish it from hand-written tasks.
gym.register(
    id="Isaac-Auto-PickPlace-UnitreeG1-BottledWaterC01-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.auto_pickplace_unitree_g1_bottled_water_c01_teleop_env_cfg:AutoPickPlaceUnitreeG1BottledWaterC01TeleopEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

# Auto-generated task registration. Keep the Auto prefix to distinguish it from hand-written tasks.
gym.register(
    id="Isaac-Auto-PickPlace-UnitreeG1-BottledWaterC01-MotionController-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.auto_pickplace_unitree_g1_bottled_water_c01_motion_controller_teleop_env_cfg:AutoPickPlaceUnitreeG1BottledWaterC01MotionControllerTeleopEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

# Auto-generated task registration. Keep the Auto prefix to distinguish it from hand-written tasks.
gym.register(
    id="Isaac-Auto-PickPlace-UnitreeG1-Mug-MotionController-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.auto_pickplace_unitree_g1_mug_motion_controller_teleop_env_cfg:AutoPickPlaceUnitreeG1MugMotionControllerTeleopEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

# Auto-generated task registration. Keep the Auto prefix to distinguish it from hand-written tasks.
gym.register(
    id="Isaac-Auto-PickPlace-R11-Mug-MotionController-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.auto_pickplace_r11_mug_motion_controller_teleop_env_cfg:AutoPickPlaceR11MugMotionControllerTeleopEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)
