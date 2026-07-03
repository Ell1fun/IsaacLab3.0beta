# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ISAACLAB_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../assets/seres_r11"))
R11_USD_PATH = os.path.join(ISAACLAB_ASSETS_DIR, "r11_0615.usd")

LEG_JOINT_NAMES = [
    ".*_hip_.*_Joint",
    ".*_knee_pitch_Joint",
    ".*_ankle_.*_Joint",
]
NO_LEG_JOINT_NAMES = [
    "torso_.*_Joint",
    ".*_shoulder_.*_Joint",
    ".*_elbow_pitch_Joint",
    ".*_wrist_.*_Joint",
]
ANKLE_JOINT_NAMES = [
    ".*_ankle_.*_Joint",
]
FEET_LINK_NAMES = [
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
]
WAIST_JOINT_NAMES = [
    "torso_.*_Joint",
]
ARM_JOINT_NAMES = [
    ".*_shoulder_.*_Joint",
    ".*_elbow_pitch_Joint",
    ".*_wrist_.*_Joint",
]
HEAD_JOINT_NAMES = [
    "head_.*_Joint",
]
LEFT_HAND_JOINT_NAMES = [
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_pinky_proximal_joint",
    "left_ring_proximal_joint",
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
]
RIGHT_HAND_JOINT_NAMES = [
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_pinky_proximal_joint",
    "right_ring_proximal_joint",
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
]
HAND_JOINT_NAMES = LEFT_HAND_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES
LEFT_HAND_MIMIC_JOINT_NAMES = [
    "left_index_distal_joint",
    "left_middle_distal_joint",
    "left_pinky_distal_joint",
    "left_ring_distal_joint",
    "left_thumb_distal_joint",
]
RIGHT_HAND_MIMIC_JOINT_NAMES = [
    "right_index_distal_joint",
    "right_middle_distal_joint",
    "right_pinky_distal_joint",
    "right_ring_distal_joint",
    "right_thumb_distal_joint",
]
HAND_ACTUATOR_JOINT_NAMES = HAND_JOINT_NAMES + LEFT_HAND_MIMIC_JOINT_NAMES + RIGHT_HAND_MIMIC_JOINT_NAMES
NO_HAND_JOINT_NAMES = [
    ".*_hip_.*_Joint",
    "torso_.*_Joint",
    ".*_knee_pitch_Joint",
    ".*_shoulder_.*_Joint",
    ".*_ankle_.*_Joint",
    ".*_elbow_pitch_Joint",
    ".*_wrist_.*_Joint",
]
RIGHT_HAND_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_Joint",
    "right_shoulder_roll_Joint",
    "right_shoulder_yaw_Joint",
    "right_elbow_pitch_Joint",
    "right_wrist_roll_Joint",
    "right_wrist_pitch_Joint",
    "right_wrist_yaw_Joint",
    *RIGHT_HAND_JOINT_NAMES,
]
LEFT_HAND_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_Joint",
    "left_shoulder_roll_Joint",
    "left_shoulder_yaw_Joint",
    "left_elbow_pitch_Joint",
    "left_wrist_roll_Joint",
    "left_wrist_pitch_Joint",
    "left_wrist_yaw_Joint",
    *LEFT_HAND_JOINT_NAMES,
]

DEFAULT_PELVIS_HEIGHT = 1.02  # 正常站直是1.05，根据首轮训练姿态反推

# 基础常数（严格对齐你的G1代码格式）
PI = 3.1415926535
DAMPING_RATIO = 2.0  # 论文设定的过阻尼比

# --------------------------
# 1. 谐波减速器（kgu系列）：自然频率 6Hz（抑制柔性谐振）
# --------------------------
NATURAL_FREQ_HARMONIC = 6 * 2.0 * PI  # 6Hz → 37.6991 rad/s

# 电枢值（ARMATURE = 关节端等效惯量Jref）
ARMATURE_KGU_08 = 0.18
ARMATURE_KGU_11 = 0.32
ARMATURE_KGU_14 = 0.19

# PD刚度KP（Stiffness）
STIFFNESS_KGU_08 = ARMATURE_KGU_08 * NATURAL_FREQ_HARMONIC**2 * 0.5
STIFFNESS_KGU_11 = ARMATURE_KGU_11 * NATURAL_FREQ_HARMONIC**2 * 0.5
STIFFNESS_KGU_14 = ARMATURE_KGU_14 * NATURAL_FREQ_HARMONIC**2 * 0.5

# PD阻尼KD（Damping）
DAMPING_KGU_08 = 2.0 * DAMPING_RATIO * ARMATURE_KGU_08 * NATURAL_FREQ_HARMONIC * 0.3
DAMPING_KGU_11 = 2.0 * DAMPING_RATIO * ARMATURE_KGU_11 * NATURAL_FREQ_HARMONIC * 0.3
DAMPING_KGU_14 = 2.0 * DAMPING_RATIO * ARMATURE_KGU_14 * NATURAL_FREQ_HARMONIC * 0.3

# EFFORT 力矩上限（和 saturation_effort 一致，用于actuator参数）
EFFORT_LIMIT_KGU08 = 18.56
EFFORT_LIMIT_KGU11 = 30.91
EFFORT_LIMIT_KGU14 = 45.74

# SATURATION 力矩饱和（saturation_effort）
SATURATION_EFFORT_KGU08 = EFFORT_LIMIT_KGU08
SATURATION_EFFORT_KGU11 = EFFORT_LIMIT_KGU11
SATURATION_EFFORT_KGU14 = EFFORT_LIMIT_KGU14

# VELOCITY 最大速度限制（velocity_limit）
VELOCITY_LIMIT_KGU08 = 5.70
VELOCITY_LIMIT_KGU11 = 6.48
VELOCITY_LIMIT_KGU14 = 7.13


# 关节摩擦力矩（关节端）
DYNAMIC_FRICTION_KGU_08 = 0.17
DYNAMIC_FRICTION_KGU_11 = 0.14
DYNAMIC_FRICTION_KGU_14 = 0.15

STATIC_FRICTION_KGU_08 = DYNAMIC_FRICTION_KGU_08 * 1.2
STATIC_FRICTION_KGU_11 = DYNAMIC_FRICTION_KGU_11 * 1.2
STATIC_FRICTION_KGU_14 = DYNAMIC_FRICTION_KGU_14 * 1.2

# -------------------_-------
# 2. 行星减速器（setz系列）：自然频率 10Hz（论文标准刚性配置）
# --------------------------
NATURAL_FREQ_PLANET = 10 * 2.0 * PI  # 10Hz → 62.8319 rad/s

# 电枢值（ARMATURE = 转子惯量×减速比² 转换后的关节端等效惯量）
ARMATURE_SETZ70 = 0.013312
ARMATURE_SETZ90 = 0.04851

# PD刚度KP（Stiffness）
STIFFNESS_SETZ70 = ARMATURE_SETZ70 * NATURAL_FREQ_PLANET**2
STIFFNESS_SETZ90 = ARMATURE_SETZ90 * NATURAL_FREQ_PLANET**2

# PD阻尼KD（Damping）减小阻尼
DAMPING_SETZ70 = 2.0 * DAMPING_RATIO * ARMATURE_SETZ70 * NATURAL_FREQ_PLANET * 0.5
DAMPING_SETZ90 = 2.0 * DAMPING_RATIO * ARMATURE_SETZ90 * NATURAL_FREQ_PLANET * 0.5

# 关节摩擦力矩（关节端，补充业内常见值）
DYNAMIC_FRICTION_SETZ70 = 0.35
DYNAMIC_FRICTION_SETZ90 = 0.55

STATIC_FRICTION_SETZ70 = DYNAMIC_FRICTION_SETZ70 * 1.2
STATIC_FRICTION_SETZ90 = DYNAMIC_FRICTION_SETZ90 * 1.2


# EFFORT 力矩上限（和 saturation_effort 一致，用于actuator参数）
EFFORT_LIMIT_SETZ70 = 85  # 额定12.0
EFFORT_LIMIT_SETZ90 = 150  # 额定31.5

# SATURATION 力矩饱和（saturation_effort）
SATURATION_EFFORT_SETZ70 = EFFORT_LIMIT_SETZ70
SATURATION_EFFORT_SETZ90 = EFFORT_LIMIT_SETZ90

# VELOCITY 最大速度限制（velocity_limit）
VELOCITY_LIMIT_SETZ70 = 25.13  # 对应 240 RPM
VELOCITY_LIMIT_SETZ90 = 13.09  # 对应 125 RPM


R11_A2_32DOF_REAL = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=R11_USD_PATH,
        # Disable hands to accelerate training.
        variants={"Physics": "PhysX"},
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.15),
        joint_pos={
            "left_hip_pitch_Joint": 0.16,
            "left_knee_pitch_Joint": -0.30,
            "left_ankle_pitch_Joint": -0.20,
            "right_hip_pitch_Joint": -0.16,
            "right_knee_pitch_Joint": 0.30,
            "right_ankle_pitch_Joint": -0.20,
            "left_elbow_pitch_Joint": -1.047,
            "right_elbow_pitch_Joint": 1.047,
            "left_shoulder_roll_Joint": -0.21,
            "right_shoulder_roll_Joint": 0.21,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_Joint",
                ".*_hip_roll_Joint",
                ".*_hip_pitch_Joint",
                ".*_knee_pitch_Joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_Joint": EFFORT_LIMIT_SETZ90,
                ".*_hip_roll_Joint": EFFORT_LIMIT_SETZ90,
                ".*_hip_pitch_Joint": EFFORT_LIMIT_SETZ90,
                ".*_knee_pitch_Joint": EFFORT_LIMIT_SETZ90,
            },
            velocity_limit_sim={
                ".*_hip_yaw_Joint": VELOCITY_LIMIT_SETZ90,
                ".*_hip_roll_Joint": VELOCITY_LIMIT_SETZ90,
                ".*_hip_pitch_Joint": VELOCITY_LIMIT_SETZ90,
                ".*_knee_pitch_Joint": VELOCITY_LIMIT_SETZ90,
            },
            stiffness={
                ".*_hip_pitch_Joint": STIFFNESS_SETZ90,
                ".*_hip_roll_Joint": STIFFNESS_SETZ90,
                ".*_hip_yaw_Joint": STIFFNESS_SETZ90,
                ".*_knee_pitch_Joint": STIFFNESS_SETZ90,
            },
            damping={
                ".*_hip_pitch_Joint": DAMPING_SETZ90,
                ".*_hip_roll_Joint": DAMPING_SETZ90,
                ".*_hip_yaw_Joint": DAMPING_SETZ90,
                ".*_knee_pitch_Joint": DAMPING_SETZ90,
            },
            armature={
                ".*_hip_pitch_Joint": ARMATURE_SETZ90,
                ".*_hip_roll_Joint": ARMATURE_SETZ90,
                ".*_hip_yaw_Joint": ARMATURE_SETZ90,
                ".*_knee_pitch_Joint": ARMATURE_SETZ90,
            },
            friction={
                ".*_hip_yaw_Joint": DYNAMIC_FRICTION_SETZ90,
                ".*_hip_roll_Joint": DYNAMIC_FRICTION_SETZ90,
                ".*_hip_pitch_Joint": DYNAMIC_FRICTION_SETZ90,
                ".*_knee_pitch_Joint": DYNAMIC_FRICTION_SETZ90,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=2.0 * EFFORT_LIMIT_SETZ70,
            velocity_limit_sim=VELOCITY_LIMIT_SETZ70,
            joint_names_expr=ANKLE_JOINT_NAMES,
            stiffness=2.0 * STIFFNESS_SETZ70,
            damping=2.0 * DAMPING_SETZ70,
            armature=2.0 * ARMATURE_SETZ70,
            friction=2.0 * DYNAMIC_FRICTION_SETZ70,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=WAIST_JOINT_NAMES,
            effort_limit_sim={
                "torso_yaw_Joint": EFFORT_LIMIT_SETZ70,
                "torso_roll_Joint": 2.0 * EFFORT_LIMIT_KGU14,
                "torso_pitch_Joint": 2.0 * EFFORT_LIMIT_KGU14,
            },
            velocity_limit_sim={
                "torso_yaw_Joint": VELOCITY_LIMIT_SETZ70,
                "torso_roll_Joint": VELOCITY_LIMIT_KGU14,
                "torso_pitch_Joint": VELOCITY_LIMIT_KGU14,
            },
            stiffness={
                "torso_yaw_Joint": STIFFNESS_SETZ70,
                "torso_roll_Joint": 2.0 * STIFFNESS_KGU_14,
                "torso_pitch_Joint": 2.0 * STIFFNESS_KGU_14,
            },
            damping={
                "torso_yaw_Joint": DAMPING_SETZ70,
                "torso_roll_Joint": 2.0 * DAMPING_KGU_14,
                "torso_pitch_Joint": 2.0 * DAMPING_KGU_14,
            },
            armature={
                "torso_yaw_Joint": ARMATURE_SETZ70,
                "torso_roll_Joint": 2.0 * ARMATURE_KGU_14,
                "torso_pitch_Joint": 2.0 * ARMATURE_KGU_14,
            },
            friction={
                "torso_yaw_Joint": DYNAMIC_FRICTION_SETZ70,
                "torso_roll_Joint": 2.0 * DYNAMIC_FRICTION_KGU_14,
                "torso_pitch_Joint": 2.0 * DYNAMIC_FRICTION_KGU_14,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=HEAD_JOINT_NAMES,
            effort_limit_sim={
                "head_roll_Joint": EFFORT_LIMIT_KGU08,
                "head_pitch_Joint": EFFORT_LIMIT_KGU08,
                "head_yaw_Joint": EFFORT_LIMIT_KGU08,
            },
            velocity_limit_sim={
                "head_roll_Joint": VELOCITY_LIMIT_KGU08,
                "head_pitch_Joint": VELOCITY_LIMIT_KGU08,
                "head_yaw_Joint": VELOCITY_LIMIT_KGU08,
            },
            stiffness={
                "head_roll_Joint": STIFFNESS_KGU_08,
                "head_pitch_Joint": STIFFNESS_KGU_08,
                "head_yaw_Joint": STIFFNESS_KGU_08,
            },
            damping={
                "head_roll_Joint": DAMPING_KGU_08,
                "head_pitch_Joint": DAMPING_KGU_08,
                "head_yaw_Joint": DAMPING_KGU_08,
            },
            armature=ARMATURE_KGU_08,
            friction=DYNAMIC_FRICTION_KGU_08,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim={
                ".*_shoulder_pitch_Joint": EFFORT_LIMIT_KGU14,
                ".*_shoulder_roll_Joint": EFFORT_LIMIT_KGU14,
                ".*_shoulder_yaw_Joint": EFFORT_LIMIT_KGU14,
                ".*_elbow_pitch_Joint": EFFORT_LIMIT_KGU11,
                ".*_wrist_roll_Joint": EFFORT_LIMIT_KGU08,
                ".*_wrist_pitch_Joint": EFFORT_LIMIT_KGU08,
                ".*_wrist_yaw_Joint": EFFORT_LIMIT_KGU08,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_Joint": VELOCITY_LIMIT_KGU14,
                ".*_shoulder_roll_Joint": VELOCITY_LIMIT_KGU14,
                ".*_shoulder_yaw_Joint": VELOCITY_LIMIT_KGU14,
                ".*_elbow_pitch_Joint": VELOCITY_LIMIT_KGU11,
                ".*_wrist_roll_Joint": VELOCITY_LIMIT_KGU08,
                ".*_wrist_pitch_Joint": VELOCITY_LIMIT_KGU08,
                ".*_wrist_yaw_Joint": VELOCITY_LIMIT_KGU08,
            },
            stiffness={
                ".*_shoulder_pitch_Joint": STIFFNESS_KGU_14,
                ".*_shoulder_roll_Joint": STIFFNESS_KGU_14,
                ".*_shoulder_yaw_Joint": STIFFNESS_KGU_14,
                ".*_elbow_pitch_Joint": STIFFNESS_KGU_11,
                ".*_wrist_.*_Joint": STIFFNESS_KGU_08,
            },
            damping={
                ".*_shoulder_pitch_Joint": DAMPING_KGU_14,
                ".*_shoulder_roll_Joint": DAMPING_KGU_14,
                ".*_shoulder_yaw_Joint": DAMPING_KGU_14,
                ".*_elbow_pitch_Joint": DAMPING_KGU_11,
                ".*_wrist_.*_Joint": DAMPING_KGU_08,
            },
            armature={
                ".*_shoulder_.*": ARMATURE_KGU_14,
                ".*_elbow_.*": ARMATURE_KGU_11,
                ".*_wrist_.*_Joint": ARMATURE_KGU_08,
            },
            friction={
                ".*_shoulder_.*": DYNAMIC_FRICTION_KGU_14,
                ".*_elbow_.*": DYNAMIC_FRICTION_KGU_11,
                ".*_wrist_.*_Joint": DYNAMIC_FRICTION_KGU_08,
            },
        ),
        # Finger driving must match the working G1 Inspire setup. In the USD the
        # proximal joints carry a PhysX drive while the distal joints are PhysX
        # *mimic* joints (``physxMimicJoint:*`` attrs, gearing -1.155 for the
        # fingers and -1.0 for the thumb, naturalFrequency 25, dampingRatio 0.005)
        # coupled to their proximal joint. That mimic spring is far too weak to
        # hold the ~1e-7 kg*m^2 distal links against any load, so on its own the
        # distal joints just swing freely. Putting every finger joint (proximal +
        # distal) into one actuator makes Isaac Lab create a real position drive on
        # the distal joints too; the per-step targets from the dex-retargeting mimic
        # adaptor agree with the gearing, so drive and mimic constraint reinforce.
        # ``armature`` is required: without it the tiny-inertia distal joints are
        # numerically unstable under an implicit PD drive.
        "hands": ImplicitActuatorCfg(
            joint_names_expr=HAND_ACTUATOR_JOINT_NAMES,
            effort_limit_sim=2000,
            velocity_limit_sim=87,
            stiffness=17000,
            damping=500,
            armature=0.001,
        ),
    },
)

R11_A2_HIGH_PD_CFG = R11_A2_32DOF_REAL.copy()
R11_A2_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
# The USD's baked world-weld joint (`/Robot/joints/FixedJoint`) is a SIBLING of the
# articulation-root prim (`/Robot/pelvis`), so PhysX does not treat it as the
# articulation's fixed base: the root stays floating and PhysX ignores the reset
# root-pose write, snapping pelvis to the joint anchor (origin+identity) on the first
# physics step. Setting fix_root_link=True makes Isaac Lab create a *proper* fixed
# joint and move ArticulationRootAPI to the parent, welding the base at its spawned
# init_state pose so it no longer jumps.
R11_A2_HIGH_PD_CFG.spawn.articulation_props.fix_root_link = True
R11_A2_HIGH_PD_CFG.actuators["waist"].stiffness = 4400.0
R11_A2_HIGH_PD_CFG.actuators["waist"].damping = 40.0
R11_A2_HIGH_PD_CFG.actuators["waist"].armature = 0.01
R11_A2_HIGH_PD_CFG.actuators["arms"].stiffness = 4400.0
R11_A2_HIGH_PD_CFG.actuators["arms"].damping = 40.0
R11_A2_HIGH_PD_CFG.actuators["arms"].armature = 0.01
"""Configuration for the R11 robot with stiffer upper-body PD control for task-space teleoperation."""

R11_ACTION_SCALE_LOWER = {}
for actuator_name, actuator_cfg in R11_A2_32DOF_REAL.actuators.items():
    if actuator_name != "legs" and actuator_name != "feet":
        continue
    e = actuator_cfg.effort_limit_sim
    s = actuator_cfg.stiffness
    names = actuator_cfg.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            R11_ACTION_SCALE_LOWER[n] = 0.25 * e[n] / s[n]
