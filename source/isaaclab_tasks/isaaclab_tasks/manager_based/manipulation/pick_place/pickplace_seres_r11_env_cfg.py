# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile
from collections.abc import Callable

import numpy as np
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrCfg
from scipy.spatial.transform import Rotation

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import (
    DampingTaskCfg,
    FrameTaskCfg,
    NullSpacePostureTaskCfg,
    PinkIKControllerCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import schemas
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.sim.utils.queries import get_all_matching_child_prims
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, check_file_path, retrieve_file_path
from isaaclab.utils.configclass import configclass

from . import mdp

from isaaclab_assets.robots.seres_r11 import (  # isort: skip
    HAND_JOINT_NAMES,
    R11_A2_HIGH_PD_CFG,
)


ISAACLAB_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../assets/seres_r11"))

R11_HAND_JOINT_NAMES = HAND_JOINT_NAMES
R11_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_Joint",
    "left_shoulder_roll_Joint",
    "left_shoulder_yaw_Joint",
    "left_elbow_pitch_Joint",
    "left_wrist_roll_Joint",
    "left_wrist_pitch_Joint",
    "left_wrist_yaw_Joint",
    "right_shoulder_pitch_Joint",
    "right_shoulder_roll_Joint",
    "right_shoulder_yaw_Joint",
    "right_elbow_pitch_Joint",
    "right_wrist_roll_Joint",
    "right_wrist_pitch_Joint",
    "right_wrist_yaw_Joint",
]
R11_WAIST_JOINT_NAMES = ["torso_yaw_Joint", "torso_roll_Joint", "torso_pitch_Joint"]
R11_NULLSPACE_JOINT_NAMES = [
    # Nullspace（零空间）关节列表：当 IK 的主要任务（末端位姿）存在冗余解时，
    # Pink IK 会在不破坏主要任务的前提下，利用这些关节去优化“次要目标”（secondary objective），
    # 例如保持更自然的上半身姿态、远离关节限位、减少怪异扭转。
    #
    # 这里选择肩/肘 + 腰作为 nullspace 的可调关节，让躯干和大关节承担姿态调整；
    # 手腕（wrist_*）通常更直接影响 EEF 朝向，放入 nullspace 可能导致末端姿态更“飘”，因此不纳入。
    "left_shoulder_pitch_Joint",
    "left_shoulder_roll_Joint",
    "left_shoulder_yaw_Joint",
    "left_elbow_pitch_Joint",
    "right_shoulder_pitch_Joint",
    "right_shoulder_roll_Joint",
    "right_shoulder_yaw_Joint",
    "right_elbow_pitch_Joint",
    *R11_WAIST_JOINT_NAMES,
]

R11_FIXED_JOINT_POS = {
    # lower body: held by high-PD actuators and kept out of the teleop action space
    "left_hip_yaw_Joint": 0.0,
    "left_hip_roll_Joint": 0.0,
    "left_hip_pitch_Joint": 0.16,
    "left_knee_pitch_Joint": -0.30,
    "left_ankle_pitch_Joint": -0.20,
    "left_ankle_roll_Joint": 0.0,
    "right_hip_yaw_Joint": 0.0,
    "right_hip_roll_Joint": 0.0,
    "right_hip_pitch_Joint": -0.16,
    "right_knee_pitch_Joint": 0.30,
    "right_ankle_pitch_Joint": -0.20,
    "right_ankle_roll_Joint": 0.0,
    # head: fixed for this upper-body pick/place teleop task
    "head_roll_Joint": 0.0,
    "head_pitch_Joint": 0.0,
    "head_yaw_Joint": 0.0,
    # Revo2 distal joints are mimic joints in the hand URDF. They are excluded
    # from teleop actions and initialized here if they are exported as DOFs.
    ".*_distal_joint": 0.0,
}
R11_UPPER_BODY_INITIAL_JOINT_POS = {
    "torso_yaw_Joint": 0.0,
    "torso_roll_Joint": 0.0,
    "torso_pitch_Joint": 0.0,
    "left_shoulder_pitch_Joint": 0.0,
    "left_shoulder_roll_Joint": -0.21,
    "left_shoulder_yaw_Joint": 0.0,
    "left_elbow_pitch_Joint": 0.0,
    "left_wrist_roll_Joint": 0.0,
    "left_wrist_pitch_Joint": 0.0,
    "left_wrist_yaw_Joint": 0.0,
    "right_shoulder_pitch_Joint": 0.0,
    "right_shoulder_roll_Joint": 0.21,
    "right_shoulder_yaw_Joint": 0.0,
    "right_elbow_pitch_Joint": 0.0,
    "right_wrist_roll_Joint": 0.0,
    "right_wrist_pitch_Joint": 0.0,
    "right_wrist_yaw_Joint": 0.0,
    **dict.fromkeys(R11_HAND_JOINT_NAMES, 0.0),
}

LEFT_EEF_LINK_NAME = "left_wrist_yaw_Link"
RIGHT_EEF_LINK_NAME = "right_wrist_yaw_Link"
R11_LEFT_IDLE_WRIST_POSE = (-0.249205, 0.204645, 1.156897, -0.074109, 0.074109, -0.703206, 0.703206)
R11_RIGHT_IDLE_WRIST_POSE = (0.249213, 0.204636, 1.156897, 0.074109, -0.074109, -0.703206, 0.703206)

# Motion controller 版本使用这两组手部姿态做抓握插值：
# trigger=0 时使用 open pose，trigger=1 时使用 closed pose，中间值线性插值。
# 这里只配置会主动控制的手指关节，未配置的关节在 retargeter 中默认保持 0.0。
# 注意：这不是 OpenXR 手追踪（hand tracking）的逐关节角度映射；它是“没有手指关节追踪时”的简化方案，
# 用一个 grasp 标量（trigger）把整只手从 OPEN 插值到 CLOSED。手追踪版本使用 DexHandRetargeter（见 _build_r11_pickplace_pipeline）。
R11_HAND_OPEN_POSE = {
    "left_index_proximal_joint": 0.0,
    "left_middle_proximal_joint": 0.0,
    "left_pinky_proximal_joint": 0.0,
    "left_ring_proximal_joint": 0.0,
    "left_thumb_metacarpal_joint": 0.9,
    "left_thumb_proximal_joint": 0.0,
    "right_index_proximal_joint": 0.0,
    "right_middle_proximal_joint": 0.0,
    "right_pinky_proximal_joint": 0.0,
    "right_ring_proximal_joint": 0.0,
    "right_thumb_metacarpal_joint": 0.9,
    "right_thumb_proximal_joint": 0.0,
}
R11_HAND_CLOSED_POSE = {
    "left_index_proximal_joint": 1.2,
    "left_middle_proximal_joint": 1.2,
    "left_pinky_proximal_joint": 1.2,
    "left_ring_proximal_joint": 1.2,
    "left_thumb_metacarpal_joint": 0.95,
    "left_thumb_proximal_joint": 1.03,
    "right_index_proximal_joint": 1.2,
    "right_middle_proximal_joint": 1.2,
    "right_pinky_proximal_joint": 1.2,
    "right_ring_proximal_joint": 1.2,
    "right_thumb_metacarpal_joint": 0.95,
    "right_thumb_proximal_joint": 1.03,
}


class R11WristPoseRebaser:
    """Rebase absolute Pico wrist poses around R11's idle wrist pose."""

    def __init__(self, idle_pose: tuple[float, ...], name: str):
        from isaacteleop.retargeting_engine.interface import BaseRetargeter

        class _Retargeter(BaseRetargeter):
            def __init__(self, idle_pose: tuple[float, ...], name: str):
                super().__init__(name=name)
                self._idle_pose = np.asarray(idle_pose, dtype=np.float32)
                self._idle_rot = Rotation.from_quat(self._idle_pose[3:7])
                self._start_pose: np.ndarray | None = None
                self._start_rot: Rotation | None = None

            def input_spec(self):
                from isaacteleop.retargeting_engine.interface.tensor_group_type import TensorGroupType
                from isaacteleop.retargeting_engine.tensor_types import DLDataType, NDArrayType

                return {
                    "ee_pose": TensorGroupType(
                        "ee_pose",
                        [NDArrayType("pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32)],
                    )
                }

            def output_spec(self):
                from isaacteleop.retargeting_engine.interface.tensor_group_type import TensorGroupType
                from isaacteleop.retargeting_engine.tensor_types import DLDataType, NDArrayType

                return {
                    "ee_pose": TensorGroupType(
                        "ee_pose",
                        [NDArrayType("pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32)],
                    )
                }

            def _compute_fn(self, inputs, outputs, context) -> None:
                raw_pose = np.asarray(inputs["ee_pose"][0], dtype=np.float32)
                is_running = getattr(getattr(context, "execution_events", None), "execution_state", None)
                is_running = getattr(is_running, "name", str(is_running)) == "RUNNING"

                if getattr(getattr(context, "execution_events", None), "reset", False) or not is_running:
                    self._start_pose = None
                    self._start_rot = None
                    outputs["ee_pose"][0] = self._idle_pose
                    return

                if self._start_pose is None:
                    self._start_pose = raw_pose.copy()
                    self._start_rot = Rotation.from_quat(raw_pose[3:7])
                    outputs["ee_pose"][0] = self._idle_pose
                    return

                assert self._start_rot is not None
                delta_pos = raw_pose[:3] - self._start_pose[:3]
                delta_rot = Rotation.from_quat(raw_pose[3:7]) * self._start_rot.inv()
                target_pose = self._idle_pose.copy()
                target_pose[:3] = self._idle_pose[:3] + delta_pos
                target_pose[3:7] = (delta_rot * self._idle_rot).as_quat().astype(np.float32)
                outputs["ee_pose"][0] = target_pose

        self._retargeter = _Retargeter(idle_pose, name)

    def __getattr__(self, name: str):
        return getattr(self._retargeter, name)


class R11ControllerTriggerGraspRetargeter:
    """Map a motion-controller trigger value to R11 open/closed hand poses."""

    def __init__(
        self,
        hand_joint_names: list[str],
        name: str,
        open_pose: dict[str, float],
        closed_pose: dict[str, float],
    ):
        from isaacteleop.retargeting_engine.interface import BaseRetargeter

        class _Retargeter(BaseRetargeter):
            def __init__(
                self,
                hand_joint_names: list[str],
                name: str,
                open_pose: dict[str, float],
                closed_pose: dict[str, float],
            ):
                self._hand_joint_names = hand_joint_names
                self._open_pose = open_pose
                self._closed_pose = closed_pose
                super().__init__(name=name)

            def input_spec(self):
                from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource

                # ControllersSource 输出左右手柄的 grip/aim pose、摇杆、按钮、trigger 等信息。
                # 这里的 retargeter 只需要单侧 controller 数据，左右手复用同一个 input spec。
                return {"controller": ControllersSource(name="controller_spec").output_spec()[ControllersSource.LEFT]}

            def output_spec(self):
                from isaacteleop.retargeting_engine.interface.tensor_group_type import TensorGroupType
                from isaacteleop.retargeting_engine.tensor_types import FloatType

                return {
                    "hand_joints": TensorGroupType(
                        "hand_joints",
                        [FloatType(joint_name) for joint_name in self._hand_joint_names],
                    )
                }

            def _compute_fn(self, inputs, outputs, context) -> None:
                controller = inputs["controller"]
                trigger = 0.0
                if not getattr(controller, "is_none", False):
                    # ControllersSource 的第 13 个 tensor 是 controller_trigger_value，范围通常是 [0, 1]。
                    trigger = float(controller[13])
                trigger = float(np.clip(trigger, 0.0, 1.0))

                for index, joint_name in enumerate(self._hand_joint_names):
                    # 对每个关节做 open -> closed 的线性插值，实现“按下扳机逐渐闭合”。
                    open_value = self._open_pose.get(joint_name, 0.0)
                    closed_value = self._closed_pose.get(joint_name, open_value)
                    outputs["hand_joints"][index] = open_value + trigger * (closed_value - open_value)

        self._retargeter = _Retargeter(hand_joint_names, name, open_pose, closed_pose)

    def __getattr__(self, name: str):
        return getattr(self._retargeter, name)


def _build_r11_pickplace_pipeline():
    """Build an IsaacTeleop retargeting pipeline for R11 pick-place teleoperation.

    Creates two Se3AbsRetargeters for left and right wrist pose tracking and
    two DexHandRetargeters for left and right Revo2 hand finger control from
    hand tracking data. All outputs are flattened into a single action tensor
    via TensorReorderer.
    """
    # 构建“手追踪（hand tracking）版本”的遥操管线：
    # OpenXR 手追踪 ->（坐标系对齐/偏置）-> 左右腕目标 pose(7) ->（rebase 到 idle）-> 最终腕目标
    # OpenXR 手追踪 -> DexHandRetargeter -> 手指关节目标
    # 最后把多个输出拼成一个一维 action，顺序必须与 ActionsCfg.upper_body_ik 的 action space 完全一致。

    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # 1) OpenXR 手追踪输入源（左右手）
    hands = HandsSource(name="hands")

    # 2) XR 的 tracking space 通常以“anchor”为参考系，这里通过 world_T_anchor 把输入变换到仿真 world 坐标系
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # GR1T2 works with left identity and right yaw-180 SE3 offsets because its
    # wrist task frames are aligned with that OpenXR convention. R11's USD
    # wrist_yaw_Link frames are rotated relative to GR1T2's hand_roll_link
    # frames, so compose the GR1T2 offsets with the measured neutral-frame
    # rotation: R_gr1_task.T @ R_r11_task.
    # 3) 左手：把 OpenXR 的手腕 pose 变成“左腕 EEF 目标 pose(7)”
    # target_offset_* 是对齐 OpenXR 约定与 R11 wrist_yaw_Link task frame 的固定补偿（单位：度）
    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=90.0,
            target_offset_pitch=77.968,
            target_offset_yaw=90.0,
        ),
        name="left_ee_pose",
    )
    connected_left_se3 = left_se3.connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)})

    # 4) Rebase：把“绝对 wrist pose”重定位到机器人预设 idle wrist pose 附近
    # 直觉：以操作者的起始姿态为零点，用 delta 叠加到机器人 idle_pose 上，避免一启动就跳到奇怪姿态
    left_pose_rebaser = R11WristPoseRebaser(R11_LEFT_IDLE_WRIST_POSE, name="left_ee_pose_rebased")
    connected_left_pose = left_pose_rebaser.connect({"ee_pose": connected_left_se3.output("ee_pose")})

    # 5) 右手：逻辑与左手一致（target_offset_pitch 为负是因为左右手坐标系/骨架约定不同）
    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=90.0,
            target_offset_pitch=-77.968,
            target_offset_yaw=90.0,
        ),
        name="right_ee_pose",
    )
    connected_right_se3 = right_se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})
    right_pose_rebaser = R11WristPoseRebaser(R11_RIGHT_IDLE_WRIST_POSE, name="right_ee_pose_rebased")
    connected_right_pose = right_pose_rebaser.connect({"ee_pose": connected_right_se3.output("ee_pose")})

    # 6) 手指 retarget 配置（DexPilot）
    left_yaml_path = os.path.join(ISAACLAB_ASSETS_DIR, "revo2_left_hand_dexpilot.yml")
    right_yaml_path = os.path.join(ISAACLAB_ASSETS_DIR, "revo2_right_hand_dexpilot.yml")
    left_urdf_path = os.path.join(ISAACLAB_ASSETS_DIR, "revo2_left_hand.urdf")
    right_urdf_path = os.path.join(ISAACLAB_ASSETS_DIR, "revo2_right_hand.urdf")

    # OpenXR hand tracking 坐标系到 Dex/URDF 约定坐标系的变换（3x3，按行展开）
    operator2mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)

    # 7) 每只手输出的关节名列表（每手 11 DOF）。
    # 这些名字必须和 URDF/DexPilot 配置一致，同时也决定后面 TensorReorderer 的输入维度与顺序。
    left_hand_joint_names = [
        "left_index_proximal_joint",
        "left_middle_proximal_joint",
        "left_pinky_proximal_joint",
        "left_ring_proximal_joint",
        "left_thumb_metacarpal_joint",
        "left_thumb_distal_joint",
        "left_index_distal_joint",
        "left_middle_distal_joint",
        "left_pinky_distal_joint",
        "left_ring_distal_joint",
        "left_thumb_proximal_joint",
    ]

    right_hand_joint_names = [
        "right_index_proximal_joint",
        "right_middle_proximal_joint",
        "right_pinky_proximal_joint",
        "right_ring_proximal_joint",
        "right_thumb_metacarpal_joint",
        "right_thumb_distal_joint",
        "right_index_distal_joint",
        "right_middle_distal_joint",
        "right_pinky_distal_joint",
        "right_ring_distal_joint",
        "right_thumb_proximal_joint",
    ]

    # 8) DexHandRetargeter：把 OpenXR 的手追踪数据映射到机器人手指关节目标
    left_dex_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=left_yaml_path,
        hand_urdf=left_urdf_path,
        hand_joint_names=left_hand_joint_names,
        hand_side="left",
        handtracking_to_baselink_frame_transform=operator2mano,
    )
    left_dex = DexHandRetargeter(left_dex_cfg, name="left_hand")
    connected_left_dex = left_dex.connect(
        {
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
        }
    )

    right_dex_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=right_yaml_path,
        hand_urdf=right_urdf_path,
        hand_joint_names=right_hand_joint_names,
        hand_side="right",
        handtracking_to_baselink_frame_transform=operator2mano,
    )
    right_dex = DexHandRetargeter(right_dex_cfg, name="right_hand")
    connected_right_dex = right_dex.connect(
        {
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # 9) 组装最终 action 的元素名与顺序（用于 TensorReorderer 重排）
    # 左右腕各 7 维：pos(3) + quat(4)，四元数顺序是 (qx,qy,qz,qw)
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]
    output_order = (
        left_ee_elements
        + right_ee_elements
        + [
            # hand_joint_names indices 0-4 (left proximal + thumb yaw)
            "left_index_proximal_joint",
            "left_middle_proximal_joint",
            "left_pinky_proximal_joint",
            "left_ring_proximal_joint",
            "left_thumb_metacarpal_joint",
            # hand_joint_names indices 5-9 (right proximal + thumb yaw)
            "right_index_proximal_joint",
            "right_middle_proximal_joint",
            "right_pinky_proximal_joint",
            "right_ring_proximal_joint",
            "right_thumb_metacarpal_joint",
            # hand_joint_names indices 10-14 (left distal + thumb pitch)
            "left_index_distal_joint",
            "left_middle_distal_joint",
            "left_pinky_distal_joint",
            "left_ring_distal_joint",
            "left_thumb_proximal_joint",
            # hand_joint_names indices 15-19 (right distal + thumb pitch)
            "right_index_distal_joint",
            "right_middle_distal_joint",
            "right_pinky_distal_joint",
            "right_ring_distal_joint",
            "right_thumb_proximal_joint",
            # hand_joint_names indices 20-22 (thumb distal)
            "left_thumb_distal_joint",
            "right_thumb_distal_joint",
        ]
    )

    # 10) TensorReorderer：把“左腕 pose / 右腕 pose / 左手关节 / 右手关节”合并成一个一维 action，并重排到 output_order 指定的顺序
    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_joint_names,
            "right_hand_joints": right_hand_joint_names,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_pose.output("ee_pose"),
            "right_ee_pose": connected_right_pose.output("ee_pose"),
            "left_hand_joints": connected_left_dex.output("hand_joints"),
            "right_hand_joints": connected_right_dex.output("hand_joints"),
        }
    )

    # 11) 输出合并：对外暴露一个名为 "action" 的输出，供 env/action manager 读取
    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, [left_dex, right_dex]


def _build_r11_pickplace_motion_controller_pipeline():
    """Build an IsaacTeleop pipeline that drives R11 with Pico motion controllers.

    The left/right controller grip poses drive the left/right wrists. Each
    controller trigger linearly closes the four non-thumb fingers on that side.
    """

    from isaacteleop.retargeters import (
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # 1. 创建 motion controller 数据源。
    # ControllersSource 会从 OpenXR/CloudXR 中读取左右 PICO 手柄的 pose、trigger、按钮等数据。
    controllers = ControllersSource(name="controllers")

    # 2. IsaacTeleopDevice 每帧会提供 world_T_anchor，用于把 XR 空间坐标变换到仿真世界坐标。
    # 如果不做这一步，手柄 pose 仍在 XR anchor 坐标系下，不能直接作为机器人末端目标。
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # 3. 左手柄 grip pose -> 左腕末端目标 pose。
    # use_wrist_rotation/use_wrist_position=False 表示从 controller grip pose 读取，而不是从手腕骨骼读取。
    # target_offset_* 用来补偿 PICO 手柄坐标系和 R11 wrist_yaw_Link 坐标系之间的姿态差。
    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=90.0,
            target_offset_pitch=77.968,
            target_offset_yaw=90.0,
        ),
        name="left_ee_pose",
    )
    connected_left_se3 = left_se3.connect({ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)})
    left_pose_rebaser = R11WristPoseRebaser(R11_LEFT_IDLE_WRIST_POSE, name="left_ee_pose_rebased")
    connected_left_pose = left_pose_rebaser.connect({"ee_pose": connected_left_se3.output("ee_pose")})

    # 4. 右手柄 grip pose -> 右腕末端目标 pose，逻辑和左手一样，只是使用右手柄输入和右腕 idle pose。
    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=90.0,
            target_offset_pitch=-77.968,
            target_offset_yaw=90.0,
        ),
        name="right_ee_pose",
    )
    connected_right_se3 = right_se3.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )
    right_pose_rebaser = R11WristPoseRebaser(R11_RIGHT_IDLE_WRIST_POSE, name="right_ee_pose_rebased")
    connected_right_pose = right_pose_rebaser.connect({"ee_pose": connected_right_se3.output("ee_pose")})

    # 5. 每只手的输出关节顺序。这个顺序必须和下面 TensorReorderer 的 input_config 对齐。
    # 注意：最终送给 PinkIK action 的总顺序由 output_order 决定，而不是这里的列表顺序单独决定。
    left_hand_joint_names = [
        "left_index_proximal_joint",
        "left_middle_proximal_joint",
        "left_pinky_proximal_joint",
        "left_ring_proximal_joint",
        "left_thumb_metacarpal_joint",
        "left_thumb_distal_joint",
        "left_index_distal_joint",
        "left_middle_distal_joint",
        "left_pinky_distal_joint",
        "left_ring_distal_joint",
        "left_thumb_proximal_joint",
    ]
    right_hand_joint_names = [
        "right_index_proximal_joint",
        "right_middle_proximal_joint",
        "right_pinky_proximal_joint",
        "right_ring_proximal_joint",
        "right_thumb_metacarpal_joint",
        "right_thumb_distal_joint",
        "right_index_distal_joint",
        "right_middle_distal_joint",
        "right_pinky_distal_joint",
        "right_ring_distal_joint",
        "right_thumb_proximal_joint",
    ]

    # 6. 左/右 trigger -> 左/右手部关节。
    # R11ControllerTriggerGraspRetargeter 内部会根据 trigger 在 R11_HAND_OPEN_POSE 和 R11_HAND_CLOSED_POSE 之间插值。
    left_grasp = R11ControllerTriggerGraspRetargeter(
        left_hand_joint_names,
        name="left_trigger_grasp",
        open_pose=R11_HAND_OPEN_POSE,
        closed_pose=R11_HAND_CLOSED_POSE,
    )
    connected_left_grasp = left_grasp.connect(
        {"controller": transformed_controllers.output(ControllersSource.LEFT)}
    )
    right_grasp = R11ControllerTriggerGraspRetargeter(
        right_hand_joint_names,
        name="right_trigger_grasp",
        open_pose=R11_HAND_OPEN_POSE,
        closed_pose=R11_HAND_CLOSED_POSE,
    )
    connected_right_grasp = right_grasp.connect(
        {"controller": transformed_controllers.output(ControllersSource.RIGHT)}
    )

    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]

    # 7. 最终 action tensor 的展开顺序。
    # 必须严格匹配 ActionsCfg.upper_body_ik 的期望：
    # [左腕7维 pose, 右腕7维 pose, 22个手部关节]
    output_order = (
        left_ee_elements
        + right_ee_elements
        + [
            "left_index_proximal_joint",
            "left_middle_proximal_joint",
            "left_pinky_proximal_joint",
            "left_ring_proximal_joint",
            "left_thumb_metacarpal_joint",
            "right_index_proximal_joint",
            "right_middle_proximal_joint",
            "right_pinky_proximal_joint",
            "right_ring_proximal_joint",
            "right_thumb_metacarpal_joint",
            "left_index_distal_joint",
            "left_middle_distal_joint",
            "left_pinky_distal_joint",
            "left_ring_distal_joint",
            "left_thumb_proximal_joint",
            "right_index_distal_joint",
            "right_middle_distal_joint",
            "right_pinky_distal_joint",
            "right_ring_distal_joint",
            "right_thumb_proximal_joint",
            "left_thumb_distal_joint",
            "right_thumb_distal_joint",
        ]
    )

    # 8. TensorReorderer 负责把多个 retargeter 的输出合并、重排成一个一维 action。
    # 如果 output_order 和 action space 不一致，仿真可能不报错但控制会错位。
    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_joint_names,
            "right_hand_joints": right_hand_joint_names,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_pose.output("ee_pose"),
            "right_ee_pose": connected_right_pose.output("ee_pose"),
            "left_hand_joints": connected_left_grasp.output("hand_joints"),
            "right_hand_joints": connected_right_grasp.output("hand_joints"),
        }
    )

    # 9. OutputCombiner 暴露统一的 "action" 输出，teleop_se3_agent.py 每帧会读取这个 action 并送给 env.step()。
    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, [left_se3, right_se3, left_grasp, right_grasp]


@configclass
class SingleRigidBodyUsdFileCfg(UsdFileCfg):
    """Spawn a multi-part USD asset as a single rigid body.

    Some SimReady assets (e.g. a bottle with a separate cap) ship with a ``RigidBodyAPI`` on each
    part. :class:`~isaaclab.assets.RigidObject` requires exactly one rigid body under its prim, so
    spawning such assets directly fails. This spawner imports the asset, removes the per-part rigid
    bodies (keeping their colliders), and applies a single rigid body at the asset root so the whole
    asset behaves as one grabbable object.
    """

    func: Callable | str = "{DIR}.pickplace_seres_r11_env_cfg:spawn_single_rigid_body_usd"


@clone
def spawn_single_rigid_body_usd(
    prim_path: str,
    cfg: SingleRigidBodyUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a multi-part USD asset and consolidate it into a single rigid body.

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The spawner configuration.
        translation: Translation w.r.t. the parent prim. Defaults to None.
        orientation: Orientation (x, y, z, w) w.r.t. the parent prim. Defaults to None.

    Returns:
        The spawned asset root prim.
    """
    from pxr import UsdPhysics  # noqa: PLC0415

    # resolve the asset path (downloading from Nucleus if needed)
    usd_path = cfg.usd_path
    if check_file_path(usd_path) == 0:
        raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")
    if check_file_path(usd_path) == 2:
        usd_path = retrieve_file_path(usd_path, force_download=False)

    # import the asset as a single reference at the target prim path
    stage = get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
            stage=stage,
        )

    # strip per-part rigid bodies so only the asset root remains a rigid body
    root_prim = stage.GetPrimAtPath(prim_path)
    child_rigid_bodies = get_all_matching_child_prims(
        prim_path,
        predicate=lambda prim: bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)) and prim != root_prim,
        traverse_instance_prims=False,
    )
    for prim in child_rigid_bodies:
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        prim.RemoveAppliedSchema("PhysxRigidBodyAPI")

    # remove the asset's internal joints (e.g. cap<->body): after merging the bodies these would
    # connect the single root rigid body to itself, which PhysX rejects
    internal_joints = get_all_matching_child_prims(
        prim_path,
        predicate=lambda prim: bool(prim.IsA(UsdPhysics.Joint)),
        traverse_instance_prims=False,
    )
    for prim in internal_joints:
        stage.RemovePrim(prim.GetPath())

    # apply a single rigid body (+ mass) on the asset root
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage)
    else:
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage)

    return root_prim


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the R11 PickPlace scene."""

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.45, 0.45, 0.9996), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=PhysxRigidBodyPropertiesCfg(),
        ),
        # 水瓶配置保留在这里，后续如果要切回水瓶可以恢复这段 SingleRigidBodyUsdFileCfg。
        # spawn=SingleRigidBodyUsdFileCfg(
        #     usd_path=f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Food/Beverages/Bottled_Water_C01/sm_food_beverage_bottledWater_c01_01.usd",
        #     scale=(0.75, 0.75, 0.75),
        #     rigid_props=PhysxRigidBodyPropertiesCfg(),
        #     mass_props=MassPropertiesCfg(mass=0.05),
        # ),
    )

    robot: ArticulationCfg = R11_A2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.15),
            rot=(0.0, 0.0, 0.7071, 0.7071),
            joint_pos={
                **R11_FIXED_JOINT_POS,
                **R11_UPPER_BODY_INITIAL_JOINT_POS,
            },
            joint_vel={".*": 0.0},
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the R11 PickPlace MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        # Pink IK 的“主要可控关节”（进入 IK 求解变量的那部分）。这里显式列出关节名而非用正则，
        # 目的是保证动作维度与关节顺序稳定、可读。
        pink_controlled_joint_names=R11_ARM_JOINT_NAMES.copy(),
        hand_joint_names=[
            "left_index_proximal_joint",
            "left_middle_proximal_joint",
            "left_pinky_proximal_joint",
            "left_ring_proximal_joint",
            "left_thumb_metacarpal_joint",
            "right_index_proximal_joint",
            "right_middle_proximal_joint",
            "right_pinky_proximal_joint",
            "right_ring_proximal_joint",
            "right_thumb_metacarpal_joint",
            "left_index_distal_joint",
            "left_middle_distal_joint",
            "left_pinky_distal_joint",
            "left_ring_distal_joint",
            "left_thumb_proximal_joint",
            "right_index_distal_joint",
            "right_middle_distal_joint",
            "right_pinky_distal_joint",
            "right_ring_distal_joint",
            "right_thumb_proximal_joint",
            "left_thumb_distal_joint",
            "right_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": LEFT_EEF_LINK_NAME,
            "right_wrist": RIGHT_EEF_LINK_NAME,
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            urdf_path=os.path.join(ISAACLAB_ASSETS_DIR, "r11_0615.urdf"),
            mesh_path=ISAACLAB_ASSETS_DIR,
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=22,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame=LEFT_EEF_LINK_NAME,
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                FrameTaskCfg(
                    frame=RIGHT_EEF_LINK_NAME,
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                DampingTaskCfg(
                    cost=0.5,
                ),
                NullSpacePostureTaskCfg(
                    # Nullspace 姿态偏好任务：当双手末端位姿目标导致系统冗余时，
                    # 用 controlled_joints 这些关节在零空间里“顺便”把身体姿态拉到更合理的区域。
                    # 直觉：FrameTask 决定“手到哪”；NullSpacePosture 决定“手到那时身体怎么摆更舒服/更安全”。
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        LEFT_EEF_LINK_NAME,
                        RIGHT_EEF_LINK_NAME,
                    ],
                    controlled_joints=R11_NULLSPACE_JOINT_NAMES,
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[],
        ),
        enable_gravity_compensation=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the R11 PickPlace MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": LEFT_EEF_LINK_NAME})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": LEFT_EEF_LINK_NAME})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": RIGHT_EEF_LINK_NAME})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": RIGHT_EEF_LINK_NAME})

        hand_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": R11_HAND_JOINT_NAMES},
        )
        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["head_roll_Joint", "head_pitch_Joint", "head_yaw_Joint"]},
        )

        object = ObsTerm(
            func=mdp.object_obs,
            params={"left_eef_link_name": LEFT_EEF_LINK_NAME, "right_eef_link_name": RIGHT_EEF_LINK_NAME},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the R11 PickPlace MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done_pick_place, params={"task_link_name": RIGHT_EEF_LINK_NAME})


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceR11EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the R11 PickPlace teleoperation environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    temp_urdf_dir = tempfile.gettempdir()

    idle_action = [
        -0.249205,
        0.204645,
        1.156897,
        -0.074109,
        0.074109,
        -0.703206,
        0.703206,
        0.249213,
        0.204636,
        1.156897,
        0.074109,
        -0.074109,
        -0.703206,
        0.703206,
        *([0.0] * 22),
    ]

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 6
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2

        self.sim.physics = PhysxCfg(
            gpu_found_lost_pairs_capacity=2**26,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
        )

        # Match the GR1T2 waist-enabled setup: keep the base action config as
        # upper-body-only, then opt in the torso joints during environment init.
        for joint_name in R11_WAIST_JOINT_NAMES:
            if joint_name not in self.actions.upper_body_ik.pink_controlled_joint_names:
                self.actions.upper_body_ik.pink_controlled_joint_names.append(joint_name)

        self.actions.upper_body_ik.controller.usd_path = getattr(self.scene.robot.spawn, "usd_path")
        self.actions.upper_body_ik.controller.urdf_output_dir = self.temp_urdf_dir

        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.0),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _build_r11_pickplace_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )


@configclass
class PickPlaceR11MotionControllerEnvCfg(PickPlaceR11EnvCfg):
    """R11 pick-place teleoperation with Pico motion controllers instead of hand tracking."""

    def __post_init__(self):
        """Post initialization."""
        # 先复用原始 R11 pick-place 配置：场景、机器人、IK action、XR anchor 等都保持一致。
        super().__post_init__()

        # 这里只替换 IsaacTeleop pipeline：
        # 原 PickPlaceR11EnvCfg 使用 hand tracking pipeline；
        # 当前 MotionController 任务使用 PICO 手柄 pose + trigger pipeline。
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _build_r11_pickplace_motion_controller_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
