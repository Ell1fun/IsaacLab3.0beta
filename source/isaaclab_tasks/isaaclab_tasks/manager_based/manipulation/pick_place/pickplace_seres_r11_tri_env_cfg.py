# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, check_file_path, retrieve_file_path
from isaaclab.utils.configclass import configclass

from . import mdp

from isaaclab_assets.robots.seres_r11 import (  # isort: skip
    HAND_JOINT_NAMES,
    R11_A2_HIGH_PD_CFG,
)


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
# Finger grasp pose presets for the controller-driven hand. Each entry maps one of
# the six actively driven joints per hand to its target angle [rad]. The trigger (or
# squeeze) blends linearly between the OPEN pose (released, grasp = 0) and the CLOSED
# pose (fully pressed, grasp = 1) so every finger moves together as the hand closes.
# Tune these to change the resting hand shape and the fully grasped shape. Keep the
# values within the Revo2 joint limits: finger proximal [0, 1.41], thumb metacarpal
# [0, 1.57], thumb proximal [0, 1.03]. The five distal joints per hand are PhysX
# mimic joints driven by the physics solver, so they are not listed here. The OPEN
# pose also seeds the robot's initial hand joint positions (see
# R11_UPPER_BODY_INITIAL_JOINT_POS) and the teleop idle action, so the hand starts in
# exactly the released configuration.
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
    # Initialize the hand at the released (open) grasp pose. Any driven joint not
    # listed in the open pose falls back to 0.0.
    **{name: R11_HAND_OPEN_POSE.get(name, 0.0) for name in R11_HAND_JOINT_NAMES},
}

LEFT_EEF_LINK_NAME = "left_wrist_yaw_Link"
RIGHT_EEF_LINK_NAME = "right_wrist_yaw_Link"
R11_LEFT_IDLE_WRIST_POSE = (-0.249205, 0.204645, 1.156897, -0.074109, 0.074109, -0.703206, 0.703206)
R11_RIGHT_IDLE_WRIST_POSE = (0.249213, 0.204636, 1.156897, 0.074109, -0.074109, -0.703206, 0.703206)

# Revo2 (R11) hand DOFs per side: the five driven proximal/metacarpal joints plus
# their coupled distal joints (11 per hand). These match the joint names in the R11
# USD/URDF so the teleop pipeline drives the correct DOFs by name.
R11_LEFT_HAND_RETARGET_JOINTS = [
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_pinky_proximal_joint",
    "left_ring_proximal_joint",
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
    "left_index_distal_joint",
    "left_middle_distal_joint",
    "left_pinky_distal_joint",
    "left_ring_distal_joint",
    "left_thumb_distal_joint",
]
R11_RIGHT_HAND_RETARGET_JOINTS = [
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_pinky_proximal_joint",
    "right_ring_proximal_joint",
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_distal_joint",
    "right_middle_distal_joint",
    "right_pinky_distal_joint",
    "right_ring_distal_joint",
    "right_thumb_distal_joint",
]
# Order in which the Pink IK action consumes the 22 hand joints (left + right). The
# TensorReorderer output and PinkInverseKinematicsActionCfg.hand_joint_names must
# both follow this exact order.
R11_PINK_HAND_JOINT_ORDER = [
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


class R11MotionControllerRetargeter:
    """Map a VR controller's trigger/squeeze input to the R11 Revo2 (five-finger) hand.

    IsaacTeleop ships ``TriHandMotionControllerRetargeter`` for the G1's three-finger
    hand (7 fixed DOFs). The R11 Revo2 hand has five fingers with 11 joints per hand,
    so this retargeter emits joint targets keyed by the real R11 joint names instead.

    A single grasp scalar (the larger of the trigger and squeeze axes) closes the
    whole hand, so all five fingers move together. Each of the six actively driven
    joints per hand interpolates linearly between an OPEN pose (grasp = 0) and a
    CLOSED pose (grasp = 1); see :data:`R11_HAND_OPEN_POSE` / :data:`R11_HAND_CLOSED_POSE`.

    The five distal joints per hand are PhysX mimic joints (negative gearing in the
    USD/URDF), so their targets are left at 0.0 and the physics solver curls them in
    lockstep with their proximal joints. This matches the working DexPilot pipeline in
    ``pickplace_seres_r11_env_cfg.py``, which drives the same six joints per hand.
    """

    def __init__(
        self,
        hand_joint_names: list[str],
        controller_side: str,
        name: str,
        open_pose: dict[str, float],
        closed_pose: dict[str, float],
    ):
        from isaacteleop.retargeting_engine.interface import BaseRetargeter
        from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
        from isaacteleop.retargeting_engine.tensor_types import (
            ControllerInput,
            ControllerInputIndex,
            RobotHandJoints,
        )

        # Precompute (open, span) per driven joint so _compute_fn only does a lerp.
        self._open = {name: float(open_pose.get(name, 0.0)) for name in hand_joint_names}
        self._span = {name: float(closed_pose.get(name, 0.0)) - self._open[name] for name in hand_joint_names}
        open_lookup = self._open
        span_lookup = self._span

        class _Retargeter(BaseRetargeter):
            def __init__(self, hand_joint_names: list[str], controller_side: str, name: str):
                self._hand_joint_names = list(hand_joint_names)
                self._side = controller_side.lower()
                if self._side not in ("left", "right"):
                    raise ValueError(f"controller_side must be 'left' or 'right', got: {controller_side}")
                super().__init__(name=name)

            def input_spec(self):
                return {f"controller_{self._side}": OptionalType(ControllerInput())}

            def output_spec(self):
                return {"hand_joints": RobotHandJoints(f"hand_joints_{self._side}", self._hand_joint_names)}

            def _compute_fn(self, inputs, outputs, context) -> None:
                output_group = outputs["hand_joints"]
                controller_group = inputs[f"controller_{self._side}"]
                if controller_group.is_none:
                    for i, joint_name in enumerate(self._hand_joint_names):
                        output_group[i] = open_lookup.get(joint_name, 0.0)
                    return
                trigger = float(controller_group[ControllerInputIndex.TRIGGER_VALUE])
                squeeze = float(controller_group[ControllerInputIndex.SQUEEZE_VALUE])
                grasp = max(trigger, squeeze)
                for i, joint_name in enumerate(self._hand_joint_names):
                    # Distal joints are PhysX mimic joints; leave their targets at 0.0.
                    if "distal" in joint_name:
                        output_group[i] = 0.0
                    else:
                        output_group[i] = open_lookup.get(joint_name, 0.0) + grasp * span_lookup.get(joint_name, 0.0)

        self._retargeter = _Retargeter(hand_joint_names, controller_side, name)

    def __getattr__(self, name: str):
        return getattr(self._retargeter, name)


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


def _build_r11_pickplace_pipeline():
    """Build an IsaacTeleop retargeting pipeline for R11 pick-place teleoperation.

    Uses Pico VR controllers: two Se3AbsRetargeters track each controller's grip
    pose for wrist control, and two R11MotionControllerRetargeters map the trigger
    and squeeze buttons to the Revo2 five-finger hand joints. All outputs are
    flattened into a single 36D action tensor via TensorReorderer with layout
    ``[left_wrist(7), right_wrist(7), hand_joints(22)]``.

    Returns:
        Tuple of (OutputCombiner, list): the pipeline exposing a single "action"
        output, and the list of Se3 retargeters [left_se3, right_se3] for tuning.
    """

    from isaacteleop.retargeters import (
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    # -------------------------------------------------------------------------
    # SE3 absolute pose retargeters (consume each controller's grip pose).
    # -------------------------------------------------------------------------
    # Offsets align the OpenXR controller grip frame with R11's wrist_yaw_Link
    # task frame; the R11WristPoseRebaser then re-centers tracking around the
    # robot's idle wrist pose so motion starts from a neutral configuration.
    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=90.0,
            target_offset_pitch=77.968,
            target_offset_yaw=90.0,
        ),
        name="left_ee_pose",
    )
    connected_left_se3 = left_se3.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )
    left_pose_rebaser = R11WristPoseRebaser(R11_LEFT_IDLE_WRIST_POSE, name="left_ee_pose_rebased")
    connected_left_pose = left_pose_rebaser.connect({"ee_pose": connected_left_se3.output("ee_pose")})

    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
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

    # -------------------------------------------------------------------------
    # R11 motion controller retargeters (grasp -> all five fingers).
    # -------------------------------------------------------------------------
    # Each retargeter emits 11 DOFs keyed by the real R11 Revo2 joint names (see
    # R11_LEFT_HAND_RETARGET_JOINTS). A single grasp scalar -- the larger of the
    # trigger and squeeze axes -- blends each driven joint from R11_HAND_OPEN_POSE
    # to R11_HAND_CLOSED_POSE, so all five fingers close together. The distal joints
    # are PhysX mimic joints, so the solver curls them with their proximal joints.
    left_hand = R11MotionControllerRetargeter(
        R11_LEFT_HAND_RETARGET_JOINTS,
        controller_side="left",
        name="left_hand",
        open_pose=R11_HAND_OPEN_POSE,
        closed_pose=R11_HAND_CLOSED_POSE,
    )
    connected_left_hand = left_hand.connect(
        {ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)}
    )

    right_hand = R11MotionControllerRetargeter(
        R11_RIGHT_HAND_RETARGET_JOINTS,
        controller_side="right",
        name="right_hand",
        open_pose=R11_HAND_OPEN_POSE,
        closed_pose=R11_HAND_CLOSED_POSE,
    )
    connected_right_hand = right_hand.connect(
        {ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)}
    )

    # -------------------------------------------------------------------------
    # TensorReorderer: flatten poses + hand joints into the 36D action tensor.
    # -------------------------------------------------------------------------
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]

    # Output order must match PinkInverseKinematicsActionCfg.hand_joint_names: the
    # 22 hand joints are interleaved left/right by proximal vs distal stage (see
    # R11_PINK_HAND_JOINT_ORDER). The reorderer pulls each real joint name from the
    # corresponding per-hand retargeter output.
    output_order = left_ee_elements + right_ee_elements + R11_PINK_HAND_JOINT_ORDER

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": R11_LEFT_HAND_RETARGET_JOINTS,
            "right_hand_joints": R11_RIGHT_HAND_RETARGET_JOINTS,
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
            "left_hand_joints": connected_left_hand.output("hand_joints"),
            "right_hand_joints": connected_right_hand.output("hand_joints"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, [left_se3, right_se3]


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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, 0.45, 0.9996), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=SingleRigidBodyUsdFileCfg(
            # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Food/Beverages/Bottled_Water_C01/sm_food_beverage_bottledWater_c01_01.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=PhysxRigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=0.05),
        ),
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
            urdf_path="/home/betydance/vsprojects/isaaclab_3/assets/r11_0615.urdf",
            mesh_path="/home/betydance/vsprojects/isaaclab_3/assets",
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
class PickPlaceR11TriEnvCfg(ManagerBasedRLEnvCfg):
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
        # Hand joints at the released (open) grasp pose, in Pink IK consumption order.
        # Distal joints are not in the open-pose dict, so they default to 0.0.
        *[R11_HAND_OPEN_POSE.get(joint_name, 0.0) for joint_name in R11_PINK_HAND_JOINT_ORDER],
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
