# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Auto-generated G1 motion-controller pick_place teleop env_cfg.

This file is generated from a validated PickPlaceVRSpec plus a model-proposed
G1MotionControllerAdaptationSpec. The model proposes parameters only; Python
code is rendered deterministically by my_docs/env_cfg_skills/generator_workflow.py.
"""

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg

from .pickplace_unitree_g1_inspire_hand_env_cfg import PickPlaceG1InspireFTPEnvCfg
from .pickplace_seres_r11_env_cfg import SingleRigidBodyUsdFileCfg


G1_HAND_OPEN_POSE = {'L_index_proximal_joint': 0.0, 'L_middle_proximal_joint': 0.0, 'L_pinky_proximal_joint': 0.0, 'L_ring_proximal_joint': 0.0, 'L_thumb_proximal_yaw_joint': 0.0, 'R_index_proximal_joint': 0.0, 'R_middle_proximal_joint': 0.0, 'R_pinky_proximal_joint': 0.0, 'R_ring_proximal_joint': 0.0, 'R_thumb_proximal_yaw_joint': 0.0, 'L_index_intermediate_joint': 0.0, 'L_middle_intermediate_joint': 0.0, 'L_pinky_intermediate_joint': 0.0, 'L_ring_intermediate_joint': 0.0, 'L_thumb_proximal_pitch_joint': 0.0, 'R_index_intermediate_joint': 0.0, 'R_middle_intermediate_joint': 0.0, 'R_pinky_intermediate_joint': 0.0, 'R_ring_intermediate_joint': 0.0, 'R_thumb_proximal_pitch_joint': 0.0, 'L_thumb_intermediate_joint': 0.0, 'R_thumb_intermediate_joint': 0.0, 'L_thumb_distal_joint': 0.0, 'R_thumb_distal_joint': 0.0}
G1_HAND_CLOSED_POSE = {'L_index_proximal_joint': 1.0, 'L_middle_proximal_joint': 1.0, 'L_pinky_proximal_joint': 1.0, 'L_ring_proximal_joint': 1.0, 'L_thumb_proximal_yaw_joint': 0.3, 'R_index_proximal_joint': 1.0, 'R_middle_proximal_joint': 1.0, 'R_pinky_proximal_joint': 1.0, 'R_ring_proximal_joint': 1.0, 'R_thumb_proximal_yaw_joint': 0.3, 'L_index_intermediate_joint': 1.0, 'L_middle_intermediate_joint': 1.0, 'L_pinky_intermediate_joint': 1.0, 'L_ring_intermediate_joint': 1.0, 'L_thumb_proximal_pitch_joint': 0.8, 'R_index_intermediate_joint': 1.0, 'R_middle_intermediate_joint': 1.0, 'R_pinky_intermediate_joint': 1.0, 'R_ring_intermediate_joint': 1.0, 'R_thumb_proximal_pitch_joint': 0.8, 'L_thumb_intermediate_joint': 1.0, 'R_thumb_intermediate_joint': 1.0, 'L_thumb_distal_joint': 0.8, 'R_thumb_distal_joint': 0.8}
G1_LEFT_HAND_RETARGET_JOINTS = ['L_index_proximal_joint', 'L_middle_proximal_joint', 'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_thumb_proximal_yaw_joint', 'L_index_intermediate_joint', 'L_middle_intermediate_joint', 'L_pinky_intermediate_joint', 'L_ring_intermediate_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 'L_thumb_distal_joint']
G1_RIGHT_HAND_RETARGET_JOINTS = ['R_index_proximal_joint', 'R_middle_proximal_joint', 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_thumb_proximal_yaw_joint', 'R_index_intermediate_joint', 'R_middle_intermediate_joint', 'R_pinky_intermediate_joint', 'R_ring_intermediate_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_intermediate_joint', 'R_thumb_distal_joint']
G1_PINK_HAND_JOINT_ORDER = ['L_index_proximal_joint', 'L_middle_proximal_joint', 'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_thumb_proximal_yaw_joint', 'R_index_proximal_joint', 'R_middle_proximal_joint', 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_thumb_proximal_yaw_joint', 'L_index_intermediate_joint', 'L_middle_intermediate_joint', 'L_pinky_intermediate_joint', 'L_ring_intermediate_joint', 'L_thumb_proximal_pitch_joint', 'R_index_intermediate_joint', 'R_middle_intermediate_joint', 'R_pinky_intermediate_joint', 'R_ring_intermediate_joint', 'R_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 'R_thumb_intermediate_joint', 'L_thumb_distal_joint', 'R_thumb_distal_joint']


class G1ControllerTriggerGraspRetargeter:
    """Map controller trigger/squeeze to G1 Inspire hand joint targets."""

    def __init__(self, hand_joint_names, controller_side, name, open_pose, closed_pose):
        from isaacteleop.retargeting_engine.interface import BaseRetargeter
        from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
        from isaacteleop.retargeting_engine.tensor_types import ControllerInput, ControllerInputIndex, RobotHandJoints

        open_lookup = {joint: float(open_pose.get(joint, 0.0)) for joint in hand_joint_names}
        span_lookup = {joint: float(closed_pose.get(joint, 0.0)) - open_lookup[joint] for joint in hand_joint_names}

        class _Retargeter(BaseRetargeter):
            def __init__(self, hand_joint_names, controller_side, name):
                self._hand_joint_names = list(hand_joint_names)
                self._side = controller_side.lower()
                super().__init__(name=name)

            def input_spec(self):
                return {f'controller_{self._side}': OptionalType(ControllerInput())}

            def output_spec(self):
                return {'hand_joints': RobotHandJoints(f'hand_joints_{self._side}', self._hand_joint_names)}

            def _compute_fn(self, inputs, outputs, context) -> None:
                output_group = outputs['hand_joints']
                controller_group = inputs[f'controller_{self._side}']
                if controller_group.is_none:
                    grasp = 0.0
                else:
                    trigger = float(controller_group[ControllerInputIndex.TRIGGER_VALUE])
                    squeeze = float(controller_group[ControllerInputIndex.SQUEEZE_VALUE])
                    grasp = max(trigger, squeeze)
                for i, joint_name in enumerate(self._hand_joint_names):
                    output_group[i] = open_lookup.get(joint_name, 0.0) + grasp * span_lookup.get(joint_name, 0.0)

        self._retargeter = _Retargeter(hand_joint_names, controller_side, name)

    def __getattr__(self, name):
        return getattr(self._retargeter, name)


def _build_g1_inspire_motion_controller_pickplace_pipeline():
    from isaacteleop.retargeters import Se3AbsRetargeter, Se3RetargeterConfig, TensorReorderer
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name='controllers')
    transform_input = ValueInput('world_T_anchor', TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=45.0,
            target_offset_pitch=180.0,
            target_offset_yaw=-90.0,
        ),
        name='left_ee_pose',
    )
    connected_left_se3 = left_se3.connect({ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)})

    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=False,
            use_wrist_position=False,
            target_offset_roll=-135.0,
            target_offset_pitch=0.0,
            target_offset_yaw=90.0,
        ),
        name='right_ee_pose',
    )
    connected_right_se3 = right_se3.connect({ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)})

    left_hand = G1ControllerTriggerGraspRetargeter(
        G1_LEFT_HAND_RETARGET_JOINTS, controller_side='left', name='left_hand',
        open_pose=G1_HAND_OPEN_POSE, closed_pose=G1_HAND_CLOSED_POSE,
    )
    connected_left_hand = left_hand.connect({ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)})
    right_hand = G1ControllerTriggerGraspRetargeter(
        G1_RIGHT_HAND_RETARGET_JOINTS, controller_side='right', name='right_hand',
        open_pose=G1_HAND_OPEN_POSE, closed_pose=G1_HAND_CLOSED_POSE,
    )
    connected_right_hand = right_hand.connect({ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)})

    left_ee_elements = ['l_pos_x', 'l_pos_y', 'l_pos_z', 'l_quat_x', 'l_quat_y', 'l_quat_z', 'l_quat_w']
    right_ee_elements = ['r_pos_x', 'r_pos_y', 'r_pos_z', 'r_quat_x', 'r_quat_y', 'r_quat_z', 'r_quat_w']
    output_order = left_ee_elements + right_ee_elements + G1_PINK_HAND_JOINT_ORDER
    reorderer = TensorReorderer(
        input_config={
            'left_ee_pose': left_ee_elements,
            'right_ee_pose': right_ee_elements,
            'left_hand_joints': G1_LEFT_HAND_RETARGET_JOINTS,
            'right_hand_joints': G1_RIGHT_HAND_RETARGET_JOINTS,
        },
        output_order=output_order,
        name='action_reorderer',
        input_types={
            'left_ee_pose': 'array',
            'right_ee_pose': 'array',
            'left_hand_joints': 'scalar',
            'right_hand_joints': 'scalar',
        },
    )
    connected_reorderer = reorderer.connect(
        {
            'left_ee_pose': connected_left_se3.output('ee_pose'),
            'right_ee_pose': connected_right_se3.output('ee_pose'),
            'left_hand_joints': connected_left_hand.output('hand_joints'),
            'right_hand_joints': connected_right_hand.output('hand_joints'),
        }
    )
    pipeline = OutputCombiner({'action': connected_reorderer.output('output')})
    return pipeline, [left_se3, right_se3]


class AutoPickPlaceUnitreeG1MugMotionControllerTeleopEnvCfg(PickPlaceG1InspireFTPEnvCfg):
    """Auto task for Unitree G1 Inspire FTP + motion controller."""

    def __post_init__(self):
        super().__post_init__()

        # Deterministic object/table/target overrides from PickPlaceVRSpec.
        self.scene.object.spawn = SingleRigidBodyUsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=PhysxRigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=0.05),
        )
        self.scene.object.init_state.pos = (-0.15, 0.55, 0.9996)
        self.scene.object.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.packing_table.init_state.pos = (0.0, 0.55, 0.0)
        self.scene.packing_table.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.target_pose = {'pos': [0.25, 0.45, 0.9996], 'quat_xyzw': [0.0, 0.0, 0.0, 1.0]}

        # Additional scene objects are props/distractors; success still uses self.scene.object.
        self.scene.bottled_water_c01 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SceneObjects/BottledWaterC01",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.43, 0.9996), rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=SingleRigidBodyUsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/SimReady/Residential/Kitchen/Food/Beverages/Bottled_Water_C01/sm_food_beverage_bottledWater_c01_01.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=PhysxRigidBodyPropertiesCfg(),
                mass_props=MassPropertiesCfg(mass=0.05),
            ),
        )

        # Replace hand-tracking teleop pipeline with controller trigger-open/close pipeline.
        self.isaac_teleop.pipeline_builder = lambda: _build_g1_inspire_motion_controller_pickplace_pipeline()[0]
