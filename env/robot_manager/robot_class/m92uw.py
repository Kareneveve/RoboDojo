from copy import deepcopy
import os

import numpy as np

from env.robot_manager.mobile_base import M92UWMobileBaseAdapter
from utils.pipeline_utils import get_embodiment_config_by_robot_type


class _M92UWArm:
    """One arm view of the shared M92UW articulation description."""

    side: str
    mobile_base_adapter_cls = M92UWMobileBaseAdapter

    def __init__(self, cfg: dict):
        self.robot_type = cfg.get("robot_type", None)
        base_name = cfg.get("robot_name", None)
        self.is_coupled = cfg.get("coupled", False)
        self.default_root_pos = cfg.get("default_root_pos", None)
        self.default_root_rot = cfg.get("default_root_rot", None)
        self.SceneCfg = None
        self.static_camera_list = None

        self.robot_cfg = get_embodiment_config_by_robot_type(
            robot_type=self.robot_type, robot_name=base_name
        )
        description = self.robot_cfg["robot_config"]
        common = description["common"]
        arm = description["arms"][self.side]

        self.robot_file = self.robot_cfg["robot_file"]
        self.robot_args = deepcopy(common)
        self.robot_args.update(deepcopy(arm))
        self.planner_config = deepcopy(description.get("planner"))
        self.control_groups = deepcopy(description.get("control_groups") or {})
        self.default_control_group = arm.get("default_control_group", f"{self.side}_arm_waist")
        self.robot_name = f"{self.side}_{base_name}"
        self.urdf_path = os.path.join(self.robot_file, common["urdf_path"])
        self.mesh_dir = os.path.join(self.robot_file, common.get("mesh_dir", "."))
        self.curobo_yml_path = os.path.join(self.robot_file, common["curobo_yml"])
        self.target_tool_frame = arm["ee_link"]
        self.ee_joint_name = arm["ee_joint"]
        self.ee_link_name = arm["ee_link"]
        self.arm_joints_name = list(arm["arm_joints"])
        self.gripper_joints_name = list(arm["hand_joints"])
        self.waist_joints_name = list(common.get("waist_joints", []))
        self.leg_joints_name = list(common.get("leg_joints", []))
        self.arm_home = list(arm["arm_home"])
        self.non_arm_home = deepcopy(common["non_arm_home"])
        self.ee_type = common.get("ee_type", "hand")
        self.gripper_scale = common.get("gripper_scale", [0.0, 1.0])
        self.base_link = common.get("base_link", "base_footprint")
        self.head_tool_frame = common.get("head_tool_frame", None)

        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.delta_matrix = np.array(arm.get("delta_matrix", common.get("delta_matrix", identity)))
        self.inv_delta_matrix = np.linalg.inv(self.delta_matrix)
        self.global_trans_matrix = np.array(common.get("global_trans_matrix", identity))
        self.entity_origin_pose = self.default_root_pos + self.default_root_rot

        self.camera = deepcopy(arm.get("camera", []))
        if self.side == "left":
            self.camera = deepcopy(common.get("camera", [])) + self.camera


class LeftM92UW(_M92UWArm):
    side = "left"


class RightM92UW(_M92UWArm):
    side = "right"
