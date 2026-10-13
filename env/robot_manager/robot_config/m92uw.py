from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from env.global_configs import ROBOTS_PATH


def get_robot_config():
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ROBOTS_PATH}/m92uw/M92UW.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=5000.0,
                max_angular_velocity=5000.0,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                stabilization_threshold=0.001,
                fix_root_link=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "l_arm1_shoulder_p_joint": 0.0,
                "l_arm2_shoulder_r_joint": 0.25,
                "l_arm3_shoulder_y_joint": 0.0,
                "l_arm4_elbow_p_joint": -1.4,
                "l_arm5_wrist_y_joint": 0.0,
                "l_arm6_wrist_p_joint": -0.26,
                "l_arm7_wrist_r_joint": 0.0,
                "r_arm1_shoulder_p_joint": 0.0,
                "r_arm2_shoulder_r_joint": -0.25,
                "r_arm3_shoulder_y_joint": 0.0,
                "r_arm4_elbow_p_joint": -1.4,
                "r_arm5_wrist_y_joint": 0.0,
                "r_arm6_wrist_p_joint": -0.26,
                "r_arm7_wrist_r_joint": 0.0,
                "[lr]_(index|middle|ring|pinky)_(mcp|pip)_joint": 0.0,
                "[lr]_thumb_(cmc_y|cmc_p|mcp|ip)_joint": 0.0,
                "ankle_p_joint": -0.71,
                "knee_p_joint": 1.3,
                "waist1_p_joint": -0.61,
                "waist2_y_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=["(ankle_p|knee_p|waist1_p|waist2_y)_joint"],
                effort_limit_sim=3000.0,
                velocity_limit_sim=300.0,
                stiffness=3000.0,
                damping=400.0,
                armature=0.1,
            ),
            "arm_shoulders": ImplicitActuatorCfg(
                joint_names_expr=["[lr]_arm[12]_.*_joint"],
                velocity_limit_sim=20.0,
                effort_limit_sim=1070.0,
                stiffness=3000.0,
                damping=300.0,
                armature=0.05,
            ),
            "arm_shoulder_yaw_elbow": ImplicitActuatorCfg(
                joint_names_expr=["[lr]_arm[34]_.*_joint"],
                velocity_limit_sim=20.0,
                effort_limit_sim=700.0,
                stiffness=2500.0,
                damping=250.0,
                armature=0.03,
            ),
            "arm_wrist_yaw": ImplicitActuatorCfg(
                joint_names_expr=["[lr]_arm5_.*_joint"],
                velocity_limit_sim=20.0,
                effort_limit_sim=360.0,
                stiffness=1800.0,
                damping=180.0,
                armature=0.02,
            ),
            "arm_wrist_pitch_roll": ImplicitActuatorCfg(
                joint_names_expr=["[lr]_arm[67]_.*_joint"],
                velocity_limit_sim=20.0,
                effort_limit_sim=150.0,
                stiffness=1200.0,
                damping=120.0,
                armature=0.02,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    "[lr]_(index|middle|ring|pinky)_(mcp|pip)_joint",
                    "[lr]_thumb_(cmc_[yp]|mcp|ip)_joint",
                ],
                effort_limit_sim=2000.0,
                velocity_limit_sim=2000.0,
                stiffness=250.0,
                damping=2.5,
                armature=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
