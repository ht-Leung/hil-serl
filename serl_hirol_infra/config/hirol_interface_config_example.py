"""
Example configuration for HIROLInterface
This demonstrates how to configure the interface for different scenarios
"""

# For FR3 robot with simulation
FR3_SIM_CONFIG = {
    "robot_config": {
        "robot_name": "fr3",
        "use_hardware": False,
        "use_simulation": True,
        "robot_config": {
            "fr3": {
                "dof": 7,
                "ip": "192.168.3.102",  # Not used in simulation
                "collision_behaviour": {
                    "torque_min": [20.0] * 7,
                    "torque_max": [20.0] * 7,
                    "force_min": [20.0] * 6,
                    "force_max": [20.0] * 6,
                }
            }
        },
        "tool_type": "gripper",
        "tool_config": {
            "franka_hand": {
                "ip": "192.168.3.102",  # Not used in simulation
                "width": 0.08,
            }
        },
    },
    "motion_config": {
        "model_type": "model",
        "controller_type": "ik",
        "use_trajectory_planner": True,
        "buffer_type": "cartesian",
        "plan_type": "cartesian",
        "trajectory_planner_type": "cart_polynomial",
        "traj_frequency": 450,
        "control_frequency": 1000,
        "model_config": {
            "name": "fr3_franka_hand",
            "cfg": {
                "fr3_franka_hand": {
                    "urdf_path": "assets/franka_fr3/fr3_franka_hand.urdf",
                    "mesh_offset": "assets/franka_fr3",
                    "frames": ["base", "fr3_link0", "fr3_link1", "fr3_link2", 
                              "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6", 
                              "fr3_link7", "fr3_link8", "fr3_hand_tcp", "fr3_hand", 
                              "fr3_leftfinger", "fr3_rightfinger"],
                    "base_link": "base",
                    "ee_link": "fr3_hand_tcp",
                    "fixed_base": True,
                    "lock_joints": [
                        {"base": "fr3_hand", "end": "fr3_leftfinger"},
                        {"base": "fr3_hand", "end": "fr3_rightfinger"}
                    ]
                }
            }
        },
        "controller_config": {
            "ik": {
                "rotation_weight": 0.5,
                "position_weight": 1.0,
                "smooth_weight": 1e-5,
                "damping": 1e-2,
                "solver": "dls",
            }
        },
        "trajectory_config": {
            "cartesian": {
                "size": 1000,
                "dim": 7
            },
            "cart_polynomial": {
                "degree": 5,
                "duration": 1.0,
            }
        }
    }
}

# For FR3 robot with hardware
FR3_HARDWARE_CONFIG = {
    "robot_config": {
        "robot_name": "fr3",
        "use_hardware": True,
        "use_simulation": False,
        "robot_config": {
            "fr3": {
                "dof": 7,
                "ip": "192.168.3.102",  # Real robot IP
                "collision_behaviour": {
                    "torque_min": [20.0] * 7,
                    "torque_max": [20.0] * 7,
                    "force_min": [20.0] * 6,
                    "force_max": [20.0] * 6,
                }
            }
        },
        "tool_type": "gripper",
        "tool_config": {
            "franka_hand": {
                "ip": "192.168.3.102",  # Real gripper IP
                "width": 0.08,
            }
        },
    },
    "motion_config": {
        # Same as simulation config
        **FR3_SIM_CONFIG["motion_config"]
    }
}

# Minimal configuration for testing
MINIMAL_TEST_CONFIG = {
    "robot_config": {
        "robot_name": "fr3",
        "use_hardware": False,
        "use_simulation": True,
        "robot_config": {"fr3": {"dof": 7}},
        "tool_type": "gripper",
        "tool_config": {"franka_hand": {"width": 0.08}},
    },
    "motion_config": {
        "model_type": "model",
        "controller_type": "ik",
        "use_trajectory_planner": False,
        "control_frequency": 1000,
        "model_config": {
            "name": "fr3_franka_hand",
            "cfg": {
                "fr3_franka_hand": {
                    "urdf_path": "assets/franka_fr3/fr3_franka_hand.urdf",
                    "ee_link": "fr3_hand_tcp",
                    "fixed_base": True,
                }
            }
        },
        "controller_config": {
            "ik": {"damping": 1e-2}
        },
    }
}