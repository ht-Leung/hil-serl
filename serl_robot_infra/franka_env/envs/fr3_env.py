"""Gym Interface for FR3 using FrankaInterface"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import sys

# Add paths for imports
sys.path.insert(0, "/home/hanyu/code/hil-serl/serl_hirol_infra")

# Import camera utilities
from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler

# Import FrankaInterface
from interface.franka_interface import FrankaInterface, ComplianceParams


class ImageDisplayer(threading.Thread):
    """Thread for displaying images in a separate window"""
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for FR3Env. Fill in the values below."""

    # Robot IP - FR3 default
    ROBOT_IP: str = "192.168.3.102"
    
    # Camera configuration
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": {"serial_number": "130322274175"},
        "wrist_2": {"serial_number": "127122270572"},
    }
    IMAGE_CROP: dict[str, callable] = {}
    
    # Task parameters
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    
    # Workspace limits
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    
    # Compliance parameters for smooth motion
    COMPLIANCE_PARAM: ComplianceParams = ComplianceParams(
        translational_stiffness=1500,
        translational_damping=80,
        rotational_stiffness=100,
        rotational_damping=10,
    )
    
    # Precision parameters for reset
    PRECISION_PARAM: ComplianceParams = ComplianceParams(
        translational_stiffness=2000,
        translational_damping=89,
        rotational_stiffness=150,
        rotational_damping=7,
    )
    
    # Reset compliance parameters
    RESET_PARAM: ComplianceParams = ComplianceParams(
        translational_stiffness=1800,
        translational_damping=85,
        rotational_stiffness=120,
        rotational_damping=8,
    )
    
    # Display and timing
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


class FR3Env(gym.Env):
    """FR3 Gym Environment using FrankaInterface for direct robot control"""
    
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        """
        Initialize FR3Env
        
        Args:
            hz: Control frequency  
            fake_env: Use simulation instead of real hardware
            save_video: Save video recordings of episodes
            config: Environment configuration
            set_load: Set load parameters (not implemented)
        """
        # Store configuration
        self.config = config if config is not None else DefaultEnvConfig()
        self.action_scale = self.config.ACTION_SCALE
        self._TARGET_POSE = self.config.TARGET_POSE
        self._RESET_POSE = self.config.RESET_POSE
        self._REWARD_THRESHOLD = self.config.REWARD_THRESHOLD
        self.max_episode_length = self.config.MAX_EPISODE_LENGTH
        self.display_image = self.config.DISPLAY_IMAGE
        self.gripper_sleep = self.config.GRIPPER_SLEEP
        
        # Control parameters
        self.hz = hz
        self.randomreset = self.config.RANDOM_RESET
        self.random_xy_range = self.config.RANDOM_XY_RANGE
        self.random_rz_range = self.config.RANDOM_RZ_RANGE
        self.joint_reset_cycle = self.config.JOINT_RESET_PERIOD
        
        # Convert reset pose from euler to quaternion
        self.resetpos = np.concatenate(
            [self.config.RESET_POSE[:3], euler_2_quat(self.config.RESET_POSE[3:])]
        )
        
        # State tracking
        self.curr_path_length = 0
        self.cycle_count = 0
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        
        # Video recording
        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []
        
        # Boundary box for safety
        self.xyz_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[:3],
            self.config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[3:],
            self.config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                for key in self.config.REALSENSE_CAMERAS}
                ),
            }
        )
        
        # Initialize interface if not fake_env
        if not fake_env:
            # Initialize FrankaInterface
            self.robot = FrankaInterface(robot_ip=self.config.ROBOT_IP)
            self._update_currpos()
            
            # Initialize cameras
            self.cap = None
            self.init_cameras(self.config.REALSENSE_CAMERAS)
            
            # Image display thread
            if self.display_image:
                self.img_queue = queue.Queue()
                self.displayer = ImageDisplayer(self.img_queue, "FR3Env")
                self.displayer.start()
            
            # Keyboard listener for termination
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
            
            print("Initialized FR3 Environment")
        else:
            self.robot = None
            # Initialize state variables for fake env
            self.currpos = self.resetpos.copy()
            self.currvel = np.zeros(6)
            self.currforce = np.zeros(3)
            self.currtorque = np.zeros(3)
            self.curr_gripper_pos = np.array([0.0])
            self.currjacobian = np.zeros((6, 7))
            self.q = np.zeros(7)
            self.dq = np.zeros(7)
    
    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}

    def compute_reward(self, obs: Dict) -> bool:
        """Compute binary reward based on pose threshold"""
        current_pose = obs["state"]["tcp_pose"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float) -> None:
        """Move the robot to the goal position with smooth trajectory."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        
        # Use trajectory mode for smooth motion
        self.robot.send_pos_trajectory_command(goal, finish_time=timeout)
        time.sleep(timeout + 0.1)  # Wait for motion to complete
        self.nextpos = goal
        self._update_currpos()

    def go_to_reset(self, joint_reset: bool = False) -> None:
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        
        # Update compliance parameters to precision mode
        if self.robot is not None:
            self.robot.update_params(self.config.PRECISION_PARAM)
        time.sleep(0.5)
        
        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET - Moving to home position")
            if self.robot is not None:
                self.robot.home()
            time.sleep(0.5)
        
        # Prepare reset pose
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
        else:
            reset_pose = self.resetpos.copy()
        
        # Move to reset position using smooth trajectory
        self.interpolate_move(reset_pose, timeout=1.0)
        
        # Change back to compliance mode
        if self.robot is not None:
            self.robot.update_params(self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset: bool = False, **kwargs) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        self.last_gripper_act = time.time()
        
        # Update compliance parameters
        if self.robot is not None:
            self.robot.update_params(self.config.COMPLIANCE_PARAM)
        
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self) -> None:
        """Save recorded video frames to disk"""
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    video_path = f'./videos/fr3_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict: Optional[Dict] = None) -> None:
        """Init cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def close_cameras(self) -> None:
        """Close all cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self) -> None:
        """Internal function to recover the robot from error state."""
        if self.robot is not None:
            self.robot.clear_errors()

    def _send_pos_command(self, pos: np.ndarray) -> None:
        """Internal function to send position command to the robot."""
        if self.robot is not None:
            self._recover()
            # Use servo mode for fast reactive control during steps
            self.robot.send_pos_command(pos)

    def _send_gripper_command(self, pos: float, mode: str = "binary") -> None:
        """Internal function to send gripper command to the robot."""
        if self.robot is None:
            return
            
        if mode == "binary":
            current_gripper = self.curr_gripper_pos[0] if isinstance(self.curr_gripper_pos, np.ndarray) else self.curr_gripper_pos
            if (pos <= -0.5) and (current_gripper > 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                self.robot.close_gripper()
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (current_gripper < 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                self.robot.open_gripper()
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
        else:
            # Continuous mode
            self.robot.send_gripper_command(pos, mode=mode)

    def _update_currpos(self) -> None:
        """Internal function to get the latest state of the robot and its gripper."""
        if self.robot is not None:
            state = self.robot.get_state()
            self.currpos = state["pose"]
            self.currvel = state["vel"]
            self.currforce = state["force"]
            self.currtorque = state["torque"]
            self.curr_gripper_pos = state["gripper_pos"]
            self.currjacobian = state["jacobian"]
            self.q = state["q"]
            self.dq = state["dq"]

    def update_currpos(self) -> None:
        """Public version of _update_currpos for compatibility"""
        self._update_currpos()

    def _get_obs(self) -> Dict:
        """Get observation dictionary"""
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
        if self.robot is not None:
            # Note: FrankaInterface uses context manager, but we can still try to clean up
            try:
                self.robot._robot.close()
            except:
                pass