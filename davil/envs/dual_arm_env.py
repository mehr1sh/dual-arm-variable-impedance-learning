"""
Dual-Arm MuJoCo Environment - Gymnasium compatible
Implements observations, actions, reset, and step for dual-arm manipulation tasks
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import viewer
import os
import warnings


class DualArmPandaEnv(gym.Env):
    """
    Dual-arm Franka Panda environment for manipulation tasks
    - Two 7-DOF arms with parallel grippers
    - One movable object
    - Gym-style interface with observations and actions
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}
    
    def __init__(self, render_mode=None, model_path=None, dt=0.01):
        """
        Initialize dual-arm environment
        
        Args:
            render_mode: "human" for MuJoCo viewer, "rgb_array" for images, None for no rendering
            model_path: Path to dual_arm_panda.xml (auto-finds if None)
            dt: Simulation timestep
        """
        self.render_mode = render_mode
        self.dt = dt
        self.t = 0
        self._viewer = None
        
        # Auto-find model if not provided
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), 
                '../..', 'assets', 'models', 'dual_arm_panda.xml'
            )
        
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Get body and joint indices
        self._setup_indices()
        
        # Action space: 8 controls per arm (7 joints + 1 gripper)
        # Values in [-1, 1] normalized
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )
        
        # Observation space
        # Each arm: 7 joint positions + 7 joint velocities + 3 EE pos + 4 EE quat
        # Object: 3 pos + 4 quat + 3 lin vel + 3 ang vel
        # Total: 2*(7+7+3+4) + (3+4+3+3) = 42 + 13 = 55
        obs_dim = 2 * (7 + 7 + 3 + 4) + (3 + 4 + 3 + 3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
    def _setup_indices(self):
        """Setup body and joint indices for quick access"""
        # Panda 1 joints
        self.panda1_joint_ids = [
            self.model.joint(f"panda1_joint{i}").id for i in range(1, 8)
        ]
        self.panda1_finger_ids = [
            self.model.joint(f"panda1_finger_joint{i}").id for i in range(1, 3)
        ]
        
        # Panda 2 joints
        self.panda2_joint_ids = [
            self.model.joint(f"panda2_joint{i}").id for i in range(1, 8)
        ]
        self.panda2_finger_ids = [
            self.model.joint(f"panda2_finger_joint{i}").id for i in range(1, 3)
        ]
        
        # End-effector bodies
        self.panda1_hand_id = self.model.body("panda1_hand").id
        self.panda2_hand_id = self.model.body("panda2_hand").id
        
        # Object body
        self.object_id = self.model.body("object").id
        
        # For easy access: all arm joint IDs
        self.all_arm_joints = self.panda1_joint_ids + self.panda2_joint_ids
        self.n_arm_joints = len(self.all_arm_joints)
        
    def _get_observation(self):
        """
        Get observation from current state
        
        Returns:
            obs: array of shape (55,) with all observations
        """
        obs = []
        
        # Panda 1: joint positions, velocities, EE pose
        panda1_qpos = np.array([self.data.qpos[jid] for jid in self.panda1_joint_ids])
        panda1_qvel = np.array([self.data.qvel[jid] for jid in self.panda1_joint_ids])
        panda1_ee_pos = self.data.xpos[self.panda1_hand_id].copy()
        panda1_ee_quat = self.data.xquat[self.panda1_hand_id].copy()
        
        obs.extend(panda1_qpos)
        obs.extend(panda1_qvel)
        obs.extend(panda1_ee_pos)
        obs.extend(panda1_ee_quat)
        
        # Panda 2: joint positions, velocities, EE pose
        panda2_qpos = np.array([self.data.qpos[jid] for jid in self.panda2_joint_ids])
        panda2_qvel = np.array([self.data.qvel[jid] for jid in self.panda2_joint_ids])
        panda2_ee_pos = self.data.xpos[self.panda2_hand_id].copy()
        panda2_ee_quat = self.data.xquat[self.panda2_hand_id].copy()
        
        obs.extend(panda2_qpos)
        obs.extend(panda2_qvel)
        obs.extend(panda2_ee_pos)
        obs.extend(panda2_ee_quat)
        
        # Object: position, orientation, linear velocity, angular velocity
        obj_pos = self.data.xpos[self.object_id].copy()
        obj_quat = self.data.xquat[self.object_id].copy()
        # Get velocity from body (cvel is [linear_vel, angular_vel])
        obj_cvel = self.data.cvel[self.object_id].copy()
        obj_lin_vel = obj_cvel[:3]
        obj_ang_vel = obj_cvel[3:]
        
        obs.extend(obj_pos)
        obs.extend(obj_quat)
        obs.extend(obj_lin_vel)
        obs.extend(obj_ang_vel)
        
        return np.array(obs, dtype=np.float32)
    
    def _apply_action(self, action):
        """
        Apply action to the environment
        
        Args:
            action: array of shape (16,) with normalized values [-1, 1]
        """
        assert len(action) == 16, f"Action size must be 16, got {len(action)}"
        
        # Denormalize actions to control ranges
        # First 7 joints of panda1
        for i in range(7):
            jid = self.panda1_joint_ids[i]
            ctrlrange = self.model.actuator_ctrlrange[i]  # Get from actuator
            if ctrlrange[0] != ctrlrange[1]:
                # Map [-1, 1] to [ctrlrange_min, ctrlrange_max]
                ctrl = ctrlrange[0] + (action[i] + 1) * (ctrlrange[1] - ctrlrange[0]) / 2
            else:
                ctrl = action[i] * 2.8973  # Default range
            self.data.ctrl[i] = ctrl
        
        # Gripper for panda1 (control index 7)
        self.data.ctrl[7] = (action[7] + 1) * 127.5  # Map [-1, 1] to [0, 255]
        
        # First 7 joints of panda2
        for i in range(7):
            jid = self.panda2_joint_ids[i]
            ctrlrange = self.model.actuator_ctrlrange[8 + i]
            if ctrlrange[0] != ctrlrange[1]:
                ctrl = ctrlrange[0] + (action[8 + i] + 1) * (ctrlrange[1] - ctrlrange[0]) / 2
            else:
                ctrl = action[8 + i] * 2.8973
            self.data.ctrl[8 + i] = ctrl
        
        # Gripper for panda2 (control index 15)
        self.data.ctrl[15] = (action[15] + 1) * 127.5
        
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state
        
        Returns:
            obs, info
        """
        super().reset(seed=seed)
        
        # Reset time
        self.t = 0
        
        # Set both arms to a nice home configuration
        home_config = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        
        # Panda 1 at home
        for i, jid in enumerate(self.panda1_joint_ids):
            self.data.qpos[jid] = home_config[i]
        
        # Panda 2 at home (mirrored in y)
        for i, jid in enumerate(self.panda2_joint_ids):
            self.data.qpos[jid] = home_config[i]
        
        # Gripper closed for both arms
        self.data.qpos[self.panda1_finger_ids[0]] = 0.0
        self.data.qpos[self.panda1_finger_ids[1]] = 0.0
        self.data.qpos[self.panda2_finger_ids[0]] = 0.0
        self.data.qpos[self.panda2_finger_ids[1]] = 0.0
        
        # Object at center
        obj_body = self.model.body("object")
        if obj_body.dofnum[0] >= 0:
            obj_qpos_id = int(obj_body.dofnum[0])
            self.data.qpos[obj_qpos_id:obj_qpos_id+3] = [0, 0, 0.5]
            self.data.qpos[obj_qpos_id+3:obj_qpos_id+7] = [1, 0, 0, 0]  # identity quat
        
        # Zero velocities
        self.data.qvel[:] = 0
        
        # Forward pass
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_observation()
        info = {
            "time": self.t,
            "panda1_ee_pos": self.data.xpos[self.panda1_hand_id].copy(),
            "panda2_ee_pos": self.data.xpos[self.panda2_hand_id].copy(),
            "object_pos": self.data.xpos[self.object_id].copy(),
        }
        
        return obs, info
    
    def step(self, action):
        """
        Step the environment
        
        Args:
            action: array of shape (16,) with normalized actions
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Apply action
        self._apply_action(action)
        
        # Simulate
        mujoco.mj_step(self.model, self.data)
        self.t += self.dt
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward (can be customized)
        reward = self._compute_reward(action)
        
        # Check termination
        terminated = False
        # Could add: if object falls off table, etc.
        
        # Check truncation (time limit)
        truncated = False  # Can be set based on max steps
        
        info = {
            "time": self.t,
            "panda1_ee_pos": self.data.xpos[self.panda1_hand_id].copy(),
            "panda2_ee_pos": self.data.xpos[self.panda2_hand_id].copy(),
            "object_pos": self.data.xpos[self.object_id].copy(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action):
        """
        Compute reward (placeholder)
        Can be customized based on task
        """
        # Placeholder: small negative reward for each step
        return -0.01
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self._viewer is None:
                try:
                    # Suppress Wayland-related warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=Warning)
                        self._viewer = viewer.launch_passive(self.model, self.data)
                except Exception as e:
                    warnings.warn(
                        f"Could not launch viewer: {e}\n"
                        "You're likely on Wayland. Try running with:\n"
                        "  GDK_BACKEND=x11 python your_script.py\n"
                        "Or use render_mode=None for headless operation.",
                        RuntimeWarning
                    )
                    self._viewer = None
                    return
            if self._viewer is not None:
                self._viewer.sync()
        elif self.render_mode == "rgb_array":
            # Not implemented yet, but can use mujoco._renderer
            raise NotImplementedError("RGB array rendering not yet implemented")
    
    def close(self):
        """Close the environment"""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
    
    def set_state(self, qpos, qvel=None):
        """
        Manually set the state
        
        Args:
            qpos: array of generalized positions
            qvel: array of generalized velocities (zeros if None)
        """
        self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        else:
            self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
    
    def get_state(self):
        """Get current state"""
        return self.data.qpos.copy(), self.data.qvel.copy()


if __name__ == "__main__":
    # Test the environment
    import sys
    
    # Check for Wayland and provide helpful message
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        print("\n⚠️  Wayland detected! For visualization, run with:")
        print("   GDK_BACKEND=x11 python dual_arm_env.py")
        print("\nOr use headless mode (no viewer):")
        print("   python dual_arm_env.py --headless\n")
        
        if "--headless" not in sys.argv:
            print("Switching to headless mode for this test...")
            render_mode = None
        else:
            render_mode = None
    else:
        render_mode = "human"
    
    env = DualArmPandaEnv(render_mode=render_mode)
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Simple test loop
    try:
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if (i + 1) % 20 == 0:
                print(f"Step {i+1}: panda1_ee={info['panda1_ee_pos'].round(3)}, object={info['object_pos'].round(3)}")
            
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()
    
    print("✓ Test completed successfully!")
