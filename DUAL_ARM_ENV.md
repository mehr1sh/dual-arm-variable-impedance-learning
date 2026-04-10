# Dual-Arm Panda Environment

A Gymnasium-compatible MuJoCo environment for dual-arm manipulation tasks with two Franka Panda robots.

## Overview

The `DualArmPandaEnv` simulates two Franka Panda 7-DOF arms with parallel grippers in a shared workspace, along with a movable object. The environment provides:

- **Dual 7-DOF manipulators** positioned symmetrically (side-by-side)
- **End-effector tracking** with full pose information
- **Object manipulation** with 6-DOF pose tracking
- **Gym-style interface** compatible with standard RL frameworks
- **Rich observations** including joint states, end-effector poses, object state, and velocities

## Files Created

### Core Environment
- **[davil/envs/dual_arm_env.py](davil/envs/dual_arm_env.py)** - Main `DualArmPandaEnv` class with:
  - Observation space: 55-dimensional state vector
  - Action space: 16-dimensional continuous controls (8 per arm)
  - Reset and step functions
  - State management

### Model Definition
- **[assets/models/dual_arm_panda.xml](assets/models/dual_arm_panda.xml)** - MuJoCo model with:
  - Two complete Panda arm hierarchies
  - Synchronized finger joints with tendons
  - A movable object for manipulation tasks
  - Properly configured actuators and contacts

### Examples & Tests
- **[scripts/test_dual_arm_env.py](scripts/test_dual_arm_env.py)** - Comprehensive test suite
- **[scripts/example_dual_arm.py](scripts/example_dual_arm.py)** - Usage examples

## Environment Specifications

### Observation Space (55 dimensions)

```
┌─────────────────────────────────────┬────┐
│ Panda 1 (Left Arm)                 │ 28 │
├─────────────────────────────────────┼────┤
│ - Joint positions (q)               │  7 │
│ - Joint velocities (qdot)           │  7 │
│ - EE position (x, y, z)             │  3 │
│ - EE quaternion (x, y, z, w)        │  4 │
├─────────────────────────────────────┼────┤
│ Panda 2 (Right Arm)                │ 28 │
├─────────────────────────────────────┼────┤
│ - Joint positions (q)               │  7 │
│ - Joint velocities (qdot)           │  7 │
│ - EE position (x, y, z)             │  3 │
│ - EE quaternion (x, y, z, w)        │  4 │
├─────────────────────────────────────┼────┤
│ Object State                        │ 13 │
├─────────────────────────────────────┼────┤
│ - Position (x, y, z)                │  3 │
│ - Quaternion (x, y, z, w)           │  4 │
│ - Linear velocity                   │  3 │
│ - Angular velocity                  │  3 │
└─────────────────────────────────────┴────┘
Total: 55 dimensions
```

### Action Space (16 dimensions)

- **Normalized range**: [-1.0, 1.0]
- **8 controls per arm**:
  - 7 joint actuators
  - 1 gripper actuator
- **Action decomposition**:
  ```python
  action[0:7]   # Panda1 joint commands
  action[7]     # Panda1 gripper command
  action[8:15]  # Panda2 joint commands
  action[15]    # Panda2 gripper command
  ```

### Coordinate Frame

- **Base frame**: World (z-axis pointing up)
- **Panda1**: Positioned at (-0.6, 0, 0) in world frame
- **Panda2**: Positioned at (+0.6, 0, 0) in world frame
- **Object**: Initialized at (0, 0, 0.5)

## Usage

### Basic Initialization

```python
from davil.envs import DualArmPandaEnv

# Create environment
env = DualArmPandaEnv(render_mode=None)

# Reset to initial state
obs, info = env.reset()

# Get initial information
print(f"Observation shape: {obs.shape}")
print(f"Panda1 EE position: {info['panda1_ee_pos']}")
print(f"Panda2 EE position: {info['panda2_ee_pos']}")
print(f"Object position: {info['object_pos']}")
```

### Stepping the Environment

```python
# Sample a random action
action = env.action_space.sample()

# Step the simulation
obs, reward, terminated, truncated, info = env.step(action)

# Access state information
panda1_ee_pos = info['panda1_ee_pos']
panda2_ee_pos = info['panda2_ee_pos']
object_pos = info['object_pos']
```

### Decomposing Observations

```python
# Extract components from the observation vector
panda1_q = obs[0:7]           # Panda1 joint positions
panda1_qd = obs[7:14]         # Panda1 joint velocities
panda1_ee_pos = obs[14:17]    # Panda1 EE position
panda1_ee_quat = obs[17:21]   # Panda1 EE quaternion

panda2_q = obs[21:28]         # Panda2 joint positions
panda2_qd = obs[28:35]        # Panda2 joint velocities
panda2_ee_pos = obs[35:38]    # Panda2 EE position
panda2_ee_quat = obs[38:42]   # Panda2 EE quaternion

obj_pos = obs[42:45]          # Object position
obj_quat = obs[45:49]         # Object quaternion
obj_lin_vel = obs[49:52]      # Object linear velocity
obj_ang_vel = obs[52:55]      # Object angular velocity
```

### Custom State Setting

```python
import numpy as np

# Get current state
qpos, qvel = env.get_state()

# Set custom state
env.set_state(qpos, qvel)

# Get observation after state change
obs = env._get_observation()
```

## Technical Details

### Joint Limits

All Panda joints follow the standard Franka Emika specifications:
- **Joints 1,3,5,7**: ±2.8973 rad
- **Joint 2**: ±1.7628 rad
- **Joint 4**: -3.0718 to -0.0698 rad
- **Joint 6**: -0.0175 to 3.7525 rad
- **Fingers**: 0 to 0.04 m (slide joints)

### Gripper Control

- **Open**: action ∈ [-1, 0) → gripper opens
- **Close**: action ∈ [0, 1] → gripper closes
- **Mapping**: action_normalized [-1, 1] → control value [0, 255]

### Physics Configuration

- **Integrator**: Implicit Fast (implicitfast)
- **Timestep**: 0.01 seconds (default)
- **Collision detection**: Enabled with mesh geometries
- **Tendon synchronization**: Ensures symmetric finger motion

## Extending the Environment

### Custom Reward Function

To add your own reward signal:

```python
def _compute_reward(self, action):
    """
    Compute reward based on task
    """
    # Example: minimize distance to object
    panda1_ee_pos = self.data.xpos[self.panda1_hand_id]
    obj_pos = self.data.xpos[self.object_id]
    distance = np.linalg.norm(panda1_ee_pos - obj_pos)
    
    return -distance
```

### Task-Specific Observations

You can override `_get_observation()` to include task-specific features:

```python
def _get_observation(self):
    base_obs = super()._get_observation()
    
    # Add distance to object
    panda1_ee_pos = self.data.xpos[self.panda1_hand_id]
    obj_pos = self.data.xpos[self.object_id]
    distance = np.linalg.norm(panda1_ee_pos - obj_pos)
    
    return np.concatenate([base_obs, [distance]])
```

## Integration with RL Frameworks

### Stable Baselines 3

```python
from stable_baselines3 import PPO
from davil.envs import DualArmPandaEnv

env = DualArmPandaEnv()
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100000)
```

### Custom Training Loop

```python
env = DualArmPandaEnv()
for episode in range(num_episodes):
    obs, info = env.reset()
    
    for step in range(steps_per_episode):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
```

## Troubleshooting

### Mesh Not Found

Ensure meshdir in [dual_arm_panda.xml](assets/models/dual_arm_panda.xml) points to the correct path:
```xml
<compiler angle="radian" meshdir="../../docs/mujoco_menagerie/franka_emika_panda/assets"/>
```

### Rendering Issues

- Ensure you have display drivers installed (X11 for Linux)
- Use `render_mode=None` for headless operation

### NaN Values in Simulation

This usually indicates:
1. Infeasible actions (beyond joint limits)
2. Numerical instability (very large timesteps)
3. Object falling out of workspace

## Future Enhancements

- [ ] Rendering with rgb_array mode
- [ ] Per-arm force/torque sensors
- [ ] Tactile sensor simulation
- [ ] More sophisticated gripper models
- [ ] Multi-object manipulation
- [ ] Visual observations (camera rendering)

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Franka Panda Specifications](https://franka.de/)

## Contact

For issues or questions about the dual-arm environment, refer to the main repository documentation.
