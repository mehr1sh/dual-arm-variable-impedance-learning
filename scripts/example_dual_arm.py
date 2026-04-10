#!/usr/bin/env python3
"""
Example usage of the DualArmPandaEnv

This demonstrates how to:
1. Create the environment
2. Reset and get observations
3. Apply actions and step the simulation
4. Access state information
"""

import numpy as np
from davil.envs import DualArmPandaEnv


def example_basic_usage():
    """Basic environment usage"""
    print("Creating environment...")
    env = DualArmPandaEnv(render_mode=None)
    
    # Reset to initial state
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial EE positions:")
    print(f"  Panda1: {info['panda1_ee_pos']}")
    print(f"  Panda2: {info['panda2_ee_pos']}")
    print(f"  Object: {info['object_pos']}")
    
    # Step the environment with a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nAfter one step:")
    print(f"  Panda1 EE: {info['panda1_ee_pos']}")
    print(f"  Panda2 EE: {info['panda2_ee_pos']}")
    print(f"  Reward: {reward}")
    
    env.close()


def example_controlled_motion():
    """Example of controlled joint motion"""
    print("\n" + "="*60)
    print("Controlled Motion Example")
    print("="*60)
    
    env = DualArmPandaEnv(render_mode=None)
    obs, info = env.reset()
    
    # Create an action that moves panda1 joint 1 forward
    action = np.zeros(16)
    action[0] = 0.5  # Move panda1 joint 1 in positive direction
    action[8] = -0.5  # Move panda2 joint 1 in negative direction (opposite)
    
    print("Executing controlled action...")
    for step in range(10):
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"Step {step}:")
            print(f"  Panda1 EE: {info['panda1_ee_pos']}")
            print(f"  Panda2 EE: {info['panda2_ee_pos']}")
    
    env.close()


def example_observation_decomposition():
    """Show how to decompose observations"""
    print("\n" + "="*60)
    print("Observation Decomposition Example")
    print("="*60)
    
    env = DualArmPandaEnv(render_mode=None)
    obs, info = env.reset()
    
    # Extract components from observation
    panda1_q = obs[0:7]
    panda1_qd = obs[7:14]
    panda1_ee_pos = obs[14:17]
    panda1_ee_quat = obs[17:21]
    
    panda2_q = obs[21:28]
    panda2_qd = obs[28:35]
    panda2_ee_pos = obs[35:38]
    panda2_ee_quat = obs[38:42]
    
    obj_pos = obs[42:45]
    obj_quat = obs[45:49]
    obj_lin_vel = obs[49:52]
    obj_ang_vel = obs[52:55]
    
    print(f"Panda1 joint positions: {panda1_q}")
    print(f"Panda1 joint velocities: {panda1_qd}")
    print(f"Panda1 EE position: {panda1_ee_pos}")
    print(f"Object position: {obj_pos}")
    
    env.close()


def example_trajectory():
    """Example of executing a simple trajectory"""
    print("\n" + "="*60)
    print("Trajectory Execution Example")
    print("="*60)
    
    env = DualArmPandaEnv(render_mode=None)
    obs, info = env.reset()
    
    # Define a simple trajectory: move joints to home position
    home_q1 = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
    home_q2 = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
    
    # Create action that tries to reach home (simple proportional control)
    current_q1 = obs[0:7]
    current_q2 = obs[21:28]
    
    action = np.zeros(16)
    # Simple P control: action = K * (desired_q - current_q)
    K = 0.1
    action[0:7] = K * (home_q1 - current_q1)
    action[8:15] = K * (home_q2 - current_q2)
    
    print("Executing homing trajectory...")
    for step in range(50):
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            q1_error = np.linalg.norm(home_q1 - obs[0:7])
            print(f"Step {step}: Q1 error = {q1_error:.4f}")
    
    env.close()


if __name__ == "__main__":
    print("DualArmPandaEnv Examples")
    print("========================\n")
    
    example_basic_usage()
    example_controlled_motion()
    example_observation_decomposition()
    example_trajectory()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
