#!/usr/bin/env python3
"""
Test script for dual-arm MuJoCo environment
Demonstrates basic usage and functionality
"""

import numpy as np
from davil.envs import DualArmPandaEnv


def test_dual_arm_env():
    """Test the dual-arm environment"""
    print("=" * 60)
    print("DUAL-ARM PANDA ENVIRONMENT TEST")
    print("=" * 60)
    
    # Create environment
    env = DualArmPandaEnv(render_mode=None)
    obs, info = env.reset()
    
    # Print environment info
    print(f"\n✓ Environment Created")
    print(f"  - Model: Dual-arm Franka Panda with grippers")
    print(f"  - Total DOF: {len(env.all_arm_joints)} (7 per arm)")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    
    # Observation breakdown
    print(f"\n✓ Observation Breakdown (Total: {len(obs)} dims)")
    print(f"  - Panda1 joint positions: 7")
    print(f"  - Panda1 joint velocities: 7")
    print(f"  - Panda1 end-effector position: 3")
    print(f"  - Panda1 end-effector quaternion: 4")
    print(f"  - Panda2 joint positions: 7")
    print(f"  - Panda2 joint velocities: 7")
    print(f"  - Panda2 end-effector position: 3")
    print(f"  - Panda2 end-effector quaternion: 4")
    print(f"  - Object position: 3")
    print(f"  - Object quaternion: 4")
    print(f"  - Object linear velocity: 3")
    print(f"  - Object angular velocity: 3")
    
    # Test initial state
    print(f"\n✓ Initial State:")
    print(f"  - Panda1 EE position: {info['panda1_ee_pos']}")
    print(f"  - Panda2 EE position: {info['panda2_ee_pos']}")
    print(f"  - Object position: {info['object_pos']}")
    
    # Run a few steps with random actions
    print(f"\n✓ Running 10 simulation steps with random actions...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step == 9:
            print(f"\n  Step {step + 1}:")
            print(f"    - Panda1 EE pos: {info['panda1_ee_pos']}")
            print(f"    - Panda2 EE pos: {info['panda2_ee_pos']}")
            print(f"    - Object pos: {info['object_pos']}")
            print(f"    - Reward: {reward}")
            print(f"    - Time: {info['time']:.3f}s")
    
    # Test reset
    print(f"\n✓ Testing reset...")
    obs, info = env.reset()
    print(f"  - Reset successful")
    print(f"  - Object position after reset: {info['object_pos']}")
    
    # Test custom state setting
    print(f"\n✓ Testing custom state setting...")
    home_config = np.zeros(len(env.data.qpos))
    env.set_state(home_config)
    obs = env._get_observation()
    print(f"  - Set custom state successful")
    print(f"  - Observation shape: {obs.shape}")
    
    env.close()
    
    print(f"\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_dual_arm_env()
