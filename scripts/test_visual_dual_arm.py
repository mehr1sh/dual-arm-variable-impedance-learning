#!/usr/bin/env python3
"""Test script to visualize the dual-arm Panda environment."""

from davil.envs.dual_arm_env import DualArmPandaEnv
import time

def main():
    print("Creating dual-arm environment with visualization...")
    env = DualArmPandaEnv(render_mode='human')
    
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"✓ Environment reset successfully")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Object position: {obs[-3:]}")
    
    print("\nRunning simulation for 5 seconds with random actions...")
    print("Watch the viewport - you should see two Panda arms!")
    
    for i in range(50):  # 50 steps at ~100 Hz = 0.5 seconds
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 10 == 0:
            print(f"  Step {i}: EE1 pos={obs[8:11].round(3)}, EE2 pos={obs[36:39].round(3)}")
    
    print("\n✓ Test completed! If you saw the arms moving, visualization is working.")
    env.close()

if __name__ == "__main__":
    main()
