#!/usr/bin/env python3
"""
Visualize Franka Panda forward kinematics
Week 1 visualization tool
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from davil.kinematics.franka_kinematics import FrankaKinematics

def plot_robot_pose(ax, T, label='', color='blue'):
    """Plot robot end-effector pose"""
    pos = T[:3, 3]
    R = T[:3, :3]
    
    # Plot position
    ax.scatter(*pos, s=100, c=color, label=label, zorder=5)
    
    # Plot coordinate frame
    scale = 0.05
    frame_colors = ['r', 'g', 'b']
    for i, color in enumerate(frame_colors):
        direction = R[:, i] * scale
        ax.quiver(pos[0], pos[1], pos[2], 
                 direction[0], direction[1], direction[2],
                 color=color, arrow_length_ratio=0.3, linewidth=2)

def main():
    print("Generating Franka Panda kinematics visualization...")
    
    fk = FrankaKinematics()
    
    # Test configurations
    configs = {
        'Zero': np.zeros(7),
        'Home': np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]),
        'Reach Up': np.array([0, -np.pi/6, 0, -2*np.pi/3, 0, np.pi/3, 0]),
        'Reach Side': np.array([np.pi/2, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, 0]),
    }
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (name, q) in enumerate(configs.items()):
        print(f"Computing FK for {name} configuration...")
        T = fk.forward_kinematics(q)
        pos, quat = fk.get_pose(T)
        
        print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        
        plot_robot_pose(ax, T, label=name, color=colors[i])
    
    # Workspace bounds
    ax.set_xlim([-0.5, 0.8])
    ax.set_ylim([-0.5, 0.6])
    ax.set_zlim([0, 1.0])
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Franka Panda End-Effector Workspace\n(4 Configurations)', fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect([1, 0.8, 1])
    
    plt.tight_layout()
    
    # Save plot
    import os
    os.makedirs('logs', exist_ok=True)
    plt.savefig('logs/week1_kinematics.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved visualization: logs/week1_kinematics.png")
    
    plt.show()

if __name__ == "__main__":
    main()
