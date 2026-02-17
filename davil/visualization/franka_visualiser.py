#!/usr/bin/env python3

# A helper class that uses Matplotlib to draw the arm. It calculates the 3D position 
# of every joint (not just the end-effector) using your DH parameters to create 
# a "stick" model.


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Import your FK
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kinematics.franka_kinematics import FrankaKinematics


class PandaVisualizer:
    def __init__(self):
        self.fk = FrankaKinematics()
    
    def compute_arm_positions(self, q):
        """Compute 3D positions of all 8 points: base + 7 joints."""
        T = np.eye(4)
        positions = [T[:3, 3].copy()]  # Base
        
        for i, (a, alpha, d, offset) in enumerate(self.fk.dh_params):
            theta = q[i] + offset
            T_i = self.fk.dh_transform(a, alpha, d, theta)
            T = T @ T_i
            positions.append(T[:3, 3].copy())
        
        return np.array(positions)
    
    def plot_single_config(self, q, ax, title=""):
        """Plot arm configuration on given axis."""
        positions = self.compute_arm_positions(q)
        
        # Plot links
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'o-', linewidth=4, markersize=8, color='steelblue', 
                markerfacecolor='orange', markeredgecolor='black', markeredgewidth=2)
        
        # End-effector marker
        ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
                   s=200, c='red', marker='*', edgecolors='black', linewidths=2,
                   label='End-Effector', zorder=10)
        
        # Base marker
        ax.scatter([0], [0], [0], s=150, c='green', marker='s', 
                   edgecolors='black', linewidths=2, label='Base')
        
        # Get FK position
        T = self.fk.forward_kinematics(q)
        pos, _ = self.fk.get_pose(T)
        
        # Labels
        ax.set_xlabel('X [m]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y [m]', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z [m]', fontsize=11, fontweight='bold')
        ax.set_title(f"{title}\nEE: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                     fontsize=12, fontweight='bold')
        
        # Equal aspect ratio
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([0, 1.0])
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    def visualize_configs(self, save_path='logs/franka_configs.png'):
        """Generate side-by-side plots of key configurations."""
        configs = {
            'Zero': np.zeros(7),
            'Home': np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]),
            'Reach': np.array([0.5, -1.0, 0.5, -2.0, 0, 1.2, 0.5]),
        }
        
        fig = plt.figure(figsize=(15, 5))
        
        for idx, (name, q) in enumerate(configs.items()):
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            self.plot_single_config(q, ax, title=name)
            ax.view_init(elev=20, azim=45)
        
        plt.suptitle('Franka Panda Configurations from YOUR FK', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.show()
        
        return fig


def main():
    """Demo visualization."""
    print("="*60)
    print(" VISUALIZING PANDA ARM FROM YOUR FK")
    print("="*60 + "\n")
    
    viz = PandaVisualizer()
    viz.visualize_cnonfigs()
    
    print("\nVisualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
