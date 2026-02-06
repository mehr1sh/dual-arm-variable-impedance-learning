"""
Week 1: Simple Trajectory Generator + Visualization
Quintic polynomial trajectories and SLERP orientations
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimpleTrajectoryGenerator:
    """
    Generate and visualize basic trajectories for Week 1
    """
    
    def __init__(self, dt=0.01, T=2.0):
        self.dt = dt
        self.T = T  # Total duration
        self.num_steps = int(T / dt)
    
    def quintic_position(self, p0, pf):
        """
        Generate 3D quintic polynomial trajectory
        
        p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        
        Args:
            p0: Start position [x,y,z]
            pf: End position [x,y,z]
        
        Returns:
            positions: (N, 3) array
            velocities: (N, 3) array
            accelerations: (N, 3) array
        """
        positions = np.zeros((self.num_steps, 3))
        velocities = np.zeros((self.num_steps, 3))
        accelerations = np.zeros((self.num_steps, 3))
        
        for i in range(3):  # For each axis
            coeffs = self._solve_quintic_coeffs(p0[i], pf[i])
            
            for step in range(self.num_steps):
                t = step * self.dt
                
                # Position, velocity, acceleration
                pos = (coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + 
                       coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5)
                
                vel = (coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 
                       4*coeffs[4]*t**3 + 5*coeffs[5]*t**4)
                
                acc = (2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t**2 + 
                       20*coeffs[5]*t**3)
                
                positions[step, i] = pos
                velocities[step, i] = vel
                accelerations[step, i] = acc
        
        return positions, velocities, accelerations
    
    def _solve_quintic_coeffs(self, p0, pf):
        """Solve quintic polynomial coefficients"""
        T = self.T
        
        # Boundary conditions: p(0)=p0, p(T)=pf, v(0)=v(T)=a(0)=a(T)=0
        A = np.array([
            [1, 0, 0, 0, 0, 0],      # p(0) = p0
            [0, 1, 0, 0, 0, 0],      # v(0) = 0
            [0, 0, 2, 0, 0, 0],      # a(0) = 0
            [1, T, T**2, T**3, T**4, T**5],  # p(T) = pf
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],  # v(T) = 0
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]   # a(T) = 0
        ])
        
        b = np.array([p0, 0, 0, pf, 0, 0])
        return np.linalg.solve(A, b)
    
    def slerp_orientation(self, quat_start, quat_end):
        """
        Spherical Linear Interpolation (SLERP) for orientations
        
        Args:
            quat_start: [x,y,z,w] start quaternion
            quat_end: [x,y,z,w] end quaternion
        
        Returns:
            quaternions: (N, 4) array
        """
        # Create keyframe times and rotations
        key_times = [0, self.T]
        key_rots = Rotation.concatenate([Rotation.from_quat(quat_start), 
                                        Rotation.from_quat(quat_end)])
        
        # Create SLERP interpolator
        slerp = Slerp(key_times, key_rots)
        
        # Sample trajectory
        times = np.linspace(0, self.T, self.num_steps)
        interpolated_rots = slerp(times)
        
        return interpolated_rots.as_quat()
    
    def visualize_trajectory(self, start_pose, end_pose, save_path='logs/week1_trajectory.png'):
        """
        Visualize complete SE(3) trajectory
        
        Args:
            start_pose: Dict {'position': [x,y,z], 'orientation': [x,y,z,w]}
            end_pose: Dict {'position': [x,y,z], 'orientation': [x,y,z,w]}
        """
        print("Generating trajectory visualization...")
        
        # Generate position trajectory
        pos_traj, vel_traj, acc_traj = self.quintic_position(
            start_pose['position'], end_pose['position']
        )
        
        # Generate orientation trajectory
        quat_traj = self.slerp_orientation(
            start_pose['orientation'], end_pose['orientation']
        )
        
        # Create visualization
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Position trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], 'b-', linewidth=3)
        ax1.scatter([start_pose['position'][0]], [start_pose['position'][1]], 
                   [start_pose['position'][2]], color='green', s=200, label='Start')
        ax1.scatter([end_pose['position'][0]], [end_pose['position'][1]], 
                   [end_pose['position'][2]], color='red', s=200, label='End')
        ax1.set_title('Position Trajectory\n(Quintic Polynomial)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Velocities
        ax2 = fig.add_subplot(132)
        ax2.plot(np.arange(len(vel_traj)), vel_traj[:, 0], label='Vx', linewidth=2)
        ax2.plot(np.arange(len(vel_traj)), vel_traj[:, 1], label='Vy', linewidth=2)
        ax2.plot(np.arange(len(vel_traj)), vel_traj[:, 2], label='Vz', linewidth=2)
        ax2.set_title('Velocity Profile')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accelerations
        ax3 = fig.add_subplot(133)
        ax3.plot(np.arange(len(acc_traj)), acc_traj[:, 0], label='Ax', linewidth=2)
        ax3.plot(np.arange(len(acc_traj)), acc_traj[:, 1], label='Ay', linewidth=2)
        ax3.plot(np.arange(len(acc_traj)), acc_traj[:, 2], label='Az', linewidth=2)
        ax3.set_title('Acceleration Profile')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Week 1: Smooth Trajectory Generation\nQuintic Polynomial + SLERP', fontsize=16)
        plt.tight_layout()
        
        # Save
        import os
        os.makedirs('logs', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.show()
        
        return pos_traj, quat_traj

def demo_pick_place():
    """Demo pick-and-place trajectory"""
    generator = SimpleTrajectoryGenerator(dt=0.01, T=2.0)
    
    # Start pose (object on table)
    start_pose = {
        'position': np.array([0.5, 0.0, 0.2]),
        'orientation': np.array([0, 0, 0, 1])  # Identity quaternion
    }
    
    # End pose (object picked up)
    end_pose = {
        'position': np.array([0.5, 0.0, 0.6]),
        'orientation': np.array([0, 0, 0, 1])
    }
    
    # Generate and visualize
    generator.visualize_trajectory(start_pose, end_pose)
    
    print("\n✅ Week 1 Trajectory Visualization Complete!")
    print("Features:")
    print("- Smooth quintic polynomial (zero velocity/acceleration boundaries)")
    print("- SLERP orientation interpolation")
    print("- Position, velocity, acceleration profiles")

if __name__ == "__main__":
    demo_pick_place()
