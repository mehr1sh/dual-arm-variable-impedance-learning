"""
Forward Kinematics and Jacobian for Franka Emika Panda (7-DOF)
"""

import numpy as np
from scipy.spatial.transform import Rotation

class FrankaKinematics:
    """
    Forward kinematics for Franka Emika Panda robot
    Using Denavit-Hartenberg parameters
    """
    
    def __init__(self):
        # DH parameters for Franka Panda: [a, alpha, d, theta_offset]
        # Source: Franka Emika documentation
        self.dh_params = np.array([
            [0,      0,         0.333,  0],      # Joint 1
            [0,     -np.pi/2,   0,      0],      # Joint 2
            [0,      np.pi/2,   0.316,  0],      # Joint 3
            [0.0825, np.pi/2,   0,      0],      # Joint 4
            [-0.0825, -np.pi/2, 0.384,  0],      # Joint 5
            [0,      np.pi/2,   0,      0],      # Joint 6
            [0.088,  np.pi/2,   0.107,  0],      # Joint 7
        ])
        
        # Joint limits (radians)
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, 
                               -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 
                               2.8973, 3.7525, 2.8973])
        
        # Velocity limits (rad/s)
        self.qd_max = np.array([2.175, 2.175, 2.175, 2.175, 
                                2.610, 2.610, 2.610])
    
    def dh_transform(self, a, alpha, d, theta):
        """
        Compute 4x4 transformation matrix from DH parameters
        
        Args:
            a: Link length
            alpha: Link twist
            d: Link offset
            theta: Joint angle
        
        Returns:
            T: 4x4 homogeneous transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])
        
        return T
    
    def forward_kinematics(self, joint_angles):
        """
        Compute end-effector pose from joint angles
        
        Args:
            joint_angles: numpy array of shape (7,) in radians
        
        Returns:
            T: 4x4 transformation matrix (end-effector pose)
        """
        assert len(joint_angles) == 7, "Franka Panda has 7 joints"
        
        T = np.eye(4)
        
        for i, (a, alpha, d, offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
        
        return T
    
    def get_pose(self, T):
        """
        Extract position and orientation from transformation matrix
        
        Args:
            T: 4x4 transformation matrix
        
        Returns:
            position: (3,) array [x, y, z]
            quaternion: (4,) array [x, y, z, w]
        """
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        
        # Convert to quaternion
        r = Rotation.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # Returns [x, y, z, w]
        
        return position, quaternion
    
    def compute_jacobian(self, joint_angles, delta=1e-6):
        """
        Compute 6x7 Jacobian matrix using numerical differentiation
        
        J = ∂x/∂q where x is end-effector pose (6D: position + orientation)
        
        Args:
            joint_angles: (7,) array
            delta: Step size for numerical differentiation
        
        Returns:
            J: (6, 7) Jacobian matrix
        """
        J = np.zeros((6, 7))
        
        # Reference pose
        T0 = self.forward_kinematics(joint_angles)
        pos0, quat0 = self.get_pose(T0)
        
        for i in range(7):
            # Perturb joint i
            q_plus = joint_angles.copy()
            q_plus[i] += delta
            
            T_plus = self.forward_kinematics(q_plus)
            pos_plus, quat_plus = self.get_pose(T_plus)
            
            # Position Jacobian (linear velocity)
            J[:3, i] = (pos_plus - pos0) / delta
            
            # Orientation Jacobian (angular velocity)
            # Approximate using quaternion difference
            J[3:, i] = (quat_plus[:3] - quat0[:3]) / delta
        
        return J
    
    def inverse_kinematics_numerical(self, target_pose, q_init=None, 
                                    max_iter=100, tol=1e-4):
        """
        Simple numerical IK using Jacobian pseudo-inverse
        
        Args:
            target_pose: Dict with 'position' and 'orientation' (quaternion)
            q_init: Initial joint angles
            max_iter: Maximum iterations
            tol: Convergence tolerance
        
        Returns:
            q: Joint angles that reach target (or best attempt)
            success: Whether IK converged
        """
        if q_init is None:
            q_init = np.zeros(7)
        
        q = q_init.copy()
        
        for iteration in range(max_iter):
            # Current pose
            T = self.forward_kinematics(q)
            pos, quat = self.get_pose(T)
            
            # Pose error
            pos_error = target_pose['position'] - pos
            
            # Simplified orientation error
            quat_target = target_pose['orientation']
            ori_error = quat_target[:3] - quat[:3]
            
            # 6D error
            error = np.concatenate([pos_error, ori_error])
            
            # Check convergence
            if np.linalg.norm(error) < tol:
                return q, True
            
            # Compute Jacobian
            J = self.compute_jacobian(q)
            
            # Pseudo-inverse
            J_pinv = np.linalg.pinv(J)
            
            # Update joints
            dq = J_pinv @ error
            q = q + 0.1 * dq  # Step size = 0.1
            
            # Enforce joint limits
            q = np.clip(q, self.q_min, self.q_max)
        
        return q, False


def test_forward_kinematics():
    """Test FK implementation"""
    fk = FrankaKinematics()
    
    # Test configurations
    configs = {
        'zero': np.zeros(7),
        'home': np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]),
        'random': np.random.uniform(-1, 1, 7)
    }
    
    print("Testing Forward Kinematics:\n")
    
    for name, q in configs.items():
        T = fk.forward_kinematics(q)
        pos, quat = fk.get_pose(T)
        
        print(f"{name.capitalize()} configuration:")
        print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"  Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        print()


def test_jacobian():
    """Test Jacobian computation"""
    fk = FrankaKinematics()
    
    q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
    
    print("Testing Jacobian Computation:\n")
    print(f"Joint angles: {q}")
    
    J = fk.compute_jacobian(q)
    
    print(f"\nJacobian shape: {J.shape}")
    print(f"Jacobian (6x7):\n{J}")
    
    # Test manipulability
    manipulability = np.sqrt(np.linalg.det(J @ J.T))
    print(f"\nManipulability index: {manipulability:.6f}")


if __name__ == "__main__":
    print("="*60)
    print(" Week 1: Franka Kinematics Test")
    print("="*60 + "\n")
    
    test_forward_kinematics()
    test_jacobian()
    
    print("="*60)
    print("✅ Week 1 Kinematics Complete!")
    print("="*60)
