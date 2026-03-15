import numpy as np
from scipy.spatial.transform import Rotation


class FrankaKinematics:
    """7-DOF Franka Panda FK + Jacobian (position + orientation)."""

    def __init__(self):
        # EXACTLY 7 rows for Week 2 assertions
        self.dh_params = np.array([
            [0.0, 0.0, 0.333, 0.0],
            [0.0, -np.pi/2, 0.0, 0.0],
            [0.0, np.pi/2, 0.316, 0.0],
            [0.0825, np.pi/2, 0.0, 0.0],
            [-0.0825, -np.pi/2, 0.384, 0.0],
            [0.0, np.pi/2, 0.0, 0.0],
            [0.088, np.pi/2, 0.0, 0.0],
        ])
        
        # Flange + hand (separate)
        self.hand_offset = np.array([
            [0.0, 0.0, 0.107, 0.0],
            [0.0, 0.0, 0.0, -np.pi/4],
        ])
        
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        print(f"FrankaKinematics: {len(self.dh_params)} joints")

    @staticmethod
    def dh_transform(a, alpha, d, theta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -sa*d],
            [st*sa, ct*sa, ca, ca*d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, q):
        """Returns 4x4 transform of hand frame."""
        q = np.asarray(q, dtype=float)
        assert len(q) == 7
        
        T = np.eye(4)
        for i in range(7):
            a, alpha, d, offset = self.dh_params[i]
            T = T @ self.dh_transform(a, alpha, d, q[i] + offset)
        
        for i in range(2):
            a, alpha, d, offset = self.hand_offset[i]
            T = T @ self.dh_transform(a, alpha, d, offset)
        
        return T

    @staticmethod
    def get_pose(T):
        """Returns (pos, quat) from 4x4 transform."""
        pos = T[:3, 3]
        R = Rotation.from_matrix(T[:3, :3])
        quat = R.as_quat()  # [x,y,z,w]
        return pos, quat

    def jacobian(self, q, eps=1e-6):
        """6x7 Jacobian [v_linear; v_angular] via finite differences."""
        q = np.asarray(q, dtype=float)
        J = np.zeros((6, 7))
        
        T0 = self.forward_kinematics(q)
        p0, quat0 = self.get_pose(T0)
        R0 = Rotation.from_quat(quat0).as_matrix()
        
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            
            # Position Jacobian
            T_plus = self.forward_kinematics(q + dq)
            p_plus, _ = self.get_pose(T_plus)
            J[:3, i] = (p_plus - p0) / eps
            
            # Orientation Jacobian (angular velocity)
            T_minus = self.forward_kinematics(q - dq)
            _, quat_plus = self.get_pose(T_plus)
            _, quat_minus = self.get_pose(T_minus)
            
            R_plus = Rotation.from_quat(quat_plus).as_matrix()
            R_minus = Rotation.from_quat(quat_minus).as_matrix()
            
            # Finite difference angular velocity
            dR = (R_plus - R_minus) / (2 * eps)
            omega_skew = dR @ R0.T
            J[3:, i] = [omega_skew[2,1], omega_skew[0,2], omega_skew[1,0]]
        
        return J
