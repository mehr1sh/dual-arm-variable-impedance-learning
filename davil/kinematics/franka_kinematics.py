import numpy as np
from scipy.spatial.transform import Rotation

class FrankaKinematics:
    """
    Updated Forward kinematics for Franka Emika Panda robot.
    Now uses Modified DH (Craig) parameters to match MuJoCo ground truth.
    """
    
    def __init__(self):
        # Modified DH parameters: [a_{i-1}, alpha_{i-1}, d_i, theta_offset]
        # a: distance from z_{i-1} to z_i along x_{i-1}
        # alpha: angle from z_{i-1} to z_i about x_{i-1}
        # d: distance from x_{i-1} to x_i along z_i
        # theta: angle from x_{i-1} to x_i about z_i
        self.dh_params = np.array([
            [0,       0,         0.333,  0],      # Joint 1
            [0,      -np.pi/2,   0,      0],      # Joint 2
            [0,       np.pi/2,   0.316,  0],      # Joint 3
            [0.0825,  np.pi/2,   0,      0],      # Joint 4
            [-0.0825, -np.pi/2,  0.384,  0],      # Joint 5
            [0,       np.pi/2,   0,      0],      # Joint 6
            [0.088,   np.pi/2,   0,      0],      # Joint 7
        ])
        
        # Hand offset relative to link 7: [x, y, z]
        # Note: link7 to hand includes a Z translation of 0.107m 
        # and a 45-degree rotation around Z (as seen in your XML).
        self.hand_offset_z = 0.165
        
        # Joint limits (radians)
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, 
                               -2.8973, -0.0175, -2.8973])
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 
                               2.8973, 3.7525, 2.8973])
    
    def dh_transform(self, a_prev, alpha_prev, d_curr, theta_curr):
        """
        Compute 4x4 transformation matrix using MODIFIED DH parameters.
        Order: Rot_x(alpha_{i-1}) * Trans_x(a_{i-1}) * Rot_z(theta_i) * Trans_z(d_i)
        """
        ca = np.cos(alpha_prev)
        sa = np.sin(alpha_prev)
        ct = np.cos(theta_curr)
        st = np.sin(theta_curr)
        
        T = np.array([
            [ct,       -st,      0,      a_prev],
            [st*ca,    ct*ca,   -sa,    -sa*d_curr],
            [st*sa,    ct*sa,    ca,     ca*d_curr],
            [0,         0,       0,      1]
        ])
        return T
    
    def forward_kinematics(self, joint_angles):
        """
        Compute end-effector (hand) pose from joint angles.
        """
        assert len(joint_angles) == 7, "Franka Panda has 7 joints"
        
        T = np.eye(4)
        
        # Step through the 7 joints using Modified DH
        for i, (a_prev, alpha_prev, d_curr, offset) in enumerate(self.dh_params):
            theta_curr = joint_angles[i] + offset
            T_i = self.dh_transform(a_prev, alpha_prev, d_curr, theta_curr)
            T = T @ T_i
            
        # Final offset to the hand (from your panda.xml)
        # body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834"
        # The rotation is -45 degrees about Z
        T_hand = np.eye(4)
        T_hand[2, 3] = self.hand_offset_z
        
        # Apply the -45 degree rotation for the hand frame alignment
        angle = -np.pi/4
        R_z = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle),  np.cos(angle), 0, 0],
            [0,              0,             1, 0],
            [0,              0,             0, 1]
        ])
        
        return T @ T_hand @ R_z

    def get_pose(self, T):
        """ Extracts [x, y, z] and [x, y, z, w] quaternion. """
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        r = Rotation.from_matrix(rotation_matrix)
        quaternion = r.as_quat()
        return position, quaternion