"""
Unit tests for kinematics module
"""

import pytest
import numpy as np
from davil.kinematics.franka_kinematics import FrankaKinematics

def test_forward_kinematics_zero_config():
    """Test FK at zero configuration"""
    fk = FrankaKinematics()
    q = np.zeros(7)
    
    T = fk.forward_kinematics(q)
    pos, quat = fk.get_pose(T)
    
    # At zero config, end-effector should be at specific position
    assert T.shape == (4, 4)
    assert pos.shape == (3,)
    assert quat.shape == (4,)
    
    # Z position should be sum of link offsets
    expected_z = 0.333 + 0.316 + 0.384 + 0.107
    assert np.abs(pos[2] - expected_z) < 0.01

def test_jacobian_shape():
    """Test Jacobian has correct shape"""
    fk = FrankaKinematics()
    q = np.random.uniform(-1, 1, 7)
    
    J = fk.compute_jacobian(q)
    
    assert J.shape == (6, 7)

def test_joint_limits():
    """Test joint limit enforcement"""
    fk = FrankaKinematics()
    
    # Test limits
    assert len(fk.q_min) == 7
    assert len(fk.q_max) == 7
    assert np.all(fk.q_min < fk.q_max)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
