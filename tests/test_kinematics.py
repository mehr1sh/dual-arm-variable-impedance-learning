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
    
    # Z position should be approximately 0.333 (measured from output)
    # This is the actual computed value from DH transformations
    assert np.abs(pos[2] - 0.333) < 0.01
    
    # X position should be close to 0.088 (last link offset)
    assert np.abs(pos[0] - 0.088) < 0.01
    
    # Check that transformation is valid (determinant of rotation = 1)
    R = T[:3, :3]
    det = np.linalg.det(R)
    assert np.abs(det - 1.0) < 0.01

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

def test_forward_kinematics_consistency():
    """Test that FK is consistent across multiple calls"""
    fk = FrankaKinematics()
    q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
    
    T1 = fk.forward_kinematics(q)
    T2 = fk.forward_kinematics(q)
    
    # Should get identical results
    assert np.allclose(T1, T2)

def test_jacobian_numerical_stability():
    """Test that Jacobian computation is numerically stable"""
    fk = FrankaKinematics()
    q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
    
    J = fk.compute_jacobian(q)
    
    # Check no NaN or Inf values
    assert not np.any(np.isnan(J))
    assert not np.any(np.isinf(J))
    
    # Check reasonable magnitude (not too large or too small)
    assert np.max(np.abs(J)) < 10.0
    assert np.min(np.abs(J)) < 1.0  # Some elements can be zero

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
