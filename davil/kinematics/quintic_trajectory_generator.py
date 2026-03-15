import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class QuinticTrajectory:
    """Quintic polynomial for smooth position/orientation trajectories."""
    
    def __init__(self, waypoints, times, orientations=None):
        """
        Args:
            waypoints: list of 3D positions [[x1,y1,z1], [x2,y2,z2], ...]
            times: list of times [t0, t1, t2, ...] (must be increasing)
            orientations: list of Rotation objects (optional)
        """
        self.waypoints = np.array(waypoints)
        self.times = np.array(times)
        self.n_segments = len(times) - 1
        
        # Compute quintic coefficients per segment
        self.coeffs = []
        for i in range(self.n_segments):
            p0, p1 = self.waypoints[i], self.waypoints[i+1]
            t0, t1 = self.times[i], self.times[i+1]
            dt = t1 - t0
            
            # Quintic: p(τ) = a0 + a1*τ + a2*τ² + a3*τ³ + a4*τ⁴ + a5*τ⁵
            # Boundary: p(0)=p0, p(1)=p1, v(0)=0, v(1)=0, a(0)=0, a(1)=0
            a0 = p0
            a1 = np.zeros(3)
            a2 = np.zeros(3)
            a3 = 10 * (p1 - p0)
            a4 = -15 * (p1 - p0)
            a5 = 6 * (p1 - p0)
            
            self.coeffs.append((a0, a1, a2, a3, a4, a5, dt))
        
        # Orientation interpolation
        if orientations is not None:
            self.slerp = Slerp(times, Rotation.concatenate(orientations))
        else:
            self.slerp = None
    
    def eval(self, t):
        """Returns (pos, vel, acc) at time t."""
        if t <= self.times[0]:
            return self.waypoints[0], np.zeros(3), np.zeros(3)
        if t >= self.times[-1]:
            return self.waypoints[-1], np.zeros(3), np.zeros(3)
        
        # Find segment
        idx = np.searchsorted(self.times, t) - 1
        idx = np.clip(idx, 0, self.n_segments - 1)
        
        t0 = self.times[idx]
        a0, a1, a2, a3, a4, a5, dt = self.coeffs[idx]
        
        # Normalized time τ ∈ [0,1]
        tau = (t - t0) / dt
        tau2, tau3, tau4, tau5 = tau**2, tau**3, tau**4, tau**5
        
        pos = a0 + a1*tau + a2*tau2 + a3*tau3 + a4*tau4 + a5*tau5
        vel = (a1 + 2*a2*tau + 3*a3*tau2 + 4*a4*tau3 + 5*a5*tau4) / dt
        acc = (2*a2 + 6*a3*tau + 12*a4*tau2 + 20*a5*tau3) / (dt**2)
        
        return pos, vel, acc
    
    def eval_orientation(self, t):
        """Returns (R, omega, alpha) if orientations provided."""
        if self.slerp is None:
            return np.eye(3), np.zeros(3), np.zeros(3)
        
        R = self.slerp(t).as_matrix()
        
        # Numerical derivatives for omega, alpha
        eps = 1e-4
        R_plus = self.slerp(min(t + eps, self.times[-1])).as_matrix()
        R_minus = self.slerp(max(t - eps, self.times[0])).as_matrix()
        
        dR = (R_plus - R_minus) / (2 * eps)
        omega_skew = dR @ R.T
        omega = np.array([omega_skew[2,1], omega_skew[0,2], omega_skew[1,0]])
        
        return R, omega, np.zeros(3)  # alpha approximated as 0
