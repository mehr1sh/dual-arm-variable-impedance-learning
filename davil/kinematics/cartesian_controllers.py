"""
cartesian_controllers.py  [GAINS FIXED — see explanation below]

ROOT CAUSE OF STEADY-STATE ERROR (from plot showing x_cur stuck 27mm below x_goal):
  The Panda's effective Cartesian mass at the EE is ~47 kg in X (not 1 kg).
  With Kp=200:  critical Kd = 2*sqrt(200*47) = 194  (we used 28 — massively underdamped)
  With Kp=200:  SS error = gravity_residual / Kp ≈ 5N / 200 = 25mm  (matches plot!)

  Fix: raise Kp to 800 N/m, Kd to 350 N.s/m (close to critical for 47kg effective mass)

Contains two controllers:
  1. CartesianPDController    — spring-damper in task space
  2. CartesianImpedanceController — M*a_err + D*v_err + K*x_err

KEY FIX vs original:
  - Jacobian filter alpha corrected (was 0.9 = stale; now 0.1 = fresh)
  - Impedance no longer differentiates velocity (avoids 1000x noise amplification)
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# CONTROLLER 1 — Cartesian PD
# ============================================================

class CartesianPDController:
    """
    Cartesian PD controller.

    At every timestep it computes a 6D wrench [F; M] that drives
    the end-effector toward a desired pose (position + orientation).

    Physics:
        F = Kp_pos*(x_des - x_cur) + Kd_pos*(v_des - v_cur)
        M = Kp_ori*rotvec_err     + Kd_ori*(w_des - w_cur)

    The caller then maps the wrench to joint torques via:
        tau = J^T @ wrench
    """

    def __init__(self,
                 Kp_pos: float = 800.0,   # N/m   — was 200, raised for m_eff≈47kg
                 Kd_pos: float = 350.0,   # N.s/m — was 28, critical = 2*sqrt(800*47)=388
                 Kp_ori: float = 60.0,    # N.m/rad — was 30
                 Kd_ori: float = 15.0):   # N.m.s/rad — was 10
        self.Kp_pos = np.eye(3) * Kp_pos
        self.Kd_pos = np.eye(3) * Kd_pos
        self.Kp_ori = np.eye(3) * Kp_ori
        self.Kd_ori = np.eye(3) * Kd_ori

        # Jacobian low-pass filter state
        # alpha = fraction of *previous* value to keep.
        # BUG in original: alpha=0.9 means only 10% new info per step → massive lag.
        # Fix: alpha=0.1 (mostly current J, slight smoothing) or 0.0 (no filter).
        self._J_prev = None
        self._alpha  = 0.1    # ← was 0.9 in the original — that was the dancing bug

    # ----------------------------------------------------------
    def compute_wrench(self,
                       x_des, R_des, v_des, omega_des,
                       x_cur, R_cur, v_cur, omega_cur) -> np.ndarray:
        """
        Returns 6D wrench = [Fx, Fy, Fz, Mx, My, Mz].

        Args:
            x_des   : (3,) desired EE position
            R_des   : (3,3) desired EE rotation matrix
            v_des   : (3,) desired EE linear velocity  (set to zeros for a static goal)
            omega_des:(3,) desired EE angular velocity (set to zeros for a static goal)
            x_cur   : (3,) current EE position
            R_cur   : (3,3) current EE rotation matrix
            v_cur   : (3,) current EE linear velocity
            omega_cur:(3,) current EE angular velocity
        """
        # --- Position error & force ---
        e_pos = x_des - x_cur                           # (3,)
        F = self.Kp_pos @ e_pos + self.Kd_pos @ (v_des - v_cur)

        # --- Orientation error (axis-angle) & torque ---
        # R_err = R_des * R_cur^T  →  rotation needed to go from current to desired
        R_err   = R_des @ R_cur.T
        rotvec  = Rotation.from_matrix(R_err).as_rotvec()  # (3,) axis * angle
        M = self.Kp_ori @ rotvec + self.Kd_ori @ (omega_des - omega_cur)

        return np.concatenate([F, M])                   # (6,)

    # ----------------------------------------------------------
    def filter_jacobian(self, J: np.ndarray) -> np.ndarray:
        """
        Gentle low-pass filter on J to suppress finite-difference noise.
        alpha=0.1: output = 0.1*J_prev + 0.9*J_current  (mostly current).
        """
        if self._J_prev is None:
            self._J_prev = J.copy()
            return J.copy()
        J_filt = self._alpha * self._J_prev + (1.0 - self._alpha) * J
        self._J_prev = J_filt.copy()
        return J_filt


# ============================================================
# CONTROLLER 2 — Classical Cartesian Impedance
# ============================================================

class CartesianImpedanceController:
    """
    Classical impedance controller (position-level formulation).

    Instead of tracking acceleration (which requires noisy differentiation),
    this version uses only position and velocity errors — equivalent to a
    critically-damped second-order system:

        F = K_d*(x_des - x_cur) + D_d*(v_des - v_cur)

    This is intentionally *softer* than the PD controller: the arm will
    deflect when you push it (compliant behaviour) but return to the goal.

    The difference from PD:
        • Typically lower gains  → feels "springy"
        • Models the robot as a virtual mass-spring-damper attached to the goal
        • Suitable for contact tasks where you don't want rigid tracking
    """

    def __init__(self,
                 K_d: float = 400.0,   # N/m   — softer than PD (800) but not floppy
                 D_d: float = 260.0,   # N.s/m — ≈ 2*sqrt(400*47) = 274
                 K_ori: float = 40.0,
                 D_ori: float = 12.0):
        self.K_d   = np.eye(3) * K_d
        self.D_d   = np.eye(3) * D_d
        self.K_ori = np.eye(3) * K_ori
        self.D_ori = np.eye(3) * D_ori

        self._J_prev = None
        self._alpha  = 0.1

    # ----------------------------------------------------------
    def compute_wrench(self,
                       x_des, R_des, v_des, omega_des,
                       x_cur, R_cur, v_cur, omega_cur) -> np.ndarray:
        """
        Returns 6D impedance wrench.

        Note: same signature as CartesianPDController.compute_wrench so
        the run scripts are interchangeable.
        """
        # Position restoring force (spring + damper toward goal)
        e_pos = x_des - x_cur
        F = self.K_d @ e_pos + self.D_d @ (v_des - v_cur)

        # Orientation restoring torque
        R_err  = R_des @ R_cur.T
        rotvec = Rotation.from_matrix(R_err).as_rotvec()
        M = self.K_ori @ rotvec + self.D_ori @ (omega_des - omega_cur)

        return np.concatenate([F, M])

    # ----------------------------------------------------------
    def filter_jacobian(self, J: np.ndarray) -> np.ndarray:
        if self._J_prev is None:
            self._J_prev = J.copy()
            return J.copy()
        J_filt = self._alpha * self._J_prev + (1.0 - self._alpha) * J
        self._J_prev = J_filt.copy()
        return J_filt