"""
run_trajectory_plot.py  [MUJOCO GEOM API FIXED]
================================================

FIXES vs previous version:
  1. geom.label  — is a plain Python str in MuJoCo ≥3.x, not a byte buffer.
                   Fixed: assign empty string  geom.label = ""
  2. geom.mat    — does not exist on mjvGeom. Orientation is set via
                   geom.mat[:] only if the field is a numpy array.
                   Fixed: use mujoco.mjv_initGeom() which handles all
                   field initialisation safely, then set only what we need.
  3. Plots saved to a dedicated  plots/  subfolder.

WHAT YOU SEE IN THE MUJOCO WINDOW:
  Yellow spheres  = waypoint corners (fixed in space)
  Blue cylinders  = planned square edges
  Red sphere      = moving reference point the arm is chasing
  Green dots      = fading EE trail (recent path of actual EE)
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
from scipy.spatial.transform import Rotation

from franka_kinematics import FrankaKinematics
from quintic_trajectory_generator import QuinticTrajectory

# ── Config ────────────────────────────────────────────────────
PANDA_XML_PATH = os.path.join("..", "..", "docs", "mujoco_menagerie",
                              "franka_emika_panda", "panda.xml")
PLOT_EVERY   = 50
SETTLE_TIME  = 2.0
TRAIL_LENGTH = 300   # number of past EE positions to show

# Gains
KP     = 3000.0
KD_X   = 2 * np.sqrt(KP * 47)   # 751
KD_Y   = 2 * np.sqrt(KP * 12)   # 379
KD_Z   = 2 * np.sqrt(KP * 10)   # 346
KP_ORI = 80.0
KD_ORI = 20.0

# Output folder for plots
PLOTS_DIR = "plots"
# ─────────────────────────────────────────────────────────────


def ensure_plots_dir():
    """Create the plots/ directory if it doesn't exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"  Plots will be saved to: {os.path.abspath(PLOTS_DIR)}/")


def get_joint_indices(model):
    jids    = [model.joint(f"joint{i+1}").id for i in range(7)]
    dof_ids = np.array([np.where(model.dof_jntid == j)[0][0] for j in jids])
    act_ids = np.array([np.where(model.actuator_trnid[:, 0] == j)[0][0] for j in jids])
    return dof_ids, act_ids


# ── Controller ────────────────────────────────────────────────
class CartesianPDAnisotropic:
    def __init__(self, Kp, Kd_x, Kd_y, Kd_z, Kp_ori, Kd_ori, alpha=0.15):
        self.Kp_pos = np.eye(3) * Kp
        self.Kd_pos = np.diag([Kd_x, Kd_y, Kd_z])
        self.Kp_ori = np.eye(3) * Kp_ori
        self.Kd_ori = np.eye(3) * Kd_ori
        self._J_prev = None
        self._alpha  = alpha

    def compute_wrench(self, x_des, R_des, v_des, omega_des,
                             x_cur, R_cur, v_cur, omega_cur):
        e_pos  = x_des - x_cur
        F      = self.Kp_pos @ e_pos + self.Kd_pos @ (v_des - v_cur)
        R_err  = R_des @ R_cur.T
        rotvec = Rotation.from_matrix(R_err).as_rotvec()
        M      = self.Kp_ori @ rotvec + self.Kd_ori @ (omega_des - omega_cur)
        return np.concatenate([F, M])

    def filter_jacobian(self, J):
        if self._J_prev is None:
            self._J_prev = J.copy(); return J.copy()
        J_filt = self._alpha * self._J_prev + (1.0 - self._alpha) * J
        self._J_prev = J_filt.copy()
        return J_filt


# ══════════════════════════════════════════════════════════════
# MUJOCO IN-VIEWER DRAWING
# ══════════════════════════════════════════════════════════════

def _make_rotation_matrix(direction):
    """
    Build a 3x3 rotation matrix whose Z column points along `direction`.
    Used to orient cylinders drawn as path lines.
    """
    z = np.array(direction, dtype=float)
    length = np.linalg.norm(z)
    if length < 1e-9:
        return np.eye(3)
    z /= length

    # Pick arbitrary X perpendicular to Z
    if abs(z[0]) < 0.9:
        x = np.cross(z, [1.0, 0.0, 0.0])
    else:
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])   # columns = X, Y, Z axes


def draw_sphere(scene, pos, radius, rgba, idx):
    """
    Write a sphere geom into scene.geoms[idx].
    Returns idx+1 (next free slot), or idx if scene is full.
    """
    if idx >= scene.maxgeom:
        return idx

    g = scene.geoms[idx]

    # mjv_initGeom initialises ALL fields to safe defaults.
    # This is the correct way to start — avoids the label/mat crash.
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.zeros(3),   # size (will override below)
        np.zeros(3),   # pos  (will override below)
        np.eye(3).flatten(),   # rotation matrix, row-major
        np.array(rgba, dtype=np.float32),
    )

    # Override size and position after init
    g.size[:] = [radius, radius, radius]
    g.pos[:]  = pos

    return idx + 1


def draw_cylinder(scene, p1, p2, radius, rgba, idx):
    """
    Draw a cylinder from p1 to p2.
    MuJoCo cylinders are specified by centre + half-length + orientation.
    Returns idx+1.
    """
    if idx >= scene.maxgeom:
        return idx

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    diff   = p2 - p1
    length = np.linalg.norm(diff)
    if length < 1e-6:
        return idx

    mid    = (p1 + p2) / 2.0
    R      = _make_rotation_matrix(diff)      # 3x3, Z → along cylinder axis
    R_flat = R.flatten().astype(np.float64)

    g = scene.geoms[idx]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, radius, length / 2.0]),  # size: r, r, half-length
        mid,
        R_flat,
        np.array(rgba, dtype=np.float32),
    )

    return idx + 1


def update_mujoco_visuals(scene, wp, x_des, x_cur, trail):
    """
    Rebuild all custom geometry in the MuJoCo scene every frame.

    Called BEFORE viewer.sync() so the viewer picks up the new geoms.
    We zero ngeom first so stale geoms from the previous frame
    don't bleed through if we write fewer geoms this frame.
    """
    scene.ngeom = 0
    n = 0

    # 1. Waypoint corners — yellow spheres
    for w in wp:
        n = draw_sphere(scene, w,
                        radius=0.012,
                        rgba=[1.0, 0.85, 0.0, 0.9],
                        idx=n)

    # 2. Planned square edges — blue cylinders
    for i in range(len(wp) - 1):
        n = draw_cylinder(scene, wp[i], wp[i + 1],
                          radius=0.003,
                          rgba=[0.2, 0.45, 1.0, 0.6],
                          idx=n)

    # 3. Current reference point — bright red sphere (moves along path)
    n = draw_sphere(scene, x_des,
                    radius=0.016,
                    rgba=[1.0, 0.1, 0.1, 1.0],
                    idx=n)

    # 4. EE trail — green spheres fading from bright (recent) to dim (old)
    trail_list = list(trail)
    num = len(trail_list)
    for i, pos in enumerate(trail_list):
        alpha = 0.08 + 0.75 * (i / max(num - 1, 1))
        n = draw_sphere(scene, pos,
                        radius=0.005,
                        rgba=[0.1, 0.9, 0.2, float(alpha)],
                        idx=n)

    scene.ngeom = n


# ══════════════════════════════════════════════════════════════
# MATPLOTLIB
# ══════════════════════════════════════════════════════════════

def setup_plots(wp, traj):
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Trajectory Controller — Live EE Data", fontsize=13,
                 fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: XY path
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("EE Path — top-down XY view")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.set_aspect('equal')
    plan_t  = np.linspace(0, 8.0, 400)
    plan_xy = np.array([traj.eval(t)[0][:2] for t in plan_t])
    ax1.plot(plan_xy[:, 0], plan_xy[:, 1], 'b--', lw=1.2,
             alpha=0.5, label='planned')
    for i, w in enumerate(wp):
        ax1.plot(w[0], w[1], 'bs', markersize=7, zorder=5)
        ax1.annotate(f'W{i}', (w[0], w[1]),
                     textcoords='offset points', xytext=(4, 4),
                     fontsize=7, color='steelblue')
    lpath, = ax1.plot([], [], 'r-',  lw=1.8, label='actual')
    ldot,  = ax1.plot([], [], 'ro',  markersize=8, zorder=10)
    ax1.legend(fontsize=7)

    # Panel 2: Tracking error
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Tracking Error (mm)")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("||x_des - x_cur|| (mm)")
    lerr, = ax2.plot([], [], 'purple', lw=1.5)
    for t_wp in [0, 2, 4, 6, 8]:
        ax2.axvline(t_wp, color='gray', lw=0.7, ls=':', alpha=0.7)
    ax2.axvspan(-SETTLE_TIME, 0, alpha=0.08, color='green',
                label=f'settle ({SETTLE_TIME}s)')
    ax2.legend(fontsize=7)

    # Panel 3: Speed
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("EE Speed (m/s)")
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("||v_cur|| (m/s)")
    lspd, = ax3.plot([], [], 'darkorange', lw=1.2)
    for t_wp in [0, 2, 4, 6, 8]:
        ax3.axvline(t_wp, color='gray', lw=0.7, ls=':', alpha=0.7)

    # Panel 4: X and Y desired vs actual
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("X & Y: Desired vs Actual (m)")
    ax4.set_xlabel("time (s)"); ax4.set_ylabel("position (m)")
    lxdes, = ax4.plot([], [], 'b--', lw=1.0, label='X des')
    lxact, = ax4.plot([], [], 'b-',  lw=1.5, label='X act')
    lydes, = ax4.plot([], [], 'g--', lw=1.0, label='Y des')
    lyact, = ax4.plot([], [], 'g-',  lw=1.5, label='Y act')
    ax4.legend(fontsize=7)

    plt.show(block=False)
    plt.pause(0.05)

    return fig, (ax1, ax2, ax3, ax4), (lpath, ldot, lerr, lspd,
                                        lxdes, lxact, lydes, lyact)


def update_plots(fig, axes, lines, log):
    ax1, ax2, ax3, ax4 = axes
    lpath, ldot, lerr, lspd, lxdes, lxact, lydes, lyact = lines

    t      = np.array(log['t'])
    x_hist = np.array(log['x_cur'])
    e_hist = np.array(log['pos_err_mm'])
    v_hist = np.array(log['speed'])
    xd_h   = np.array(log['x_des'])

    if len(t) < 2:
        return

    mask = t >= 0
    if mask.any():
        lpath.set_data(x_hist[mask, 0], x_hist[mask, 1])
        ldot.set_data([x_hist[-1, 0]], [x_hist[-1, 1]])

    lerr.set_data(t, e_hist)
    ax2.set_xlim(-SETTLE_TIME, 8.5)
    ax2.set_ylim(0, max(e_hist.max() * 1.1, 5))

    lspd.set_data(t, v_hist)
    ax3.set_xlim(-SETTLE_TIME, 8.5)
    ax3.relim(); ax3.autoscale_view()

    lxdes.set_data(t, xd_h[:, 0])
    lxact.set_data(t, x_hist[:, 0])
    lydes.set_data(t, xd_h[:, 1])
    lyact.set_data(t, x_hist[:, 1])
    ax4.relim(); ax4.autoscale_view()

    fig.canvas.draw_idle()
    plt.pause(0.001)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    ensure_plots_dir()

    model = mujoco.MjModel.from_xml_path(PANDA_XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    fk         = FrankaKinematics()
    controller = CartesianPDAnisotropic(
        Kp=KP, Kd_x=KD_X, Kd_y=KD_Y, Kd_z=KD_Z,
        Kp_ori=KP_ORI, Kd_ori=KD_ORI, alpha=0.15,
    )

    q_init        = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    data.qpos[:7] = q_init
    mujoco.mj_forward(model, data)

    T_home          = fk.forward_kinematics(q_init)
    x_home, q_home_ = fk.get_pose(T_home)
    R_home          = Rotation.from_quat(q_home_).as_matrix()

    print(f"  Home EE  : {x_home.round(4)}")
    print(f"  Kp       : {KP}")
    print(f"  Kd X/Y/Z : {KD_X:.0f} / {KD_Y:.0f} / {KD_Z:.0f}")
    print(f"  Settle   : {SETTLE_TIME}s")
    print()
    print("  MuJoCo window:")
    print("    Yellow spheres = waypoint corners")
    print("    Blue lines     = planned path")
    print("    Red sphere     = moving reference (arm chases this)")
    print("    Green dots     = actual EE trail")
    print()

    wp = [
        x_home.copy(),
        x_home + np.array([0.15, 0.00, 0.0]),
        x_home + np.array([0.15, 0.15, 0.0]),
        x_home + np.array([0.00, 0.15, 0.0]),
        x_home.copy(),
    ]
    times = [0.0, 2.0, 4.0, 6.0, 8.0]
    traj  = QuinticTrajectory(waypoints=wp, times=times)

    dof_ids, act_ids = get_joint_indices(model)
    fig, axes, lines = setup_plots(wp, traj)

    log = {'t': [], 'x_cur': [], 'pos_err_mm': [], 'speed': [], 'x_des': []}
    trail = deque(maxlen=TRAIL_LENGTH)

    sim_time   = -SETTLE_TIME
    step_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and sim_time <= times[-1] + 0.5:

            mujoco.mj_forward(model, data)
            q  = data.qpos[dof_ids].copy()
            qd = data.qvel[dof_ids].copy()

            T_cur              = fk.forward_kinematics(q)
            x_cur, quat_cur    = fk.get_pose(T_cur)
            R_cur              = Rotation.from_quat(quat_cur).as_matrix()

            J      = fk.jacobian(q)
            J_filt = controller.filter_jacobian(J)
            xdot   = J_filt @ qd
            v_cur  = xdot[:3]
            w_cur  = xdot[3:]

            if sim_time < 0.0:
                x_des = x_home.copy()
                v_des = np.zeros(3)
            else:
                x_des, v_des, _ = traj.eval(sim_time)

            wrench    = controller.compute_wrench(
                x_des, R_home, v_des, np.zeros(3),
                x_cur, R_cur,  v_cur, w_cur,
            )
            tau_pd    = J_filt.T @ wrench
            tau_bias  = data.qfrc_bias[dof_ids].copy()
            tau_total = tau_pd + tau_bias

            data.ctrl[:] = 0.0
            data.ctrl[act_ids] = np.clip(tau_total, -87.0, 87.0)
            mujoco.mj_step(model, data)

            # Add to trail every 5 steps once trajectory has started
            if step_count % 5 == 0 and sim_time >= 0.0:
                trail.append(x_cur.copy())

            # Draw into MuJoCo scene BEFORE viewer.sync()
            update_mujoco_visuals(viewer.user_scn, wp, x_des, x_cur, trail)

            viewer.sync()

            # Log data
            log['t'].append(sim_time)
            log['x_cur'].append(x_cur.copy())
            log['pos_err_mm'].append(np.linalg.norm(x_des - x_cur) * 1000)
            log['speed'].append(np.linalg.norm(v_cur))
            log['x_des'].append(x_des.copy())

            if step_count % PLOT_EVERY == 0:
                update_plots(fig, axes, lines, log)

            sim_time   += dt
            step_count += 1

            for t_wp in [0.0, 2.0, 4.0, 6.0, 8.0]:
                if abs(sim_time - t_wp) < dt * 1.5:
                    err = np.linalg.norm(x_des - x_cur) * 1000
                    print(f"  t={t_wp:.0f}s  err={err:.1f}mm  "
                          f"x_cur={x_cur.round(3)}")

    # ── Save final plot to plots/ folder ──────────────────────
    update_plots(fig, axes, lines, log)
    plt.ioff()

    out_path = os.path.join(PLOTS_DIR, "trajectory_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()