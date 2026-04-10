import os
import sys
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_kinematics import FrankaKinematics

# ── Paths & sim config ───────────────────────────────────────────────────────
PANDA_XML_PATH = os.path.join("..", "..", "..", "docs", "mujoco_menagerie",
                              "franka_emika_panda", "panda.xml")
EE_X_OFFSET = 0.15   # [m]
PLOT_EVERY  = 50

# ── Impedance gains ──────────────────────────────────────────────────────────
# These define the virtual spring-damper the robot emulates (slide 67, eq. 8).
# You choose B_d and K_d freely. M_d is NOT free — it must equal Λ(q)
# (the real task-space inertia) to eliminate the contact force term (slide 71).
#
# K_d : stiffness  [N/m]   — how hard the spring pulls toward goal
#        low  K_d → compliant, soft, drifts from goal under gravity
#        high K_d → stiff, snappy, fights disturbances
#
# B_d : damping    [N·s/m] — how quickly oscillation is absorbed
#        critical damping: B_d = 2*sqrt(K_d * m_eff)
#        with m_eff ≈ Λ eigenvalue ≈ 1–5 kg for Panda
#        B_d = 2*sqrt(350 * 2) ≈ 53  (conservative estimate)

K_D    = 350.0   # N/m     — translational spring
B_D    = 53.0    # N·s/m   — translational damping (critically damped for ~2kg eff mass)
K_ORI  = 40.0    # N·m/rad — rotational spring
B_ORI  = 12.0    # N·m·s   — rotational damping


# ── Joint index helpers ──────────────────────────────────────────────────────
def get_joint_indices(model):
    jids    = [model.joint(f"joint{i+1}").id for i in range(7)]
    dof_ids = np.array([np.where(model.dof_jntid == j)[0][0] for j in jids])
    act_ids = np.array([np.where(model.actuator_trnid[:, 0] == j)[0][0] for j in jids])
    return dof_ids, act_ids


# ── μ: task-space Coriolis + gravity ────────────────────────────────────────
def compute_mu(model, data, dof_ids, J, Jdot, qd, Lambda):
    """
    Compute μ = Λ(JM⁻¹h − J̇q̇)  [N]  — slide 69.

    This is the task-space equivalent of the joint-space gravity+Coriolis
    vector h = data.qfrc_bias. It appears in the Cartesian dynamics as the
    'bias force' that must be cancelled for the EE to move freely in task space.

    Args:
      J      : Jacobian (6×7) at current q
      Jdot   : time derivative of Jacobian (6×7)  — see compute_Jdot()
      qd     : joint velocities (7,)
      Lambda : task-space inertia matrix (6×6) = (JM⁻¹Jᵀ)⁻¹

    Returns:
      mu : (6,) task-space bias force [N, N·m]
    """
    # Joint-space gravity+Coriolis: h = M(q)q̈_gravity + C(q,q̇)q̇  [N·m]
    h = data.qfrc_bias[dof_ids].copy()    # shape (7,)

    # Joint-space inertia M(q): 7×7
    M_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_full, data.qM)
    M7 = M_full[np.ix_(dof_ids, dof_ids)]

    # JM⁻¹h − J̇q̇ : Cartesian bias acceleration (6,)
    M7_inv        = np.linalg.inv(M7)
    bias_cart_acc = J @ M7_inv @ h - Jdot @ qd   # (6,)

    # μ = Λ · bias_cart_acc  (project to task-space force)
    return Lambda @ bias_cart_acc   # shape (6,)


def compute_Lambda(model, data, dof_ids, J):
    """
    Λ = (JM⁻¹Jᵀ)⁻¹   task-space inertia matrix  [kg]

    This is the 'effective mass' the EE presents to the environment.
    It changes every timestep as the arm configuration changes.
    Setting M_d = Λ is the key trick from slide 71 that eliminates
    the need for a force/torque sensor.

    Returns:
      Lambda : (6×6) full task-space inertia
      Lambda3: (3×3) translational sub-block (for position control only)
    """
    M_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_full, data.qM)
    M7     = M_full[np.ix_(dof_ids, dof_ids)]
    M7_inv = np.linalg.inv(M7)
    Lambda = np.linalg.inv(J @ M7_inv @ J.T)      # (6×6)
    return Lambda


def compute_Jdot(fk, q, qd, dt=1e-4):
    """
    Numerical time derivative of Jacobian: J̇ ≈ (J(q + q̇·dt) − J(q)) / dt

    Needed to compute the Coriolis term J̇q̇ in μ.
    A small finite difference is used because analytic J̇ requires computing
    partial derivatives of the Jacobian w.r.t. q — tedious to implement.
    The step dt=1e-4 is small enough to be accurate and large enough to
    avoid numerical cancellation.
    """
    J_now  = fk.jacobian(q)
    J_next = fk.jacobian(q + qd * dt)
    return (J_next - J_now) / dt


# ── Live plot setup ──────────────────────────────────────────────────────────
def setup_plots():
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("True Impedance Controller (slides eq.12) — Live EE Data",
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("EE Position (m)")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("position (m)")
    lx,  = ax1.plot([], [], 'r-',  lw=1.5, label='X')
    ly,  = ax1.plot([], [], 'g-',  lw=1.5, label='Y')
    lz,  = ax1.plot([], [], 'b-',  lw=1.5, label='Z')
    lgx, = ax1.plot([], [], 'r--', lw=0.8, alpha=0.5, label='X goal')
    ax1.legend(fontsize=7, loc='lower right')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Position Error (mm)")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("||x_err|| (mm)")
    lerr, = ax2.plot([], [], color='steelblue', lw=1.5)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("EE Speed (m/s)")
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("||v_cur|| (m/s)")
    lspd, = ax3.plot([], [], 'darkorange', lw=1.5)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Desired vs Actual X (m)")
    ax4.set_xlabel("time (s)"); ax4.set_ylabel("X (m)")
    lxdes, = ax4.plot([], [], 'b--', lw=1.5, label='x_goal')
    lxact, = ax4.plot([], [], 'r-',  lw=1.5, label='x_cur')
    ax4.legend(fontsize=7)

    plt.show(block=False)
    plt.pause(0.05)
    return fig, (ax1, ax2, ax3, ax4), (lx, ly, lz, lgx, lerr, lspd, lxdes, lxact)


def update_plots(fig, axes, lines, log, x_goal):
    ax1, ax2, ax3, ax4 = axes
    lx, ly, lz, lgx, lerr, lspd, lxdes, lxact = lines

    t      = np.array(log['t'])
    x_hist = np.array(log['x_cur'])
    e_hist = np.array(log['pos_err_mm'])
    v_hist = np.array(log['speed'])
    if len(t) < 2:
        return

    lx.set_data(t, x_hist[:, 0]); ly.set_data(t, x_hist[:, 1])
    lz.set_data(t, x_hist[:, 2])
    lgx.set_data([t[0], t[-1]], [x_goal[0], x_goal[0]])
    ax1.relim(); ax1.autoscale_view()

    lerr.set_data(t, e_hist)
    ax2.set_xlim(0, max(t[-1], 1.0))
    ax2.set_ylim(0, max(e_hist.max() * 1.1, 10))

    lspd.set_data(t, v_hist); ax3.relim(); ax3.autoscale_view()

    lxdes.set_data([t[0], t[-1]], [x_goal[0], x_goal[0]])
    lxact.set_data(t, x_hist[:, 0]); ax4.relim(); ax4.autoscale_view()

    fig.canvas.draw_idle()
    plt.pause(0.001)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    model = mujoco.MjModel.from_xml_path(PANDA_XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    fk = FrankaKinematics()

    # ── Initial config & goal ────────────────────────────────────────────────
    q_init        = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    data.qpos[:7] = q_init
    mujoco.mj_forward(model, data)

    T_init            = fk.forward_kinematics(q_init)
    x_init, quat_init = fk.get_pose(T_init)
    R_init            = Rotation.from_quat(quat_init).as_matrix()

    x_goal = x_init.copy();  x_goal[0] += EE_X_OFFSET
    R_goal = R_init.copy()

    print(f"  x_init : {x_init.round(4)}")
    print(f"  x_goal : {x_goal.round(4)}")
    print(f"  K_d={K_D}  B_d={B_D}  (M_d=Λ, set automatically)")
    print(f"  Expected SS offset (gravity/K_d) ≈ {5.0/K_D*1000:.1f}mm — "
          f"this is CORRECT impedance behaviour\n")

    dof_ids, act_ids = get_joint_indices(model)
    fig, axes, lines = setup_plots()
    log = {'t': [], 'x_cur': [], 'pos_err_mm': [], 'speed': []}

    sim_time   = 0.0
    step_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            # ── 1. Read state ────────────────────────────────────────────────
            mujoco.mj_forward(model, data)
            q  = data.qpos[dof_ids].copy()   # joint positions  [rad]
            qd = data.qvel[dof_ids].copy()   # joint velocities [rad/s]

            # ── 2. FK: current EE pose ───────────────────────────────────────
            T_cur           = fk.forward_kinematics(q)
            x_cur, quat_cur = fk.get_pose(T_cur)
            R_cur           = Rotation.from_quat(quat_cur).as_matrix()

            # ── 3. Jacobian and EE velocity ──────────────────────────────────
            # J maps q̇ → ẋ:   ẋ = J·q̇      shape (6,7)
            J    = fk.jacobian(q)
            xdot = J @ qd               # task-space velocity (6,)
            v_cur = xdot[:3]            # linear velocity  [m/s]
            w_cur = xdot[3:]            # angular velocity [rad/s]

            # ── 4. Λ: task-space inertia  (slides eq. slide 69) ─────────────
            # Λ = (JM⁻¹Jᵀ)⁻¹  — the "effective mass" the EE presents
            # This is what we set M_d equal to (slide 71 trick).
            # It changes every timestep as configuration changes.
            Lambda = compute_Lambda(model, data, dof_ids, J)  # (6×6)

            # ── 5. J̇ and μ: task-space Coriolis+gravity (slides eq. slide 69)
            # μ = Λ(JM⁻¹h − J̇q̇)
            # This takes care of ALL gravity and Coriolis compensation.
            # We do NOT separately add tau_bias anywhere — μ replaces it.
            Jdot = compute_Jdot(fk, q, qd)                    # (6×7)
            mu   = compute_mu(model, data, dof_ids, J, Jdot, qd, Lambda)  # (6,)

            # ── 6. Translational impedance force  (slides eq. 12) ───────────
            # From slide 72:  f_m = −(B_d·ẋ + K_d·x) + μ
            # where x here means x_err = x_cur − x_goal  (displacement from equilibrium)
            #
            # NOTE: the sign convention in the slides uses x as displacement FROM
            # the equilibrium (goal), so x_err = x_cur − x_goal (positive when past goal).
            # The spring pulls back: −K_d·x_err = K_d·(x_goal − x_cur) ✓
            x_err  = x_cur - x_goal          # displacement from equilibrium [m]
            F_imp  = -K_D * x_err            # spring force  (pulls toward goal)
            F_damp = -B_D * v_cur            # damping force (opposes velocity)
            F_pos  = F_imp + F_damp + mu[:3] # total translational force [N]

            # ── 7. Rotational impedance (same structure) ─────────────────────
            R_err  = R_goal @ R_cur.T
            ori_err = Rotation.from_matrix(R_err).as_rotvec()   # axis-angle [rad]
            M_imp  = K_ORI * ori_err         # rotational spring
            M_damp = -B_ORI * w_cur          # rotational damper
            M_pos  = M_imp + M_damp + mu[3:] # total torque [N·m]

            # ── 8. Full 6D Cartesian force ───────────────────────────────────
            f_m = np.concatenate([F_pos, M_pos])   # (6,)  [N, N·m]

            # ── 9. Map to joint torques: τ = Jᵀ·f_m  (slides eq. 12) ────────
            # τ = Jᵀ(−K_d·x_err − B_d·ẋ + μ)
            #
            # This is the COMPLETE torque. No tau_bias addition needed —
            # gravity is already inside μ. Adding tau_bias would double-count.
            tau = J.T @ f_m                              # (7,)

            # ── 10. Send to actuators ────────────────────────────────────────
            data.ctrl[:] = 0.0
            data.ctrl[act_ids] = np.clip(tau, -87.0, 87.0)
            mujoco.mj_step(model, data)
            viewer.sync()

            # ── Logging ──────────────────────────────────────────────────────
            e_pos = x_goal - x_cur
            log['t'].append(sim_time)
            log['x_cur'].append(x_cur.copy())
            log['pos_err_mm'].append(np.linalg.norm(e_pos) * 1000)
            log['speed'].append(np.linalg.norm(v_cur))

            if step_count % PLOT_EVERY == 0:
                update_plots(fig, axes, lines, log, x_goal)

            if abs(sim_time % 2.0) < dt * 1.5:
                err = np.linalg.norm(e_pos) * 1000
                print(f"  t={sim_time:5.1f}s  |  x_cur={x_cur.round(3)}"
                      f"  |  err={err:.1f}mm")

            sim_time   += dt
            step_count += 1

    update_plots(fig, axes, lines, log, x_goal)
    plt.ioff()
    os.makedirs("plots", exist_ok=True)
    out = os.path.join("plots", "impedance_correct_plot.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()