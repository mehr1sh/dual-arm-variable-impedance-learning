"""
run_impedance_plot.py
=====================
Same as run_impedance.py but with LIVE MATPLOTLIB PLOTS.

EXTRA vs PD plot:
  - Subplot 2 shows position error for BOTH PD and Impedance overlaid
    so you can directly compare convergence speed (you need to run PD
    first and it saves pd_error_log.npy for comparison)
  - Otherwise identical 4-panel layout
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation

from franka_kinematics import FrankaKinematics
from cartesian_controllers import CartesianImpedanceController

# ── Config ────────────────────────────────────────────────────
PANDA_XML_PATH    = os.path.join("..", "..", "docs", "mujoco_menagerie",
                                 "franka_emika_panda", "panda.xml")
EE_X_OFFSET       = 0.15
GOAL_RAMP_SECONDS = 2.0
PLOT_EVERY        = 50
# ─────────────────────────────────────────────────────────────


def get_joint_indices(model):
    jids    = [model.joint(f"joint{i+1}").id for i in range(7)]
    dof_ids = np.array([np.where(model.dof_jntid == j)[0][0] for j in jids])
    act_ids = np.array([np.where(model.actuator_trnid[:, 0] == j)[0][0] for j in jids])
    return dof_ids, act_ids


def setup_plots(has_pd_comparison):
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Cartesian Impedance Controller — Live EE Data", fontsize=13,
                 fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("EE Position (m)")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("position (m)")
    lx,  = ax1.plot([], [], 'r-',  lw=1.5, label='X')
    ly,  = ax1.plot([], [], 'g-',  lw=1.5, label='Y')
    lz,  = ax1.plot([], [], 'b-',  lw=1.5, label='Z')
    lgx, = ax1.plot([], [], 'r--', lw=0.8, alpha=0.5, label='X goal')
    ax1.legend(fontsize=7, loc='lower right')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Position Error (mm) — Impedance vs PD")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("||error|| (mm)")
    lerr, = ax2.plot([], [], color='steelblue', lw=1.5, label='Impedance')
    lpd   = None
    if has_pd_comparison:
        lpd, = ax2.plot([], [], color='tomato', lw=1.2, ls='--',
                        alpha=0.7, label='PD (prev run)')
    ax2.axvspan(0, GOAL_RAMP_SECONDS, alpha=0.08, color='orange',
                label=f'ramp ({GOAL_RAMP_SECONDS}s)')
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("EE Speed (m/s)")
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("||v_cur|| (m/s)")
    lspd, = ax3.plot([], [], 'darkorange', lw=1.5)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Desired vs Actual X (m)")
    ax4.set_xlabel("time (s)"); ax4.set_ylabel("X position (m)")
    lxdes, = ax4.plot([], [], 'b--', lw=1.2, label='x_des')
    lxact, = ax4.plot([], [], 'r-',  lw=1.5, label='x_cur')
    ax4.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.05)

    axes  = (ax1, ax2, ax3, ax4)
    lines = (lx, ly, lz, lgx, lerr, lpd, lspd, lxdes, lxact)
    return fig, axes, lines


def update_plots(fig, axes, lines, log, x_goal, pd_log=None):
    ax1, ax2, ax3, ax4 = axes
    lx, ly, lz, lgx, lerr, lpd, lspd, lxdes, lxact = lines

    t      = np.array(log['t'])
    x_hist = np.array(log['x_cur'])
    e_hist = np.array(log['pos_err_mm'])
    v_hist = np.array(log['speed'])
    xd_hist= np.array(log['x_des_x'])

    if len(t) < 2:
        return

    lx.set_data(t, x_hist[:, 0])
    ly.set_data(t, x_hist[:, 1])
    lz.set_data(t, x_hist[:, 2])
    lgx.set_data([t[0], t[-1]], [x_goal[0], x_goal[0]])
    ax1.relim(); ax1.autoscale_view()

    lerr.set_data(t, e_hist)
    if lpd is not None and pd_log is not None:
        pt = np.array(pd_log['t'])
        pe = np.array(pd_log['pos_err_mm'])
        lpd.set_data(pt, pe)
    ax2.set_xlim(0, max(t[-1], 1.0))
    ax2.set_ylim(0, max(e_hist.max() * 1.1, 10))

    lspd.set_data(t, v_hist)
    ax3.relim(); ax3.autoscale_view()

    lxdes.set_data(t, xd_hist)
    lxact.set_data(t, x_hist[:, 0])
    ax4.relim(); ax4.autoscale_view()

    fig.canvas.draw_idle()
    plt.pause(0.001)


def main():
    model = mujoco.MjModel.from_xml_path(PANDA_XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    fk = FrankaKinematics()
    controller = CartesianImpedanceController(
        K_d=400.0,    # softer than PD (800) — still beats gravity, yields more
        D_d=260.0,    # ≈ 2*sqrt(400*47)
        K_ori=40.0,
        D_ori=12.0,
    )

    q_init        = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    data.qpos[:7] = q_init
    mujoco.mj_forward(model, data)

    T_init            = fk.forward_kinematics(q_init)
    x_init, quat_init = fk.get_pose(T_init)
    R_init            = Rotation.from_quat(quat_init).as_matrix()

    x_goal = x_init.copy(); x_goal[0] += EE_X_OFFSET
    R_goal = R_init.copy()

    print(f"  x_init : {x_init.round(4)}")
    print(f"  x_goal : {x_goal.round(4)}\n")

    dof_ids, act_ids = get_joint_indices(model)

    # Load PD comparison data if it exists from a previous run
    pd_log = None
    pd_log_path = "pd_error_log.npy"
    if os.path.exists(pd_log_path):
        pd_log = np.load(pd_log_path, allow_pickle=True).item()
        print(f"  Loaded PD comparison data from {pd_log_path}")

    fig, axes, lines = setup_plots(has_pd_comparison=(pd_log is not None))

    log = {'t': [], 'x_cur': [], 'pos_err_mm': [], 'speed': [], 'x_des_x': []}

    sim_time   = 0.0
    step_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

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

            α     = min(sim_time / GOAL_RAMP_SECONDS, 1.0)
            x_des = x_init + α * (x_goal - x_init)
            v_des = (x_goal - x_init) / GOAL_RAMP_SECONDS if α < 1.0 else np.zeros(3)

            wrench    = controller.compute_wrench(
                x_des, R_goal, v_des, np.zeros(3),
                x_cur, R_cur,  v_cur, w_cur,
            )
            tau_imp   = J_filt.T @ wrench
            tau_bias  = data.qfrc_bias[dof_ids].copy()
            tau_total = tau_imp + tau_bias

            data.ctrl[:] = 0.0
            data.ctrl[act_ids] = np.clip(tau_total, -87.0, 87.0)
            mujoco.mj_step(model, data)
            viewer.sync()

            log['t'].append(sim_time)
            log['x_cur'].append(x_cur.copy())
            log['pos_err_mm'].append(np.linalg.norm(x_des - x_cur) * 1000)
            log['speed'].append(np.linalg.norm(v_cur))
            log['x_des_x'].append(x_des[0])

            if step_count % PLOT_EVERY == 0:
                update_plots(fig, axes, lines, log, x_goal, pd_log)

            sim_time   += dt
            step_count += 1

    update_plots(fig, axes, lines, log, x_goal, pd_log)
    plt.ioff()
    os.makedirs("plots", exist_ok=True)
    out = os.path.join("plots", "impedance_controller_plot.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()