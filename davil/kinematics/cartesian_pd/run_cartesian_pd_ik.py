# # # Using joint position control
# # Forward kinematics:  x = FK(q)         maps joints → Cartesian
# # Inverse kinematics:  q = IK(x)         maps Cartesian → joints (what we need)

# we are converting the goal position to the desired joint angles using IK

import os
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from franka_kinematics import FrankaKinematics

PANDA_XML_PATH = os.path.join("..", "..", "..", "docs", "mujoco_menagerie", "franka_emika_panda", "panda.xml")

EE_X_OFFSET = 0.15      #in metres
GOAL_RAMP_SECONDS = 1.5     #in seconds
PLOT_EVERY = 50      #steps

# IK_solver parameters
IK_MAX_ITER = 150   #max newton steps per solve
IK_ALPHA = 0.5      #step size - smaller is more stable - learning rate
IK_TOL_POS = 1e-4   #metres - cpnvergnece threshold in position error
IK_TOL_ORI = 1e-3   #radians - convergence threshold in orientation error
IK_DAMP = 1e-4      #damping 

# joint index helpers- helps map different indexing systems
def get_joint_indices(model):

    #an f string is a formatted string 
    #range(7) generates numbers 0 to 6 
    #model.joint("joint1").id returns the numeric id of each joint therefore giving a list of joint ids 
    jids = [model.joint(f"joint{i+1}").id for i in range(7)]
    
    #loop over each joint id checking which dof belongs to which joint
    #model.dof_jntid==j creates a boolean array 
    #np.where return indices where the condition is true
    #[0][0] extracts the integer - gets the array of indices and then the first match 
    dof_ids = np.array([np.where(model.dof_jntid == j)[0][0] for j in jids])

    #now we convert joint ids to actuator indices 
    #model.actuator_trnid a table that tells us which actuator controls which joint 
    #basically an array that tells us which actuator controls which joint 
    act_ids = np.array([np.where(model.actuator_trnid[:, 0]==j)[0][0] for j in jids])

    return dof_ids, act_ids

#jacobian - pseudoinverse IK solver
def solve_ik(fk, q_init, x_goal, R_goal, 
             max_iter=IK_MAX_ITER, alpha=IK_ALPHA, 
             tol_pos=IK_TOL_POS, tol_ori=IK_TOL_ORI, 
             damp=IK_DAMP):
    
    #iterative ik via damped jacobian pseudoinverse
    # q_init is the initial joint configuration - numpy array angles
    # .copy() creates a new array (not a reference) and changing q will not change q_init
    q = q_init.copy()

    for iteration in range(max_iter):
        # take current joint angles and compute the transformation matrix T
        T_cur = fk.forward_kinematics(q)
        # position and quaternion 
        x_cur, quat_c = fk.get_pose(T_cur)
        # Rotation.from_quat is a scipy function that returns a rotation container
        # then .as_matrix() turns it into a rotation matrix
        R_cur = Rotation.from_quat(quat_c).as_matrix()

        # translational error
        e_pos = x_goal - x_cur

        # rotational error
        R_err = R_goal @ R_cur.T
        e_ori = Rotation.from_matrix(R_err).as_rotvec()

        # full 6D error
        e = np.concatenate([e_pos, e_ori])

        # np.linalg.norm computer magnitude of vector
        if np.linalg.norm(e_pos) < tol_pos and np.linalg.norm(e_ori) < tol_ori:
            break #IK has converged

        # Compute Jacobian 
        # J(q) maps joint velocities to EE velocity: ẋ = J(q) * q̇
        # Shape: (6, 7)  — 6 task-space DOF, 7 joint DOF
        J=fk.jacobian(q)

        JJT = J @ J.T
        # λ (damp) adds a small positive value to the diagonal of JJᵀ,
        # preventing it from becoming singular and bounding the size of Δ
        damped = JJT + damp*np.eye(6)
        # damped least squares
        J_dls = J.T @ np.linalg.inv(damped)

        # α < 1 shrinks the step for stability — prevents overshooting
        delta_q = alpha * J_dls @ e 
        q = q + delta_q

    return q 


def setup_plots():
    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("IK Joint Position Control — Live EE Data",
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
 
    # Plot 1: EE position XYZ
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("EE Position (m)")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("position (m)")
    lx,  = ax1.plot([], [], 'r-',  lw=1.5, label='X')
    ly,  = ax1.plot([], [], 'g-',  lw=1.5, label='Y')
    lz,  = ax1.plot([], [], 'b-',  lw=1.5, label='Z')
    lgx, = ax1.plot([], [], 'r--', lw=0.8, alpha=0.5, label='X goal')
    ax1.legend(fontsize=7, loc='lower right')
 
    # Plot 2: Cartesian position error
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Cartesian Position Error (mm)")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("||x_des - x_cur|| (mm)")
    lerr, = ax2.plot([], [], 'purple', lw=1.5)
    ax2.axvspan(0, GOAL_RAMP_SECONDS, alpha=0.08, color='orange',
                label=f'ramp ({GOAL_RAMP_SECONDS}s)')
    ax2.legend(fontsize=7)
 
    # Plot 3: EE speed
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("EE Speed (m/s)")
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("||v_cur|| (m/s)")
    lspd, = ax3.plot([], [], 'darkorange', lw=1.5)
 
    # Plot 4: Desired vs actual X
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Desired vs Actual X (m)")
    ax4.set_xlabel("time (s)"); ax4.set_ylabel("X position (m)")
    lxdes, = ax4.plot([], [], 'b--', lw=1.2, label='x_des')
    lxact, = ax4.plot([], [], 'r-',  lw=1.5, label='x_cur')
    ax4.legend(fontsize=7)
 
    # Plot 5: Joint angle tracking (q_desired vs q_actual) — 7 joints
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title("Joint Angle Tracking — q_desired (dashed) vs q_actual (solid)")
    ax5.set_xlabel("time (s)"); ax5.set_ylabel("joint angle (rad)")
    colors = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6']
    lq_act  = [ax5.plot([], [], '-',  color=colors[i], lw=1.2,
                         label=f'j{i+1}')[0] for i in range(7)]
    lq_des  = [ax5.plot([], [], '--', color=colors[i], lw=0.8,
                         alpha=0.6)[0] for i in range(7)]
    ax5.legend(fontsize=6, ncol=7, loc='upper right')
 
    plt.show(block=False)
    plt.pause(0.05)
 
    axes  = (ax1, ax2, ax3, ax4, ax5)
    lines = (lx, ly, lz, lgx, lerr, lspd, lxdes, lxact, lq_act, lq_des)
    return fig, axes, lines
 
 
def update_plots(fig, axes, lines, log, x_goal):
    ax1, ax2, ax3, ax4, ax5 = axes
    lx, ly, lz, lgx, lerr, lspd, lxdes, lxact, lq_act, lq_des = lines
 
    t      = np.array(log['t'])
    x_hist = np.array(log['x_cur'])       # (N, 3)
    e_hist = np.array(log['pos_err_mm'])  # (N,)
    v_hist = np.array(log['speed'])       # (N,)
    xd_hist= np.array(log['x_des_x'])    # (N,)
    qa_hist= np.array(log['q_actual'])    # (N, 7)
    qd_hist= np.array(log['q_desired'])  # (N, 7)
 
    if len(t) < 2:
        return
 
    lx.set_data(t, x_hist[:, 0])
    ly.set_data(t, x_hist[:, 1])
    lz.set_data(t, x_hist[:, 2])
    lgx.set_data([t[0], t[-1]], [x_goal[0], x_goal[0]])
    ax1.relim(); ax1.autoscale_view()
 
    lerr.set_data(t, e_hist)
    ax2.set_xlim(0, max(t[-1], 1.0))
    ax2.set_ylim(0, max(e_hist.max() * 1.1, 10))
 
    lspd.set_data(t, v_hist)
    ax3.relim(); ax3.autoscale_view()
 
    lxdes.set_data(t, xd_hist)
    lxact.set_data(t, x_hist[:, 0])
    ax4.relim(); ax4.autoscale_view()
 
    for i in range(7):
        lq_act[i].set_data(t, qa_hist[:, i])
        lq_des[i].set_data(t, qd_hist[:, i])
    ax5.relim(); ax5.autoscale_view()
 
    fig.canvas.draw_idle()
    plt.pause(0.001)


def main():
    # load model
    model = mujoco.MjModel.from_xml_path(PANDA_XML_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # FrankaKinematics uses the DH parameters of the Panda to compute
    # forward kinematics and the geometric Jacobian analytically.
    # We use it both inside the IK solver and to read x_cur each step.
    fk = FrankaKinematics()

    # put the arm in a known, initial position - ready or neutral configuration 
    q_init = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    # from 0 upto not including 7 = 0 to 6 
    data.qpos[:7] = q_init #set robot joints to that 
    # propagate q pos and all the other derived quantities - update the entire simulation 
    mujoco.mj_forward(model, data) 

    T_init = fk.forward_kinematics(q_init)
    x_init, quat_init = fk.get_pose(T_init)
    R_init = Rotation.from_quat(quat_init).as_matrix()

    x_goal = x_init.copy()
    x_goal[0]+= EE_X_OFFSET
    R_goal = R_init.copy()

    # round(4) rounds all values to 4 decimal places 
    print(f"  x_init   : {x_init.round(4)}")
    print(f"  x_goal   : {x_goal.round(4)}")
    print(f"  Δx       : {EE_X_OFFSET} m along X\n") 

    # we use offline IK - gradulaly reahcing the goal for smooh transition 
    # then once the goal is reached we can optionally use online IK if the world is not static 
    # or if we have to follow a moving target 
    print("  Solving IK for goal configuration...")
    q_goal = solve_ik(fk, q_init, x_goal, R_goal)

    T_check = fk.forward_kinematics(q_goal)
    # get pose extracts end-effector position and orientation but here we only need position 
    # so we ignore the quaternion orientation part 
    x_check, _ = fk.get_pose(T_check)
    # convert to mm
    ik_err = np.linalg.norm(x_check - x_goal) * 1000
    print(f"  IK converged:  q_goal = {q_goal.round(3)}")
    print(f"  IK residual:   {ik_err:.2f} mm\n")

    dof_ids, act_ids = get_joint_indices(model)
    fig, axes, lines = setup_plots()

    # logging dictionary to store data over time 
    log = {
        't':          [],
        'x_cur':      [],
        'pos_err_mm': [],
        'speed':      [],
        'x_des_x':    [],
        'q_actual':   [],
        'q_desired':  [],
    }

    sim_time = 0.0
    step_count = 0
    
    # in passive viewer we control the simulation manually- it jsut renders
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():  

            mujoco.mj_forward(model, data)
            q = data.qpos[dof_ids].copy()

            qd = data.qvel[dof_ids].copy()

            T_cur = fk.forward_kinematics(q)
            x_cur, quat_cur = fk.get_pose(T_cur)

            J = fk.jacobian(q)
            xdot = J @ qd
            v_cur = xdot[:3]

            alpha = min(sim_time / GOAL_RAMP_SECONDS, 1.0)
            q_desired = q_init + alpha * (q_goal - q_init)

            data.ctrl[:] = 0.0
            data.ctrl[act_ids] = q_desired

            mujoco.mj_step(model, data)
            viewer.sync()

            T_des = fk.forward_kinematics(q_desired)
            x_des, _ = fk.get_pose(T_des)

            log['t'].append(sim_time)
            log['x_cur'].append(x_cur.copy())
            log['pos_err_mm'].append(np.linalg.norm(x_des - x_cur) * 1000)
            log['speed'].append(np.linalg.norm(v_cur))
            log['x_des_x'].append(x_des[0])
            log['q_actual'].append(q.copy())
            log['q_desired'].append(q_desired.copy())

            if step_count % PLOT_EVERY ==0:
                update_plots(fig, axes, lines, log, x_goal)

            if abs(sim_time % 2.0) < dt * 1.5:
                err = np.linalg.norm(x_des - x_cur) * 1000
                print(f"  t={sim_time:5.1f}s  |  x_cur={x_cur.round(3)}"
                      f"  |  α={alpha:.2f}"
                      f"  |  err={err:.1f}mm")
 
            sim_time   += dt
            step_count += 1

    update_plots(fig, axes, lines, log, x_goal)
    plt.ioff()
    os.makedirs("plots", exist_ok=True)
    out = os.path.join("plots", "ik_position_control_plot.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out}")
    plt.show()
 
 
if __name__ == "__main__":
    main()