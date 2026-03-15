import os
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation

from franka_kinematics import FrankaKinematics

PANDA_XML_PATH = "../../docs/mujoco_menagerie/franka_emika_panda/panda.xml"

def main():
    model = mujoco.MjModel.from_xml_path(PANDA_XML_PATH)
    data = mujoco.MjData(model)
    
    fk = FrankaKinematics()
    
    # ===== CONFIGURATION =====
    q_init = np.zeros(7)
    T_init = fk.forward_kinematics(q_init)
    x_init, _ = fk.get_pose(T_init)
    
    x_goal = x_init.copy()
    x_goal[0] += 0.15
    
    print(f"q_init: {q_init}")
    print(f"x_init: {x_init}")
    print(f"x_goal: {x_goal}\n")
    
    # SIMPLER: Just manually set a different joint config
    # Skip IK - use known config that moves EE forward
    q_goal = np.array([0.5, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0])
    
    T_goal = fk.forward_kinematics(q_goal)
    x_check, _ = fk.get_pose(T_goal)
    print(f"q_goal: {q_goal}")
    print(f"x_goal (FK): {x_check}")
    print(f"Distance from init: {np.linalg.norm(x_check - x_init):.3f}m\n")
    
    # ===== MAP JOINTS =====
    joint_ids = [model.joint(f"joint{i+1}").id for i in range(7)]
    dof_ids = np.array([np.where(model.dof_jntid == j_id)[0][0] for j_id in joint_ids])
    act_ids = np.array([np.where(model.actuator_trnid[:, 0] == j_id)[0][0] for j_id in joint_ids])
    
    print(f"DOF indices: {dof_ids}")
    print(f"Actuator indices: {act_ids}\n")
    
    # STRONG gains
    Kp_joint = np.array([400, 800, 400, 600, 200, 200, 100]) 
    Kd_joint = np.array([80, 100, 80, 80, 40, 40, 20])
    
    duration = 3.0
    
    def smooth_trajectory(t, q0, q1, T):
        if t >= T:
            return q1, np.zeros(7), np.zeros(7)
        
        tau = np.clip(t / T, 0, 1)
        s = 6*tau**5 - 15*tau**4 + 10*tau**3
        ds = (30*tau**4 - 60*tau**3 + 30*tau**2) / T
        
        q = q0 + s * (q1 - q0)
        qd = ds * (q1 - q0)
        
        return q, qd, np.zeros(7)
    
    data.qpos[:7] = q_init
    mujoco.mj_forward(model, data)
    
    print("Starting motion...\n")
    
    sim_time = 0.0
    dt = model.opt.timestep
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        
        while viewer.is_running():  # ← No time limit
            viewer.sync()
            
            # Desired state: trajectory for first 3s, then hold
            if sim_time < duration:
                q_des, qd_des, _ = smooth_trajectory(sim_time, q_init, q_goal, duration)
            else:
                q_des = q_goal.copy()  # Hold at goal
                qd_des = np.zeros(7)
            
            # Current
            q_cur = data.qpos[dof_ids].copy()
            qd_cur = data.qvel[dof_ids].copy()
            
            # PD control
            e_q = q_des - q_cur
            e_qd = qd_des - qd_cur
            tau_pd = Kp_joint * e_q + Kd_joint * e_qd
            
            # Gravity comp
            mujoco.mj_forward(model, data)
            tau_bias = data.qfrc_bias[dof_ids].copy()
            tau_total = tau_bias + tau_pd
            
            # DEBUG PRINTS (every 0.5s)
            if step_count % 500 == 0:
                T_cur = fk.forward_kinematics(q_cur)
                x_cur, _ = fk.get_pose(T_cur)
                ee_err = np.linalg.norm(x_cur - x_init)  # Distance from start
                
                print(f"t={sim_time:.2f}s")
                print(f"  q_des: {q_des[:3]} ...")
                print(f"  q_cur: {q_cur[:3]} ...")
                print(f"  EE moved: {ee_err*1000:.1f}mm from start\n")
            
            # Apply
            data.ctrl[:] = 0.0
            data.ctrl[act_ids] = np.clip(tau_total, -87, 87)
            
            mujoco.mj_step(model, data)
            sim_time += dt
            step_count += 1

        print("Viewer closed by user")


if __name__ == "__main__":
    main()
