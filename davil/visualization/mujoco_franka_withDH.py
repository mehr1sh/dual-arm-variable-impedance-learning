# A verification script. It sets the MuJoCo robot to a pose and then draws a red 
# sphere at the $(x, y, z)$ coordinates calculated by your custom FrankaKinematics class.
#Screenshot of the MuJoCo viewer should show the red sphere exactly on the robot's fingertip and is saved in logs 

import mujoco
import mujoco_viewer
import numpy as np
import sys
import os

# Assuming your class is in franka_kinematics.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kinematics.franka_kinematics import FrankaKinematics

def verify_math_with_visuals():
    # 1. Initialize your DH math class
    fk_engine = FrankaKinematics()
    
    # 2. Define test configuration (Home)
    q_test = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
    
    # 3. CALCULATE position using YOUR math
    T_matrix = fk_engine.forward_kinematics(q_test)
    my_pos, _ = fk_engine.get_pose(T_matrix)
    
    # 4. Load MuJoCo model
    model = mujoco.MjModel.from_xml_path('../../docs/mujoco_menagerie/franka_emika_panda/panda.xml')
    data = mujoco.MjData(model)
    
    # Set the robot to the same pose in MuJoCo
    data.qpos[:7] = q_test
    mujoco.mj_forward(model, data)

    # 5. Launch Viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    print("\n--- ERROR TRACKING ---")
    
    while viewer.is_alive:
        # Get MuJoCo's actual fingertip position (left_finger) to compare
        mj_target_pos = data.body('left_finger').xpos
        error = np.linalg.norm(my_pos - mj_target_pos)
        
        # Print error directly to terminal (overwriting the same line)
        print(f"Current Error: {error*1000:.4f} mm", end='\r')

        # Draw red sphere at your calculated math position
        viewer.add_marker(
            pos=my_pos,
            size=[0.05, 0.05, 0.05],
            rgba=[1, 0, 0, 0.5],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            label=f"Math Target (Err: {error*1000:.2f}mm)"
        )
        
        viewer.render()

if __name__ == "__main__":
    verify_math_with_visuals()