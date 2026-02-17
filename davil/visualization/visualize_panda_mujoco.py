# A standalone visualization script that allows you to pass specific joint angles as 
# command-line arguments (e.g., --joints 0 -0.7 0 -2.3 0 1.5 0.7).

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import time
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize Franka Panda Arm Configuration in MuJoCo")
    parser.add_argument('--joints', nargs=7, type=float, help="7 joint angles in radians (e.g., 0 0 0 -1.5 0 1.5 0.7)")
    
    args = parser.parse_args()

    # Path to the MuJoCo model
    # Adjust this path if necessary to match your directory structure
    model_path = os.path.join(os.path.dirname(__file__), "../../docs/mujoco_menagerie/franka_emika_panda/panda.xml")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from: {model_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to load MuJoCo model: {e}")
        return

    # Default configuration if no joints provided
    if args.joints:
        q = np.array(args.joints)
    else:
        print("No joint angles provided. Using default 'Home' configuration.")
        q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

    print(f"Visualizing configuration: {q}")

    # Set joint positions
    # Note: panda.xml joints might be named or ordered. usually qpos[:7] corresponds to the 7 joints if it's the only robot.
    # We assume the first 7 DOFs are the arm joints.
    data.qpos[:7] = q
    
    # Forward kinematics to update body positions
    mujoco.mj_forward(model, data)

    print("Launching MuJoCo viewer...")
    print("Press 'Esc' to exit the viewer.")

    # Launch the viewer
    # We use a loop here to ensure the viewer stays open and the configuration is held
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # We are not simulating physics (stepping), just visualizing the static pose.
            # So we constantly reset qpos to the desired configuration to prevent gravity from pulling it down
            # if we were stepping. But since we are NOT stepping, setting it once *should* be enough if we just sync.
            # However, to be safe and allow interaction without physics messing it up:
            data.qpos[:7] = q
            mujoco.mj_forward(model, data) # Update kinematics

            viewer.sync()

            # Sleep to limit FPS
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
