# This file is just where you can look and control the parameters for the panda robot in mujoco. 
# We can use this to validate the FK implementation by comparing the end-effector position from 
# mujoco with the one we computed using the FK code.


import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "../../docs/mujoco_menagerie/franka_emika_panda/panda.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Set joint angles
q = np.zeros(7)
data.qpos[:7] = q

# Forward kinematics only
mujoco.mj_forward(model, data)

# Get end effector
ee_id = model.body("hand").id
print("EE position:", data.xpos[ee_id])

# Just display pose (no stepping loop)
mujoco.viewer.launch(model, data)