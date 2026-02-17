import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "../docs/mujoco_menagerie/franka_emika_panda/panda.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Example pose (HOME pose)
q_home = np.array([0, -0.785, 0, -2.356, 0, 1.57, 0.785])

# Panda joints are first 7 qpos entries
data.qpos[:7] = q_home

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
