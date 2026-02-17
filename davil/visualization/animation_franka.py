#!/usr/bin/env python3

# Animate Franka arm moving through configurations
# Creates a .gif animation using Matplotlib. It interpolates joint angles 
# between a "Zero" and "Home" pose and renders them as a 3D stick figure.


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kinematics.franka_kinematics import FrankaKinematics
from franka_visualiser import PandaVisualizer


viz = PandaVisualizer()

configs = {
    'Zero': np.zeros(7),
    'Home': np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]),
}

# Interpolate between configs
q_start = configs['Zero']
q_end = configs['Home']
frames = 50
q_traj = [q_start + (q_end - q_start) * t for t in np.linspace(0, 1, frames)]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.cla()
    viz.plot_single_config(q_traj[frame], ax, title=f"Frame {frame+1}/{frames}")
    ax.view_init(elev=20, azim=45)

anim = FuncAnimation(fig, update, frames=frames, interval=50, repeat=True)
anim.save('logs/franka_motion.gif', writer='pillow', fps=10)
print("Saved: logs/franka_motion.gif")
plt.show()
