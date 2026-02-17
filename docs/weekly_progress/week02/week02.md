**Week 2 – Control & Trajectories**
Learn PD and impedance control concepts
Implement quintic position + SLERP orientation trajectory generation

1. Visualize Franka Panda arm configurations from DH-based FK in the file **davil/visualization/visualize_panda_mujoco.py**
2. Implement quintic polynomial position trajectories and SLERP orientation interpolation

Files created: 
1. the visualisation folder inside davil with logs
2. trajectory_generator.py verified in davil/envs
3. interactive_panda.py in davil/kinematics
4. Downloaded official Franka Panda model from Google DeepMind's MuJoCo Menagerie in davil/docs - resolved XML parsing issues amnd set up viewer backend

