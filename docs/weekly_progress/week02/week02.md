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

## Impedance Control Resources

### Beginner Tutorial
- [Impedance Control Tutorial](https://www.youtube.com/watch?v=XxjHxX7mEOM)

### Conceptual Overview  
- [Impedance Control Explained](https://robotics-explained.com/impedancecontrol/)

### Research Review
- [Variable Impedance Control Review](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2020.590681/full)

### Implementation Guide
- [Advanced Impedance Control - MathWorks](https://www.mathworks.com/company/technical-articles/enhancing-robot-precision-and-safety-with-impedance-control.html)
