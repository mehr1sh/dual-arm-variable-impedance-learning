Week 0 – Environment Setup
Set up Python, PyTorch, MuJoCo, CVXPY, Stable-Baselines3
Run a basic MuJoCo simulation successfully

Week 1 – Robotics & Math Basics
Understand robot joints, end-effectors, frames
Implement simple forward kinematics and visualize trajectories

Week 2 – Control & Trajectories
Learn PD and impedance control concepts
Implement quintic position + SLERP orientation trajectory generation

Week 3 – Simulation Environment
Build a Gym-style dual-arm MuJoCo environment
Implement observations and resets (object pose, joints, end-effector states)

Week 4 – QP-Based Controller (Core DAVIL Component)
Implement CVXPY-based QP impedance controller
Enforce joint, torque, and end-effector constraints
Convert accelerations to torques and validate in simulation

Week 5 – Reinforcement Learning (PPO)
Understand PPO and actor–critic training
Train a PPO policy to predict stiffness values (K)
Integrate RL policy with the QP controller

Week 6 – Rewards, Stability & Logging
Implement DAVIL reward terms (tracking, infeasibility penalty, EMA smoothness)
Add logging, checkpoints, and seed control
Train stable policies on simple object cases

Week 7 – Full DAVIL Training
Train across multiple objects and masses
Evaluate tracking performance over many trajectories

Week 8 – Baselines & Ablations
Implement baselines (fixed impedance, RL+IC, optimization-only)
Run ablations (e.g., remove EMA reward)
Reproduce main quantitative results (Table II)

Week 9 – Finalization (optional)
Clean codebase and document reproducibility steps
Generate plots, demos, and short technical report