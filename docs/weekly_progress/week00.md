**Week 0 – Environment Setup**
Set up Python, PyTorch, MuJoCo, CVXPY, Stable-Baselines3
Run a basic MuJoCo simulation successfully

**PyTorch Ecosystem**
PyTorch handles math heavy calculations, robot simulations, optimisations and tracking 

**Robot Simulation**
MuJoCo - A physics engine simulating robot arms, joints, forces and contacts

Gymnasium - Creates environments where AI agents trial-and-error tasks, tracking rewards for successful grasps or movements

**RL Training Framework**
stable-baselines - Pre-built RL algorithms (like PPO, SAC) that train policies on top of gymnasium and mujoco

**Math and optimisation**
cvxpy - solves convex optimisation problems - like finding the most efficient and safest impedance settings for robot arms under constraints 

numpy, scipy, matplotlib: NumPy/scipy for fast array math and scientific computing; matplotlib for plotting learning curves, robot trajectories, or impedances.

**Experiment Tracking**
tensorboard: Web dashboard to visualize training progress—loss curves, rewards, robot videos—making debugging easier during long RL runs.

1. Python Environment Setup
```
bash
# Create conda environment
conda create -n davil python=3.10
conda activate davil

# Install PyTorch (check your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core libraries
pip install mujoco==3.1.1
pip install stable-baselines3[extra]==2.2.1
pip install cvxpy==1.4.2
pip install gymnasium==0.29.1
pip install numpy scipy matplotlib
pip install tensorboard

#run basic simulation
python scripts/basic_mujoco.py
```

**Stuff done:**
requirements specified and installed all dependencies
setup mujoco and conda environment 
ran a basic simple mujoco simulation 
**made the files scripts/{test_installation.py, basic_mujoco.py}**