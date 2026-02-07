# Franka Kinematics

Franka kinematics refers to the forward and inverse kinematic models of the Franka Emika Panda, a 7-DOF collaborative robot arm with spherical shoulder, offset elbow, and non-spherical wrist.

Uses DH parameters for joint transformations that are pre-measured from the Franka Panda datasheet provided by robot manufacturers (we don't calculate them).

The code implements Denavit-Hartenberg (DH) forward kinematics for a 7-DOF Franka Emika Panda robot arm. It computes:

- **Forward Kinematics (FK)**: Joint angles → End-effector pose
- **Jacobian**: Relates joint velocities to end-effector velocities
- **Inverse Kinematics (IK)**: End-effector pose → Joint angles

## 1. DH Transformation Matrix

Converts $[a, \alpha, d, \theta]$ → 4×4 transformation matrix.

$$T = \begin{bmatrix} 
c\theta & -s\theta \cos\alpha & s\theta \sin\alpha & a \cos\theta \\
s\theta & c\theta \cos\alpha & -c\theta \sin\alpha & a \sin\theta \\
0 & \sin\alpha & \cos\alpha & d \\
0 & 0 & 0 & 1
\end{bmatrix}$$

Where:
- **Upper-left 3×3**: Rotation matrix
- **Right column (rows 0-2)**: Position $[x, y, z]^T$
- **Bottom row**: Homogeneous coordinate (always $[0, 0, 0, 1]$)

## 2. Forward Kinematics

Steps:
1. Start with identity matrix $I_{4×4}$ (no transformation yet)
2. For each joint $i$:
   - Get DH parameters $(a, \alpha, d)$
   - Apply joint angle: $\theta_i = q_i$ (plus any offset)
   - Compute transformation: $T_i$
   - Chain multiply: $T \mathrel{↦} T \cdot T_i$
3. Final $T$ is the end-effector pose (position + orientation)

## 3. Extract Position and Quaternion

- **Position**: Extract the last column (translation): $\mathbf{p} = [x, y, z]^T$
- **Orientation**: Convert rotation matrix $R$ to quaternion $(x, y, z, w)$
- A quaternion is a compact representation: $\mathbf{q} = x\mathbf{i} + y\mathbf{j} + z\mathbf{k} + w$

## 4. Jacobian Computation

The Jacobian is a 6×7 matrix relating joint velocities to end-effector velocities:

$$\begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} = J \begin{bmatrix} \dot{q}_1 \\ \vdots \\ \dot{q}_7 \end{bmatrix}$$

Where:
- $\mathbf{v}$ = linear velocity (3D)
- $\boldsymbol{\omega}$ = angular velocity (3D)
- $\dot{q}_i$ = joint velocities

**Numerical differentiation**:

$$J_{i,j} = \frac{\partial x_i}{\partial q_j} \approx \frac{x_i(q+\Delta q) - x_i(q)}{\Delta q}$$

**Columns 0-2** (position component):
$$J_{k,i} = \frac{\partial \mathbf{p}_k}{\partial q_i}$$

**Columns 3-5** (orientation component - simplified):
$$J_{k,i} \approx \frac{\partial q_k}{\partial q_i}$$

(quaternion components)

## 5. Numerical Inverse Kinematics

**Problem**: Given desired end-effector pose, find joint angles

**Method**: Jacobian Pseudo-Inverse

1. Compute error (6D): $\mathbf{e} = [\mathbf{p}_{target} - \mathbf{p}_{current}, \mathbf{q}_{target} - \mathbf{q}_{current}]^T$

2. Compute Jacobian at current configuration

3. Pseudo-inverse (handles singularities):
   $$J^+ = (J^T J)^{-1} J^T$$

4. Update joint angles:
   $$\Delta q = J^+ \mathbf{e}$$
   $$q_{new} = q_{old} + \alpha \Delta q$$
   
   Where $\alpha = 0.1$ is a step size (prevents large jumps)

5. Enforce limits:
   $$q_{new} = \text{clip}(q_{new}, q_{min}, q_{max})$$

6. Repeat until $\lVert \mathbf{e} \rVert < \text{tolerance}$

## 6. Example Test Cases

- **Zero config**: All joints at 0° (arm stretched)
- **Home config**: Special pose (typical safe configuration)

For each, it computes FK and prints:
- **Position**: $(x, y, z)$ in meters
- **Quaternion**: $(x, y, z, w)$ orientation

## 7. Manipulability Index

Measures how well the robot can move in all directions:

$$\mu = \sqrt{\det(J J^T)}$$

- High $\mu$ → Good controllability
- Low $\mu$ → Near singularity (arm gets "stuck")



**Resources**
1. [The Franka Emika Robot: A Standard Platform in Robotics Research [Survey]](https://ieeexplore.ieee.org/document/10693652)
2. [Franka World](https://franka.world/resources)

