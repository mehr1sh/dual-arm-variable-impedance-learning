# Forward Kinematics

## Definition

Forward kinematics is the calculation of the **position and orientation of the end-effector** (gripper) from the **joint angles**.

**Input**: Joint angles ($\theta_1, \theta_2, ..., \theta_n$)  
**Output**: End-effector position $(x, y, z)$ and orientation $(R)$

---

## Problem Statement

Given:
- Robot arm with $n$ joints
- Current joint angles: $q = [\theta_1, \theta_2, ..., \theta_n]^T$

Find:
- Where does the end-effector reach in 3D space?
- Which direction is it pointing?

---

## Example: 3R Planar Robot

Consider a 2D robot with 3 revolute joints and link lengths $L_1, L_2, L_3$.

**Using basic trigonometry:**

$$x = L_1 \cos \theta_1 + L_2 \cos(\theta_1 + \theta_2) + L_3 \cos(\theta_1 + \theta_2 + \theta_3)$$

$$y = L_1 \sin \theta_1 + L_2 \sin(\theta_1 + \theta_2) + L_3 \sin(\theta_1 + \theta_2 + \theta_3)$$

$$\psi = \theta_1 + \theta_2 + \theta_3$$

Where:
- $(x, y)$ = end-effector position
- $\psi$ = end-effector orientation

---

## Systematic Approach: Homogeneous Transformation Matrices

For more complex robots, use **Denavit-Hartenberg (DH) parameters** and transformation matrices.

**Procedure:**

1. Assign a reference frame to the base (frame $\{0\}$)
2. Attach a frame to each joint/link
3. Attach a frame to the end-effector (frame $\{n\}$)
4. Compute transformation matrices for each joint

**Combined transformation:**

$$T_{0n} = T_{01} \cdot T_{12} \cdot T_{23} \cdot ... \cdot T_{n-1,n}$$

Each matrix $T_{i-1,i}$ contains:
- Rotation from joint $i-1$ to joint $i$
- Translation from joint $i-1$ to joint $i$

---

## Example: 3R Robot Using Matrices

$$T_{04} = T_{01} T_{12} T_{23} T_{34}$$

Where each $4 \times 4$ transformation matrix has the form:

$$T = \begin{bmatrix} 
\cos \theta_i & -\sin \theta_i & 0 & L_i \\
\sin \theta_i & \cos \theta_i & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

The final transformation $T_{04}$ gives:
- **Position**: Last column (first 3 rows)
- **Orientation**: Upper-left $3 \times 3$ rotation matrix

---

## Key Points

- Forward kinematics is **deterministic** (always one answer)
- Requires knowing the **robot geometry** (link lengths, joint types)
- Forms the basis for **Jacobian** and **inverse kinematics**
- Result is a **homogeneous transformation matrix** combining position + orientation

---

## References

1. [Modern Robotics Chapter 4](http://modernrobotics.org)
2. [Robotics 101 Playlist](https://www.youtube.com/playlist?list=PL1YrgW7ROFofBqPGiWAmTqIwDc5SrzZrA)
3. [Clemson Robotics - Forward Kinematics](https://opentextbooks.clemson.edu/wangrobotics/chapter/forward-kinematics/)
4. [DH Parameters Guide](https://blog.robotiq.com/how-to-calculate-a-robots-forward-kinematics-in-5-easy-steps)