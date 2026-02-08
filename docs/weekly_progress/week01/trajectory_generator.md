# Trajectory Generation

## Definition

Trajectory generation is the **planning of smooth, continuous paths** that a robot must follow to move from one position to another while respecting constraints on velocity and acceleration.

**Input**: Start pose (position + orientation) and end pose  
**Output**: Time-parameterized path with position, velocity, and acceleration profiles

---

## Problem Statement

Given:
- Robot's current pose: $\mathbf{p}_0, \mathbf{R}_0$
- Desired end pose: $\mathbf{p}_f, \mathbf{R}_f$
- Motion duration: $T$ seconds

Find:
- A smooth path from start to end
- Velocity and acceleration at each time step
- Minimize jerks (sudden accelerations)

---

## Quintic Polynomial Trajectories

### Why Quintic (5th Order)?

A quintic polynomial satisfies the boundary conditions needed for smooth motion:

$$p(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

**Constraints**:
- $p(0) = p_0$ (start position)
- $p(T) = p_f$ (end position)
- $\dot{p}(0) = 0$ (start with zero velocity)
- $\dot{p}(T) = 0$ (end with zero velocity)
- $\ddot{p}(0) = 0$ (start with zero acceleration)
- $\ddot{p}(T) = 0$ (end with zero acceleration)

This gives us **6 equations** for **6 unknowns** $[a_0, a_1, a_2, a_3, a_4, a_5]$.

### Solving for Coefficients

Set up the system:

$$A \cdot \mathbf{a} = \mathbf{b}$$

Where:

$$A = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 2 & 0 & 0 & 0 \\
1 & T & T^2 & T^3 & T^4 & T^5 \\
0 & 1 & 2T & 3T^2 & 4T^3 & 5T^4 \\
0 & 0 & 2 & 6T & 12T^2 & 20T^3
\end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix}
p_0 \\ 0 \\ 0 \\ p_f \\ 0 \\ 0
\end{bmatrix}$$

Solve: $\mathbf{a} = A^{-1} \mathbf{b}$

### Velocity and Acceleration

Once coefficients are found:

**Velocity**:
$$\dot{p}(t) = a_1 + 2a_2 t + 3a_3 t^2 + 4a_4 t^3 + 5a_5 t^4$$

**Acceleration**:
$$\ddot{p}(t) = 2a_2 + 6a_3 t + 12a_4 t^2 + 20a_5 t^3$$

---

## SLERP Orientation Interpolation

For smooth **rotation** between start and end orientations, use **Spherical Linear Interpolation (SLERP)**.

### Quaternion Representation

Orientations are represented as unit quaternions: $\mathbf{q} = [x, y, z, w]^T$

### SLERP Formula

$$\text{SLERP}(\mathbf{q}_1, \mathbf{q}_2, t) = \frac{\sin((1-t)\theta)}{\sin \theta} \mathbf{q}_1 + \frac{\sin(t\theta)}{\sin \theta} \mathbf{q}_2$$

Where:
- $\theta = \arccos(\mathbf{q}_1 \cdot \mathbf{q}_2)$ is the angle between quaternions
- $t \in [0, 1]$ is the interpolation parameter
- $t = 0$ → start orientation
- $t = 1$ → end orientation

**Properties**:
- Shortest rotation path
- Constant angular velocity
- Smooth interpolation in rotation space

---

## Visualization: `week1_trajectory.png`

The trajectory visualization has **3 subplots**:

### Subplot 1: 3D Position Path

```
Shows the spatial path from start to end
- Green dot = Start position (x₀, y₀, z₀)
- Blue curve = Quintic polynomial trajectory
- Red dot = End position (xf, yf, zf)
```

**What it tells you**:
- The exact path the robot's gripper traces in space
- Whether the motion is smooth and continuous
- Visual workspace validation

### Subplot 2: Velocity Profile

```
Shows Vx(t), Vy(t), Vz(t) over time
- Starts at 0 (stationary)
- Peaks in the middle of motion
- Returns to 0 at end (smooth stop)
```

**Characteristics of quintic trajectory**:
- Bell-shaped velocity curve (smooth, continuous)
- No abrupt velocity changes
- Zero initial and final velocity

### Subplot 3: Acceleration Profile

```
Shows Ax(t), Ay(t), Az(t) over time
- Symmetric S-shape curve
- Positive then negative (smooth jerk)
- Avoids mechanical stress on joints
```

**Why this matters**:
- Real robots have limited acceleration capacity
- Smoother acceleration = less wear and tear
- Better control and stability

---

## Example: Pick-and-Place Motion

**Task**: Move gripper from table to shelf height

**Start Pose**:
- Position: $(0.5, 0.0, 0.2)$ m (on table)
- Orientation: Identity (no rotation)

**End Pose**:
- Position: $(0.5, 0.0, 0.6)$ m (at shoulder height)
- Orientation: Identity

**Generated Motion**:
1. Gripper accelerates upward
2. Reaches peak velocity at $t = 1$ s
3. Decelerates smoothly to stop at $t = 2$ s
4. Zero acceleration at endpoints (safe mechanical operation)

---

## Key Advantages of Quintic Polynomials

| Property | Benefit |
|----------|---------|
| **Zero boundary velocities** | Safe start/stop with no jerk |
| **Zero boundary accelerations** | Smooth mechanical operation |
| **Deterministic path** | Same trajectory every time |
| **Computationally efficient** | Real-time calculation possible |
| **Minimal energy** | Reduces motor wear |

---

## Mathematical Summary

**Position trajectory**:
- Input: $p_0, p_f, T$
- Solve: System of 6 linear equations
- Output: Smooth quintic path with controlled velocity/acceleration

**Orientation trajectory**:
- Input: $\mathbf{q}_0, \mathbf{q}_f, T$
- Interpolate: SLERP in quaternion space
- Output: Smooth rotation without gimbal lock

**Combined SE(3) trajectory**:
- Concatenate position and orientation
- Use for full 6-DOF robot motion planning

---

## References

1. [Modern Robotics Chapter 9 - Trajectory Planning](http://modernrobotics.org)
2. [Quintic Polynomials in Robotics](https://en.wikipedia.org/wiki/Quintic_function)
3. [SLERP Interpolation](https://en.wikipedia.org/wiki/Slerp)
4. [Introduction to Robotics: Mechanics and Control - Craig](https://mitpress.mit.edu/9780201543612/)
