# Dual-Arm Environment - Wayland Compatibility Guide

## Problem
Your system uses **Wayland** (modern display server), but MuJoCo's viewer uses GLFW which has limited Wayland support. This causes window positioning errors when running with `render_mode="human"`.

## Solutions

### ✅ Solution 1: Use XWayland (Recommended for Visualization)
Run with XWayland compatibility layer:

```bash
conda activate davil
GDK_BACKEND=x11 python scripts/test_visual_dual_arm.py
```

This uses X11 compatibility mode via XWayland, allowing the window to display properly.

**Advantages:**
- Full visualization with window
- Smooth rendering
- Interactive camera control

---

### ✅ Solution 2: Headless Mode (No Window)
Run without rendering for faster execution:

```bash
conda activate davil
python davil/envs/dual_arm_env.py
```

The environment automatically detects Wayland and uses headless mode.

**Advantages:**
- No window overhead
- Faster simulation
- No Wayland conflicts

**Use for:**
- Training algorithms
- Data collection
- Remote/SSH execution

---

### ✅ Solution 3: In Code - Force Headless Mode
```python
from davil.envs.dual_arm_env import DualArmPandaEnv

# No visualization
env = DualArmPandaEnv(render_mode=None)

# Use as normal
obs, info = env.reset()
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

---

### ✅ Solution 4: Switch Display Server (Advanced)
If you need X11 globally:

```bash
# Logout and select "Xorg" at login screen
# OR logout and run:
export GDK_SESSION_TYPE=x11
startx
```

---

## Verify Your Setup

Check current display server:
```bash
echo $XDG_SESSION_TYPE  # Returns: wayland or x11
```

Test visualization with XWayland:
```bash
GDK_BACKEND=x11 python -c "
from davil.envs.dual_arm_env import DualArmPandaEnv
env = DualArmPandaEnv(render_mode='human')
obs, info = env.reset()
for i in range(50):
    obs, _, _, _, info = env.step(env.action_space.sample())
    env.render()
    if i % 10 == 0:
        print(f'Step {i}: Running...')
env.close()
print('✓ Visualization test passed!')
"
```

---

## Performance Comparison

| Mode | Usage | Speed | Visualization |
|------|-------|-------|-------------|
| Headless (`render_mode=None`) | Training | ⚡ Fastest | ❌ No |
| XWayland (`GDK_BACKEND=x11`) | Development | 🟡 Medium | ✅ Yes |
| Wayland (default) | N/A | ❌ Crashes | ❌ No |

---

## Troubleshooting

**Window still closes with XWayland?**
```bash
# Try with more details
GDK_BACKEND=x11 python -u scripts/test_visual_dual_arm.py
```

**Need to save videos instead?**
- Use `render_mode="rgb_array"` (coming soon) with `imageio` to save frames

**SSH/Remote execution?**
- Always use headless mode (`render_mode=None`)
- Transfer results via SCP or commit to git

---

## Updated Code Features

The `DualArmPandaEnv` now includes:
- ✅ Automatic Wayland detection
- ✅ Graceful error handling
- ✅ Helpful error messages
- ✅ Fallback to headless mode
- ✅ Suppressed GLFW warnings

Enjoy your dual-arm environment! 🤖🤖
