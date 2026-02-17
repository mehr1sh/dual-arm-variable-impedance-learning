**MuJoCo Integration & Validation**
1. Set up mujoco menagerie fron google deepmind
2. Frame Calibration
   - Our DH model vs MuJoCo URDF use different coordinate frames:
   - DH convention: Standard robotics (Craig 1989)
   - MuJoCo: URDF-based kinematic tree
   - 721mm position mismatch at zero pose before calibration 
3. MuJoCo uses MJCF (MuJoCo XML) or URDF formats, not DH tables directly

**Resources**
1. Official Documentation: https://mujoco.readthedocs.io/
2. MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie

**Frame Calibration**
1. Murray, R. M., Li, Z., & Sastry, S. S. (1994). A Mathematical Introduction to Robotic Manipulation. CRC Press. Chapter 2.