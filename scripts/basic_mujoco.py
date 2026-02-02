#!/usr/bin/env python3
"""
Week 0: Basic MuJoCo Simulation Test (Interactive)
Press ESC, Q, or close window to exit
"""

import mujoco
import mujoco.viewer
import numpy as np

def create_simple_scene():
    """Create a simple falling box scene"""
    xml = """
    <mujoco model="falling_box">
        <option timestep="0.001" gravity="0 0 -9.81"/>
        
        <visual>
            <global offwidth="1280" offheight="960"/>
        </visual>
        
        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                     rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
            <material name="grid" texture="grid" texrepeat="1 1"/>
        </asset>
        
        <worldbody>
            <light name="top" pos="0 0 2" dir="0 0 -1"/>
            <geom name="ground" type="plane" size="2 2 0.1" material="grid"/>
            
            <body name="box" pos="0 0 1">
                <joint type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1" mass="1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return xml

def test_simulation():
    """Interactive MuJoCo test - stays open until ESC/Q"""
    print("Creating MuJoCo model...")
    xml = create_simple_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    print(f"✓ Model created")
    print(f"  Bodies: {model.nbody}, Geoms: {model.ngeom}")
    print(f"  Timestep: {model.opt.timestep}s")
    print("\n▶ Launching INTERACTIVE viewer...")
    print("  Controls: Mouse drag=rotate, Scroll=zoom, ESC/Q=quit")
    print("  Watch red box fall + bounce!\n")
    
    # INTERACTIVE: Runs until you close it
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Runs FOREVER until ESC/Q - no step limit!
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

    print("\n✅ MuJoCo viewer closed successfully!")

if __name__ == "__main__":
    print("="*60)
    print(" Week 0: MuJoCo Interactive Test")
    print("="*60 + "\n")
    
    try:
        test_simulation()
        print("="*60)
        print("✅ Week 0 Complete!")
        print("Next: Week 1 - Forward Kinematics")
        print("="*60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
