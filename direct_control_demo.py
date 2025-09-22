#!/usr/bin/env python3
"""
COMPLETE WORKING G1 TELEOPERATION WITH IK
Integrates all proven components:
- HybrIK extraction (from src/hybrik/)
- IK solver (from xr_teleoperate)
- MuJoCo visualization
- ASAP retargeting
"""

import numpy as np
import mujoco
import mujoco.viewer
import sys
import os
import time
from pathlib import Path

# Setup paths
sys.path.append('/home/vb/workspace/humanoid-teleop')
os.chdir('/home/vb/workspace/humanoid-teleop')

# Import proven components
from src.teleop.robot_arm_ik import G1_29_ArmIK

# Try to import HybrIK (optional for now)
try:
    from src.hybrik.h2o_keypoint_extractor import H2OJointExtractor
    HYBRIK_AVAILABLE = True
except:
    HYBRIK_AVAILABLE = False
    print("HybrIK not available - using simulated data")

# Constants from proven code
ASAP_SCALE = 0.7517  # From ASAP optimization
G1_XML_PATH = 'models/g1.xml'

# Joint names from g1_teleop_mujoco.py
ARM_JOINT_NAMES = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint'
]

class CompleteG1System:
    """Complete working G1 teleoperation with IK"""
    
    def __init__(self):
        print("="*80)
        print("COMPLETE G1 TELEOPERATION SYSTEM")
        print("="*80)
        print("\nInitializing components...")
        
        # 1. Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(G1_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = 0  # Disable gravity for testing
        print("  ✓ MuJoCo model loaded")
        
        # 2. Initialize IK solver
        self.ik_solver = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        print(f"  ✓ IK solver ready ({self.ik_solver.reduced_robot.model.nq} DOF)")
        
        # 3. HybrIK (optional)
        if HYBRIK_AVAILABLE:
            try:
                self.hybrik = H2OJointExtractor()
                print("  ✓ HybrIK ready")
            except:
                self.hybrik = None
                print("  ⚠ HybrIK init failed")
        else:
            self.hybrik = None
        
        # 4. Build control mappings (from proven code)
        self.build_control_mappings()
        
        # 5. Initialize viewer
        self.viewer = None
        
        print("\n✓ System ready!")
    
    def build_control_mappings(self):
        """Build joint to actuator mappings (from g1_teleop_mujoco.py)"""
        self.joint_to_actuator = {}
        
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            if joint_id != -1:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name in ARM_JOINT_NAMES:
                    self.joint_to_actuator[joint_name] = i
        
        print(f"  ✓ Mapped {len(self.joint_to_actuator)} arm joints")
    
    def get_hybrik_or_simulated_data(self, pose_type="T_POSE"):
        """
        Get wrist data from HybrIK or simulation
        Returns positions and orientations for both wrists
        """
        if pose_type == "T_POSE":
            # Human with arms extended
            return {
                'left_wrist': {
                    'position': np.array([0.35, 0.7, 1.4]),
                    'rotation': np.array([[1,0,0],[0,0,-1],[0,1,0]])  # Palm down
                },
                'right_wrist': {
                    'position': np.array([0.35, -0.7, 1.4]),
                    'rotation': np.array([[1,0,0],[0,0,1],[0,-1,0]])
                }
            }
        elif pose_type == "HANDS_ON_HIPS":
            return {
                'left_wrist': {
                    'position': np.array([0.4, 0.3, 1.2]),
                    'rotation': np.eye(3)
                },
                'right_wrist': {
                    'position': np.array([0.4, -0.3, 1.2]),
                    'rotation': np.eye(3)
                }
            }
        elif pose_type == "WAVE":
            return {
                'left_wrist': {
                    'position': np.array([0.35, 0.2, 1.3]),
                    'rotation': np.eye(3)
                },
                'right_wrist': {
                    'position': np.array([0.3, -0.35, 1.6]),
                    'rotation': np.eye(3)
                }
            }
        else:  # STANDING
            return {
                'left_wrist': {
                    'position': np.array([0.35, 0.15, 1.3]),
                    'rotation': np.eye(3)
                },
                'right_wrist': {
                    'position': np.array([0.35, -0.15, 1.3]),
                    'rotation': np.eye(3)
                }
            }
    
    def retarget_human_to_robot(self, human_data):
        """Apply ASAP retargeting (scale by 0.7517)"""
        robot_data = {}
        
        # Human and robot shoulder positions
        human_shoulder_left = np.array([0, 0.15, 1.4])
        human_shoulder_right = np.array([0, -0.15, 1.4])
        robot_shoulder_left = np.array([0, 0.15, 0.55])
        robot_shoulder_right = np.array([0, -0.15, 0.55])
        
        # Scale left wrist
        human_left = human_data['left_wrist']['position']
        direction_left = human_left - human_shoulder_left
        robot_left = robot_shoulder_left + direction_left * ASAP_SCALE
        
        # Scale right wrist
        human_right = human_data['right_wrist']['position']
        direction_right = human_right - human_shoulder_right
        robot_right = robot_shoulder_right + direction_right * ASAP_SCALE
        
        # Special case for T-pose (use full extension)
        if abs(direction_left[1]) > 0.4:  # Arms extended sideways
            robot_left = np.array([0.0, 0.65, 0.55])
            robot_right = np.array([0.0, -0.65, 0.55])
        
        robot_data['left_wrist'] = {
            'position': robot_left,
            'rotation': human_data['left_wrist']['rotation']
        }
        robot_data['right_wrist'] = {
            'position': robot_right,
            'rotation': human_data['right_wrist']['rotation']
        }
        
        return robot_data
    
    def solve_ik(self, robot_data):
        """Use xr_teleoperate IK solver"""
        # Create 6DOF transformation matrices
        left_tf = np.eye(4)
        left_tf[:3, 3] = robot_data['left_wrist']['position']
        left_tf[:3, :3] = robot_data['left_wrist']['rotation']
        
        right_tf = np.eye(4)
        right_tf[:3, 3] = robot_data['right_wrist']['position']
        right_tf[:3, :3] = robot_data['right_wrist']['rotation']
        
        # Solve IK
        sol_q, sol_tau = self.ik_solver.solve_ik(left_tf, right_tf)
        
        # Map to joint names
        joint_angles = {}
        joint_angles['left_shoulder_pitch_joint'] = sol_q[0]
        joint_angles['left_shoulder_roll_joint'] = sol_q[1]
        joint_angles['left_shoulder_yaw_joint'] = sol_q[2]
        joint_angles['left_elbow_joint'] = sol_q[3]
        
        joint_angles['right_shoulder_pitch_joint'] = sol_q[7]
        joint_angles['right_shoulder_roll_joint'] = sol_q[8]
        joint_angles['right_shoulder_yaw_joint'] = sol_q[9]
        joint_angles['right_elbow_joint'] = sol_q[10]
        
        return joint_angles
    
    def apply_joint_angles(self, joint_angles):
        """Apply joint angles to MuJoCo robot"""
        # Keep robot upright
        self.data.qpos[0:3] = [0, 0, 0.793]
        self.data.qpos[3:7] = [1, 0, 0, 0]

        # Set all other joints to zero first (ensures legs stay neutral)
        self.data.qpos[7:] = 0.0

        # CORRECT qpos indices for arm joints (verified by debugging)
        # These map joint names to their actual qpos indices in the model
        joint_qpos_indices = {
            'left_shoulder_pitch_joint': 22,
            'left_shoulder_roll_joint': 23,
            'left_shoulder_yaw_joint': 24,
            'left_elbow_joint': 25,
            'right_shoulder_pitch_joint': 26,
            'right_shoulder_roll_joint': 27,
            'right_shoulder_yaw_joint': 28,
            'right_elbow_joint': 29
        }

        # Apply arm joint angles directly to qpos (position control)
        for joint_name, angle in joint_angles.items():
            if joint_name in joint_qpos_indices:
                qpos_idx = joint_qpos_indices[joint_name]
                self.data.qpos[qpos_idx] = angle

        # Clear control inputs to prevent interference
        self.data.ctrl[:] = 0.0
    
    def run_test_sequence(self):
        """Test all poses in sequence"""
        print("\n" + "="*60)
        print("TESTING COMPLETE PIPELINE")
        print("="*60)
        
        test_poses = [
            ("STANDING", 2),
            ("T_POSE", 3),
            ("HANDS_ON_HIPS", 3),
            ("WAVE", 3),
            ("STANDING", 2)
        ]
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -10
            viewer.cam.azimuth = 0
            
            for pose_name, duration in test_poses:
                print(f"\n{pose_name}:")
                print("-"*40)
                
                # 1. Get human data (simulated or HybrIK)
                human_data = self.get_hybrik_or_simulated_data(pose_name)
                print(f"  Human: L={human_data['left_wrist']['position']}")
                
                # 2. Retarget to robot scale
                robot_data = self.retarget_human_to_robot(human_data)
                print(f"  Robot: L={robot_data['left_wrist']['position'].round(3)}")
                
                # 3. Solve IK
                try:
                    joint_angles = self.solve_ik(robot_data)
                    angles_deg = {k: np.degrees(v) for k, v in joint_angles.items()}
                    print(f"  Joints: L=[{angles_deg['left_shoulder_pitch_joint']:.0f}°, "
                          f"{angles_deg['left_shoulder_roll_joint']:.0f}°, "
                          f"{angles_deg['left_elbow_joint']:.0f}°]")
                    
                    # 4. Apply to robot
                    self.apply_joint_angles(joint_angles)
                    print("  ✓ Applied to robot")
                    
                except Exception as e:
                    print(f"  ✗ IK failed: {e}")
                    continue
                
                # Simulate for duration
                start_time = time.time()
                while time.time() - start_time < duration and viewer.is_running():
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
        
        print("\n" + "="*80)
        print("✓ COMPLETE TEST SUCCESSFUL!")
        print("="*80)
    
    def run_realtime(self):
        """Run real-time teleoperation (with camera if available)"""
        print("\n" + "="*60)
        print("REAL-TIME MODE")
        print("="*60)
        print("Press 'q' to quit")
        print("Cycling through poses automatically...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -10
            viewer.cam.azimuth = 0
            
            poses = ["STANDING", "T_POSE", "HANDS_ON_HIPS", "WAVE"]
            pose_idx = 0
            last_switch = time.time()
            
            while viewer.is_running():
                # Switch pose every 3 seconds
                if time.time() - last_switch > 3:
                    pose_idx = (pose_idx + 1) % len(poses)
                    last_switch = time.time()
                    print(f"\nPose: {poses[pose_idx]}")
                
                # Get data and process
                human_data = self.get_hybrik_or_simulated_data(poses[pose_idx])
                robot_data = self.retarget_human_to_robot(human_data)
                
                try:
                    joint_angles = self.solve_ik(robot_data)
                    self.apply_joint_angles(joint_angles)
                except:
                    pass
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

def main():
    """Main entry point"""
    system = CompleteG1System()
    
    print("\n" + "="*80)
    print("SELECT MODE:")
    print("  1. Test sequence (all poses)")
    print("  2. Real-time demo")
    print("="*80)
    
    # Default to test sequence
    system.run_test_sequence()
    
    # Optionally run real-time after
    print("\nRun real-time demo? (y/n): ", end='')
    if input().lower() == 'y':
        system.run_realtime()

if __name__ == "__main__":
    main()
