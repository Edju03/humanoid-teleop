#!/usr/bin/env python3
"""
G1 Teleoperation Consumer for hybrik_realsense_fixed.py
Reads H2O keypoints from the HybrIK process and controls G1 robot
"""

import numpy as np
import mujoco
import mujoco.viewer
import sys
import os
import json
import subprocess
import threading
import queue
import time

# Setup paths
sys.path.append('/home/vb/workspace/humanoid-teleop')
os.chdir('/home/vb/workspace/humanoid-teleop')

# Import IK solver
from src.teleop.robot_arm_ik import G1_29_ArmIK

# Constants
ASAP_SCALE = 0.7517  # From ASAP optimization
G1_XML_PATH = 'models/g1.xml'
HYBRIK_SCRIPT = '/home/vb/workspace/HybrIK/hybrik_realsense_fixed.py'

# H2O keypoint indices (from hybrik_realsense_fixed.py)
# Order: [L_Ankle, R_Ankle, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist]
H2O_INDICES = {
    'left_ankle': 0,
    'right_ankle': 1,
    'left_shoulder': 2,
    'right_shoulder': 3,
    'left_elbow': 4,
    'right_elbow': 5,
    'left_wrist': 6,
    'right_wrist': 7
}

# Joint qpos indices for G1
JOINT_QPOS_INDICES = {
    'left_shoulder_pitch_joint': 22,
    'left_shoulder_roll_joint': 23,
    'left_shoulder_yaw_joint': 24,
    'left_elbow_joint': 25,
    'right_shoulder_pitch_joint': 26,
    'right_shoulder_roll_joint': 27,
    'right_shoulder_yaw_joint': 28,
    'right_elbow_joint': 29
}

class HybrIKConsumer:
    """Consumes H2O keypoints from hybrik_realsense_fixed.py and controls G1"""

    def __init__(self):
        print("="*80)
        print("G1 TELEOPERATION FROM HYBRIK_REALSENSE_FIXED.PY")
        print("="*80)

        # 1. MuJoCo model
        print("\nInitializing G1 robot...")
        self.model = mujoco.MjModel.from_xml_path(G1_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = 0  # Disable gravity for testing
        print("  MuJoCo model loaded")

        # 2. IK solver
        self.ik_solver = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        print(f"  IK solver ready ({self.ik_solver.reduced_robot.model.nq} DOF)")

        # 3. Queue for H2O data
        self.h2o_queue = queue.Queue(maxsize=10)
        self.latest_h2o = None

        # 4. State tracking
        self.last_joints = None
        self.smoothing = 0.3  # Reduced from 0.7 for better response

        # 5. Process handle
        self.hybrik_process = None
        self.reader_thread = None

        # 6. Set stable standing pose once
        self.init_stable_pose()

        print("\nSystem ready!")

    def init_stable_pose(self):
        """Set initial stable standing pose for robot"""
        # Position robot standing upright
        self.data.qpos[0:3] = [0, 0, 0.793]  # Base position
        self.data.qpos[3:7] = [1, 0, 0, 0]  # Base orientation (quaternion)

        # Set stable leg configuration (slight knee bend for stability)
        self.data.qpos[7] = 0.0   # left_hip_yaw_joint
        self.data.qpos[8] = 0.0   # left_hip_roll_joint
        self.data.qpos[9] = -0.1  # left_hip_pitch_joint (slight forward)
        self.data.qpos[10] = 0.2  # left_knee_joint (slight bend)
        self.data.qpos[11] = -0.1 # left_ankle_pitch_joint
        self.data.qpos[12] = 0.0  # left_ankle_roll_joint

        self.data.qpos[13] = 0.0  # right_hip_yaw_joint
        self.data.qpos[14] = 0.0  # right_hip_roll_joint
        self.data.qpos[15] = -0.1 # right_hip_pitch_joint (slight forward)
        self.data.qpos[16] = 0.2  # right_knee_joint (slight bend)
        self.data.qpos[17] = -0.1 # right_ankle_pitch_joint
        self.data.qpos[18] = 0.0  # right_ankle_roll_joint

        # Torso/waist joints
        self.data.qpos[19] = 0.0  # torso_joint
        self.data.qpos[20] = 0.0  # left_elbow_pitch_joint (if exists)
        self.data.qpos[21] = 0.0  # right_elbow_pitch_joint (if exists)

        # Arms will be set by IK solver (indices 22-29)

    def start_hybrik_process(self):
        """Start hybrik_realsense_fixed.py as subprocess"""
        print("\n" + "="*60)
        print("Starting HybrIK process...")
        print("="*60)

        # Check if script exists
        if not os.path.exists(HYBRIK_SCRIPT):
            print(f"Script not found: {HYBRIK_SCRIPT}")
            return False

        try:
            # Use conda run to ensure correct environment for HybrIK
            cmd = [
                'conda', 'run', '-n', 'base',
                'python', HYBRIK_SCRIPT
            ]

            # Start the process with proper environment
            self.hybrik_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                cwd='/home/vb/workspace/HybrIK'
            )

            # Start reader thread
            self.reader_thread = threading.Thread(target=self.read_h2o_data)
            self.reader_thread.daemon = True
            self.reader_thread.start()

            print("HybrIK process started")
            print("  The HybrIK window will show RGB + Depth")
            print("  This window shows the G1 robot")
            print("\nControls:")
            print("  'f' in HybrIK window = Toggle flip")
            print("  'q' in HybrIK window = Quit")
            print("  ESC in robot window = Quit")

            return True

        except Exception as e:
            print(f"Failed to start HybrIK: {e}")
            return False

    def read_h2o_data(self):
        """Read H2O keypoints from HybrIK stdout"""
        print("\nReading H2O data stream...")

        while self.hybrik_process and self.hybrik_process.poll() is None:
            try:
                line = self.hybrik_process.stdout.readline()
                if not line:
                    continue

                # Look for H2O_KEYPOINTS output
                if line.startswith("H2O_KEYPOINTS:"):
                    # Extract JSON data
                    json_str = line[14:].strip()
                    h2o_data = json.loads(json_str)

                    # Store in queue (drop old if full)
                    if self.h2o_queue.full():
                        try:
                            self.h2o_queue.get_nowait()
                        except:
                            pass
                    self.h2o_queue.put(h2o_data)
                    self.latest_h2o = h2o_data

                # Also print other messages
                elif not line.startswith("Frame"):  # Skip frame counter
                    print(f"[HybrIK] {line.rstrip()}")

            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"Reader error: {e}")
                break

        print("HybrIK process ended")

    def h2o_to_wrist_poses(self, h2o_data):
        """Convert H2O keypoints to wrist poses for IK"""
        if h2o_data is None or len(h2o_data) != 8:
            return None

        # Extract positions from H2O data
        left_shoulder = np.array(h2o_data[H2O_INDICES['left_shoulder']])
        right_shoulder = np.array(h2o_data[H2O_INDICES['right_shoulder']])
        left_elbow = np.array(h2o_data[H2O_INDICES['left_elbow']])
        right_elbow = np.array(h2o_data[H2O_INDICES['right_elbow']])
        left_wrist = np.array(h2o_data[H2O_INDICES['left_wrist']])
        right_wrist = np.array(h2o_data[H2O_INDICES['right_wrist']])

        # Compute orientations from elbow-wrist vectors
        left_forearm = left_wrist - left_elbow
        left_forearm = left_forearm / (np.linalg.norm(left_forearm) + 1e-6)

        right_forearm = right_wrist - right_elbow
        right_forearm = right_forearm / (np.linalg.norm(right_forearm) + 1e-6)

        # Create rotation matrices
        def create_rotation(direction):
            z = direction
            if abs(z[2]) < 0.9:
                x = np.cross([0, 0, 1], z)
            else:
                x = np.cross([1, 0, 0], z)
            x = x / (np.linalg.norm(x) + 1e-6)
            y = np.cross(z, x)
            return np.column_stack([x, y, z])

        return {
            'left_wrist': {
                'position': left_wrist,
                'rotation': create_rotation(left_forearm)
            },
            'right_wrist': {
                'position': right_wrist,
                'rotation': create_rotation(right_forearm)
            },
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder
        }

    def retarget_human_to_robot(self, human_poses):
        """Apply ASAP retargeting"""
        if human_poses is None:
            return None

        # Get actual shoulder positions from HybrIK
        human_shoulder_left = human_poses['left_shoulder']
        human_shoulder_right = human_poses['right_shoulder']

        # Robot shoulder offsets in robot frame
        robot_shoulder_offset_y = 0.15

        # Calculate robot base position from human torso center
        human_center = (human_shoulder_left + human_shoulder_right) / 2
        robot_base_height = 0.55

        # Map human shoulders to robot shoulders
        shoulder_width = np.linalg.norm(human_shoulder_right - human_shoulder_left)
        robot_shoulder_left = human_center + np.array([0, robot_shoulder_offset_y, robot_base_height - human_center[2]])
        robot_shoulder_right = human_center + np.array([0, -robot_shoulder_offset_y, robot_base_height - human_center[2]])

        # Scale and translate wrist positions relative to shoulders
        human_left = human_poses['left_wrist']['position']
        direction_left = human_left - human_shoulder_left
        robot_left = robot_shoulder_left + direction_left * ASAP_SCALE

        human_right = human_poses['right_wrist']['position']
        direction_right = human_right - human_shoulder_right
        robot_right = robot_shoulder_right + direction_right * ASAP_SCALE

        return {
            'left_wrist': {
                'position': robot_left,
                'rotation': human_poses['left_wrist']['rotation']
            },
            'right_wrist': {
                'position': robot_right,
                'rotation': human_poses['right_wrist']['rotation']
            }
        }

    def solve_ik(self, robot_poses):
        """Solve IK for robot wrist poses"""
        if robot_poses is None:
            return None

        # Create 6DOF transformation matrices
        left_tf = np.eye(4)
        left_tf[:3, 3] = robot_poses['left_wrist']['position']
        left_tf[:3, :3] = robot_poses['left_wrist']['rotation']

        right_tf = np.eye(4)
        right_tf[:3, 3] = robot_poses['right_wrist']['position']
        right_tf[:3, :3] = robot_poses['right_wrist']['rotation']

        try:
            sol_q, _ = self.ik_solver.solve_ik(left_tf, right_tf)

            # The reduced robot has 14 DOF total (indices 0-13):
            # 0-5: locked torso/waist (we ignore these)
            # 6-9: left arm joints
            # 10-13: right arm joints
            joint_angles = {}
            joint_angles['left_shoulder_pitch_joint'] = sol_q[6]
            joint_angles['left_shoulder_roll_joint'] = sol_q[7]
            joint_angles['left_shoulder_yaw_joint'] = sol_q[8]
            joint_angles['left_elbow_joint'] = sol_q[9]

            joint_angles['right_shoulder_pitch_joint'] = sol_q[10]
            joint_angles['right_shoulder_roll_joint'] = sol_q[11]
            joint_angles['right_shoulder_yaw_joint'] = sol_q[12]
            joint_angles['right_elbow_joint'] = sol_q[13]

            return joint_angles

        except Exception as e:
            return None

    def apply_joint_angles(self, joint_angles):
        """Apply joint angles to robot"""
        if joint_angles is None:
            return

        # Smooth joint angles (less aggressive for better response)
        if self.last_joints is not None:
            for joint_name in joint_angles:
                joint_angles[joint_name] = (
                    self.smoothing * self.last_joints.get(joint_name, 0) +
                    (1 - self.smoothing) * joint_angles[joint_name]
                )

        self.last_joints = joint_angles.copy()

        # Apply arm joint angles only - legs remain in stable pose from init
        for joint_name, angle in joint_angles.items():
            if joint_name in JOINT_QPOS_INDICES:
                qpos_idx = JOINT_QPOS_INDICES[joint_name]
                self.data.qpos[qpos_idx] = angle

        # Zero control torques for position control
        self.data.ctrl[:] = 0.0

    def process_h2o_data(self):
        """Process latest H2O data through pipeline"""
        # Get latest H2O data
        try:
            h2o_data = self.h2o_queue.get_nowait()
        except queue.Empty:
            h2o_data = self.latest_h2o

        if h2o_data is None:
            return

        # Pipeline: H2O → Wrist poses → Retarget → IK → Apply
        human_poses = self.h2o_to_wrist_poses(h2o_data)
        robot_poses = self.retarget_human_to_robot(human_poses)
        joint_angles = self.solve_ik(robot_poses)
        self.apply_joint_angles(joint_angles)

    def run(self):
        """Main teleoperation loop"""
        # Start HybrIK process
        if not self.start_hybrik_process():
            print("Failed to start HybrIK process")
            return

        # Wait for first data
        print("\nWaiting for HybrIK data...")
        timeout = 10
        start = time.time()
        while self.latest_h2o is None and time.time() - start < timeout:
            time.sleep(0.1)

        if self.latest_h2o is None:
            print("No data received from HybrIK. Check if camera is connected.")
            return

        print("Receiving H2O keypoints!")
        print("\n" + "="*60)
        print("TELEOPERATION ACTIVE")
        print("="*60)

        # Launch MuJoCo viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -10
            viewer.cam.azimuth = 0

            frame_count = 0

            while viewer.is_running():
                # Check if HybrIK is still running
                if self.hybrik_process and self.hybrik_process.poll() is not None:
                    print("\nHybrIK process ended")
                    break

                # Process H2O data
                self.process_h2o_data()

                # Step simulation
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                frame_count += 1
                if frame_count % 300 == 0:  # Every 10 seconds at 30fps
                    print(f"Teleoperation frame {frame_count}")

        # Cleanup
        if self.hybrik_process:
            self.hybrik_process.terminate()
            self.hybrik_process.wait()

def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("G1 TELEOPERATION SYSTEM")
    print("="*80)
    print("\nThis will:")
    print("  1. Start hybrik_realsense_fixed.py for pose extraction")
    print("  2. Read H2O keypoints from its output")
    print("  3. Control G1 robot via IK solver")
    print("\nMake sure:")
    print("  - RealSense camera is connected to USB 3.0")
    print("  - You are in the HybrIK conda environment")

    consumer = HybrIKConsumer()
    consumer.run()

    print("\n" + "="*80)
    print("TELEOPERATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()