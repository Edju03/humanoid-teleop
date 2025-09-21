#!/usr/bin/env python3
"""
G1 Robot Teleoperation with HybrIK
Position control implementation with MuJoCo visualization
"""

import numpy as np
import mujoco
import mujoco.viewer
import cv2
import pyrealsense2 as rs
import time
import sys
import os

# Add HybrIK to path
HYBRIK_PATH = os.environ.get('HYBRIK_PATH', '/home/vb/workspace/HybrIK')
sys.path.append(HYBRIK_PATH)

from h2o_keypoint_extractor import H2OJointExtractor

# Get package directory
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G1_XML_PATH = os.path.join(PACKAGE_DIR, 'models', 'g1.xml')
KP_GAIN = 300
SMOOTHING = 0.3
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

ARM_JOINT_NAMES = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
]

BODY_NAMES = {
    'left_shoulder': 'left_shoulder_roll_link',
    'left_elbow': 'left_elbow_link',
    'right_shoulder': 'right_shoulder_roll_link',
    'right_elbow': 'right_elbow_link',
}

JOINT_LIMITS = {
    'left_shoulder_pitch_joint': (-3.0892, 2.6704),
    'left_shoulder_roll_joint': (-1.5882, 2.2515),
    'left_shoulder_yaw_joint': (-2.618, 2.618),
    'left_elbow_joint': (-1.0472, 2.0944),
    'right_shoulder_pitch_joint': (-3.0892, 2.6704),
    'right_shoulder_roll_joint': (-2.2515, 1.5882),
    'right_shoulder_yaw_joint': (-2.618, 2.618),
    'right_elbow_joint': (-1.0472, 2.0944),
}

class RealSenseCamera:
    """RealSense camera interface"""
    
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, 
                           rs.format.bgr8, CAMERA_FPS)
        config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT,
                           rs.format.z16, CAMERA_FPS)
        
        print("Starting RealSense camera with DEPTH...")
        profile = self.pipeline.start(config)
        
        # Create align object to align depth to color
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # Get camera intrinsics for accurate 3D reconstruction
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.intrinsics = color_profile.get_intrinsics()
        print(f"Camera intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
        print(f"   Principal point: cx={self.intrinsics.ppx:.1f}, cy={self.intrinsics.ppy:.1f}")
        
        for _ in range(10):
            self.pipeline.wait_for_frames()

        print("RealSense ready")
    
    def get_frames(self):
        """Get aligned color and depth frames"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
        except Exception as e:
            print(f"Camera error: {e}")
            return None, None
    
    def close(self):
        """Close camera"""
        self.pipeline.stop()

class G1TeleoperationSystem:
    """G1 teleoperation system"""
    
    def __init__(self):
        print("Initializing G1 Teleoperation System...")

        # Load MuJoCo model
        self.mujoco_model = mujoco.MjModel.from_xml_path(G1_XML_PATH)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        self.camera = RealSenseCamera()
        self.h2o_extractor = H2OJointExtractor()
        self.viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)

        self.build_control_mappings()

        self.body_indices = {}
        for name, body_name in BODY_NAMES.items():
            body_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                self.body_indices[name] = body_id
                print(f"  Found {name} body: {body_name} -> id {body_id}")

        self.standing_pose = self.get_standing_pose()
        self.apply_standing_pose()

        print("G1 Teleoperation System ready!")
    
    def get_standing_pose(self):
        """Get standing pose joint angles"""
        standing_pose = {}

        standing_pose['left_hip_pitch_joint'] = np.radians(-12.0)
        standing_pose['left_hip_roll_joint'] = np.radians(0.0)
        standing_pose['left_hip_yaw_joint'] = np.radians(0.0)
        standing_pose['left_knee_joint'] = np.radians(25.0)
        standing_pose['left_ankle_pitch_joint'] = np.radians(-13.0)
        standing_pose['left_ankle_roll_joint'] = np.radians(0.0)

        standing_pose['right_hip_pitch_joint'] = np.radians(-12.0)
        standing_pose['right_hip_roll_joint'] = np.radians(0.0)
        standing_pose['right_hip_yaw_joint'] = np.radians(0.0)
        standing_pose['right_knee_joint'] = np.radians(25.0)
        standing_pose['right_ankle_pitch_joint'] = np.radians(-13.0)
        standing_pose['right_ankle_roll_joint'] = np.radians(0.0)

        standing_pose['waist_yaw_joint'] = np.radians(0.0)
        standing_pose['waist_roll_joint'] = np.radians(0.0)
        standing_pose['waist_pitch_joint'] = np.radians(0.0)

        for joint_name in ARM_JOINT_NAMES:
            standing_pose[joint_name] = 0.0

        return standing_pose

    def apply_standing_pose(self):
        """Apply standing pose to maintain balance"""
        for joint_name, angle in self.standing_pose.items():
            joint_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id != -1:
                actuator_id = None
                for i in range(self.mujoco_model.nu):
                    if self.mujoco_model.actuator_trnid[i, 0] == joint_id:
                        actuator_id = i
                        break
                if actuator_id is not None:
                    self.mujoco_data.ctrl[actuator_id] = angle

    def build_control_mappings(self):
        """Build mappings from joint names to actuator indices"""
        print("\nBuilding control mappings...")

        self.joint_to_actuator_map = {}
        self.actuator_to_joint_map = {}
        self.all_joint_to_actuator = {}

        for i in range(self.mujoco_model.nu):
            joint_id = self.mujoco_model.actuator_trnid[i, 0]
            if joint_id != -1:
                joint_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                self.all_joint_to_actuator[joint_name] = i
                if joint_name in ARM_JOINT_NAMES:
                    self.joint_to_actuator_map[joint_name] = i
                    self.actuator_to_joint_map[i] = joint_name
                    print(f"  {joint_name:<30} -> actuator[{i:2d}]")

        print(f"  Mapped {len(self.joint_to_actuator_map)} arm joints to actuators")
    
    def apply_joint_angles(self, joint_angles):
        """Apply joint angles using position control"""
        self.apply_standing_pose()
        for joint_name, angle in joint_angles.items():
            if joint_name in self.joint_to_actuator_map:
                actuator_id = self.joint_to_actuator_map[joint_name]
                self.mujoco_data.ctrl[actuator_id] = angle
                print(f"  Set {joint_name}: {np.degrees(angle):.1f} deg (actuator {actuator_id})")
    
    def test_arm_folding(self):
        """Test arm folding capability"""
        print("\nTesting arm folding with position control...")
        print("\nTesting DEFAULT POSE (Straight Arms)")
        default_angles = {name: 0.0 for name in ARM_JOINT_NAMES}
        self.apply_joint_angles(default_angles)
        self.step_simulation(500)
        self.check_arm_positions("DEFAULT")
        
        print("\nTesting HANDS ON WAIST (Folded Arms)")
        folded_angles = {name: 0.0 for name in ARM_JOINT_NAMES}
        folded_angles['left_elbow_joint'] = np.radians(90)
        folded_angles['right_elbow_joint'] = np.radians(90)
        folded_angles['left_shoulder_pitch_joint'] = np.radians(45)
        folded_angles['right_shoulder_pitch_joint'] = np.radians(45)
        self.apply_joint_angles(folded_angles)
        self.step_simulation(1000)
        self.check_arm_positions("HANDS ON WAIST")
        
        # Test 3: Arms up (T-pose)
        print("\nTesting ARMS UP (T-Pose)")
        arms_up_angles = {name: 0.0 for name in ARM_JOINT_NAMES}
        arms_up_angles['left_shoulder_pitch_joint'] = np.radians(-90)  # Left up
        arms_up_angles['right_shoulder_pitch_joint'] = np.radians(-90) # Right up
        self.apply_joint_angles(arms_up_angles)
        self.step_simulation(1000)
        self.check_arm_positions("ARMS UP")
        
        # Test 4: One arm folded, one straight
        print("\nTesting MIXED POSE (Left folded, Right up)")
        mixed_angles = {name: 0.0 for name in ARM_JOINT_NAMES}
        mixed_angles['left_elbow_joint'] = np.radians(90)      # Left folded
        mixed_angles['left_shoulder_pitch_joint'] = np.radians(45)   # Left forward
        mixed_angles['right_shoulder_pitch_joint'] = np.radians(-90) # Right up
        self.apply_joint_angles(mixed_angles)
        self.step_simulation(1000)
        self.check_arm_positions("MIXED POSE")
    
    def step_simulation(self, steps):
        """Step the simulation for the given number of steps"""
        for _ in range(steps):
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
            self.viewer.sync()
    
    def check_arm_positions(self, pose_name):
        """Check if arms are in expected positions"""
        # Get shoulder and elbow positions
        left_shoulder_pos = self.mujoco_data.xpos[self.body_indices['left_shoulder']]
        right_shoulder_pos = self.mujoco_data.xpos[self.body_indices['right_shoulder']]
        left_elbow_pos = self.mujoco_data.xpos[self.body_indices['left_elbow']]
        right_elbow_pos = self.mujoco_data.xpos[self.body_indices['right_elbow']]
        
        print(f"  Left shoulder: [{left_shoulder_pos[0]:.2f}, {left_shoulder_pos[1]:.2f}, {left_shoulder_pos[2]:.2f}]")
        print(f"  Left elbow:    [{left_elbow_pos[0]:.2f}, {left_elbow_pos[1]:.2f}, {left_elbow_pos[2]:.2f}]")
        print(f"  Right shoulder:[{right_shoulder_pos[0]:.2f}, {right_shoulder_pos[1]:.2f}, {right_shoulder_pos[2]:.2f}]")
        print(f"  Right elbow:   [{right_elbow_pos[0]:.2f}, {right_elbow_pos[1]:.2f}, {right_elbow_pos[2]:.2f}]")
        
        # Check if arms are folded (elbows should be lower than shoulders)
        if pose_name == "HANDS ON WAIST":
            left_folded = left_elbow_pos[2] < left_shoulder_pos[2] - 0.1
            right_folded = right_elbow_pos[2] < right_shoulder_pos[2] - 0.1
            if left_folded and right_folded:
                print("  SUCCESS! ARMS ARE FOLDED! Robot can bend elbows!")
                print("  Hands on waist pose achieved")
            else:
                print("  Arms not folded - check joint angles")
        elif pose_name == "ARMS UP":
            left_up = left_elbow_pos[2] > left_shoulder_pos[2] + 0.1
            right_up = right_elbow_pos[2] > right_shoulder_pos[2] + 0.1
            if left_up and right_up:
                print("  SUCCESS! ARMS ARE UP! Robot can raise arms!")
            else:
                print("  Arms not up - check joint angles")
        elif pose_name == "MIXED POSE":
            left_folded = left_elbow_pos[2] < left_shoulder_pos[2] - 0.1
            right_up = right_elbow_pos[2] > right_shoulder_pos[2] + 0.1
            if left_folded and right_up:
                print("  SUCCESS! Mixed pose achieved - left folded, right up")
            else:
                print("  Mixed pose not working - check joint angles")
    
    def run_teleoperation(self):
        """Main teleoperation loop with HybrIK"""
        print("\nStarting teleoperation with HybrIK...")
        print("Move your arms in front of camera to control robot")
        print("Press 'q' to quit, 't' to test arm folding")

        frame_count = 0
        try:
            while self.viewer.is_running():
                # Get camera frames
                color_image, depth_image = self.camera.get_frames()
                if color_image is None:
                    self.apply_standing_pose()
                    self.step_simulation(1)
                    continue

                # Extract full HybrIK output
                result = self.h2o_extractor.extract_full_output(color_image)

                if result is not None:
                    # Map human pose to robot joints
                    joint_angles = self.map_human_to_robot(result)

                    # Apply to robot
                    self.apply_joint_angles(joint_angles)

                    # Show visualization
                    if frame_count % 10 == 0:
                        print(f"  Tracking human pose...")

                else:
                    # No person detected - maintain standing
                    self.apply_standing_pose()

                # Step simulation
                self.step_simulation(1)

                # Show camera feed
                display_img = color_image.copy()
                cv2.putText(display_img, "HybrIK Teleoperation Active", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("RealSense Feed", display_img)

                frame_count += 1

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.test_arm_folding()

        except KeyboardInterrupt:
            print("\nTeleoperation stopped by user")
        finally:
            self.cleanup()
    
    def map_human_to_robot(self, hybrik_result):
        """Map human pose from HybrIK to robot joint angles"""
        joint_angles = {}

        # Get 3D joints from HybrIK
        joints_3d = hybrik_result['joints_3d']

        # Calculate shoulder angles based on arm direction
        # Left arm
        left_shoulder = joints_3d[16]  # Left shoulder
        left_elbow = joints_3d[18]     # Left elbow
        left_wrist = joints_3d[20]     # Left wrist

        # Right arm
        right_shoulder = joints_3d[17]  # Right shoulder
        right_elbow = joints_3d[19]     # Right elbow
        right_wrist = joints_3d[21]     # Right wrist

        # Simple mapping: use vertical angle for shoulder pitch
        left_arm_vec = left_elbow - left_shoulder
        left_pitch = np.arctan2(-left_arm_vec[1], left_arm_vec[2])  # Forward/back
        joint_angles['left_shoulder_pitch_joint'] = np.clip(left_pitch, -1.5, 1.5)

        right_arm_vec = right_elbow - right_shoulder
        right_pitch = np.arctan2(-right_arm_vec[1], right_arm_vec[2])
        joint_angles['right_shoulder_pitch_joint'] = np.clip(right_pitch, -1.5, 1.5)

        # Shoulder roll based on arm elevation
        left_roll = np.arctan2(left_arm_vec[0], -left_arm_vec[1])  # Side raise
        joint_angles['left_shoulder_roll_joint'] = np.clip(left_roll, -0.5, 1.5)

        right_roll = np.arctan2(-right_arm_vec[0], -right_arm_vec[1])
        joint_angles['right_shoulder_roll_joint'] = np.clip(right_roll, -1.5, 0.5)

        # Elbow angles based on arm bend
        left_forearm_vec = left_wrist - left_elbow
        left_elbow_angle = np.arccos(np.clip(
            np.dot(left_arm_vec, left_forearm_vec) /
            (np.linalg.norm(left_arm_vec) * np.linalg.norm(left_forearm_vec) + 1e-6),
            -1.0, 1.0
        ))
        joint_angles['left_elbow_joint'] = np.clip(left_elbow_angle, 0, 1.5)

        right_forearm_vec = right_wrist - right_elbow
        right_elbow_angle = np.arccos(np.clip(
            np.dot(right_arm_vec, right_forearm_vec) /
            (np.linalg.norm(right_arm_vec) * np.linalg.norm(right_forearm_vec) + 1e-6),
            -1.0, 1.0
        ))
        joint_angles['right_elbow_joint'] = np.clip(right_elbow_angle, 0, 1.5)

        # Yaw joints default
        joint_angles['left_shoulder_yaw_joint'] = 0.0
        joint_angles['right_shoulder_yaw_joint'] = 0.0

        return joint_angles
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.camera.close()
        if self.viewer:
            self.viewer.close()
        print("Cleanup complete!")

def main():
    """Main function"""
    print("G1 TELEOPERATION WITH HybrIK")
    print("Real-time motion capture and control")
    print()

    try:
        system = G1TeleoperationSystem()

        # Give initial time for robot to stabilize
        print("\nStabilizing robot...")
        for _ in range(100):
            system.apply_standing_pose()
            system.step_simulation(1)
            system.viewer.sync()

        print("\nReady! Starting teleoperation...")
        print("Stand in front of camera and move your arms")
        print("The robot will mirror your movements")

        # Run teleoperation
        system.run_teleoperation()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
