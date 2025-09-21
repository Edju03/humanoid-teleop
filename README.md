# G1 HybrIK Teleoperation

Real-time teleoperation system for Unitree G1 robot using HybrIK pose estimation.

## Requirements

- HybrIK framework
- RealSense camera (D435/D435i)
- CUDA-capable GPU
- Python packages: `torch`, `mujoco`, `pyrealsense2`, `opencv-python`

## Usage

```bash
# HybrIK pose estimation
./run_demo.sh 1

# G1 teleoperation with MuJoCo
./run_demo.sh 2
```

## Model Configuration

**23 DOF** (Optimized for vision-based teleoperation)

**Legs (12 DOF total):**
- Left/Right hip: pitch, roll, yaw joints
- Left/Right knee: single axis joint
- Left/Right ankle: pitch, roll joints

**Waist (3 DOF):**
- waist_yaw, waist_roll, waist_pitch joints

**Arms (8 DOF total):**
- Left/Right shoulder: pitch, roll, yaw joints
- Left/Right elbow: single axis joint
- No wrist control (HybrIK provides 3D position only, not rotation)

## System Architecture

```
Simulation:
Camera → HybrIK → Joint Mapping → MuJoCo
         (30 FPS)   (23 DOF)       (Physics)

Hardware (Target):
Camera → HybrIK → Joint Mapping → ROS2/SDK → G1 Robot
         (30 FPS)   (23 DOF)       (Control)   (Hardware)
```