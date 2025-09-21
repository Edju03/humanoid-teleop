# G1 HybrIK Teleoperation

Real-time teleoperation system for Unitree G1 robot using HybrIK pose estimation.

## Requirements

- HybrIK installed in `/home/vb/workspace/HybrIK`
- RealSense camera (D435/D435i)
- CUDA-capable GPU
- Python packages: `torch`, `mujoco`, `pyrealsense2`, `opencv-python`

## Usage

```bash
# Pose estimation demo
./run_demo.sh 1

# G1 teleoperation with MuJoCo
./run_demo.sh 2
```

## Model Configuration

**23 DOF** (Optimized for vision-based teleoperation)
- 12 DOF legs (6 per leg)
- 3 DOF waist
- 8 DOF arms (4 per arm, no wrists)

Wrists are disabled since HybrIK cannot extract wrist rotation from vision alone. This matches the H2O/ASAP teleoperation design.

## System Architecture

```
Simulation:
Camera → HybrIK → Joint Mapping → MuJoCo
         (30 FPS)   (23 DOF)       (Physics)

Hardware (Target):
Camera → HybrIK → Joint Mapping → ROS2/SDK → G1 Robot
         (30 FPS)   (23 DOF)       (Control)   (Hardware)
```