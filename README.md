# G1 Humanoid Teleoperation

Real-time teleoperation system for Unitree G1 humanoid robot using HybrIK pose estimation and IK solver.

## Components

- **`g1_teleop_from_hybrik.py`** - Main teleoperation bridge connecting HybrIK pose extraction to G1 robot control
- **`direct_control_demo.py`** - Standalone demo with simulated or real sensor data
- **`src/teleop/robot_arm_ik.py`** - IK solver using CasADi and Pinocchio

## Setup

1. Install dependencies:
```bash
conda create -n tv python=3.8
conda activate tv
pip install mujoco pinocchio casadi meshcat
```

2. Setup HybrIK in separate directory:
```bash
git clone https://github.com/Jeff-sjtu/HybrIK.git
# Follow HybrIK installation instructions
```

## Usage

### Real-time teleoperation with RealSense camera:
```bash
conda activate tv
python g1_teleop_from_hybrik.py
```

### Run demo:
```bash
./run_demo.sh
```

## Architecture

The system uses a two-process architecture:
- HybrIK process extracts H2O keypoints from RealSense camera
- Main process performs IK solving and robot control

ASAP retargeting scale factor: 0.7517