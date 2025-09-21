#!/bin/bash

# Set HybrIK path (default: assumes HybrIK is in parent directory)
HYBRIK_PATH=${HYBRIK_PATH:-"/home/vb/workspace/HybrIK"}
export HYBRIK_PATH

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to HybrIK directory for SMPL models
cd "$HYBRIK_PATH"

if [ "$1" = "1" ]; then
    python "$SCRIPT_DIR/src/hybrik/realsense_demo.py"
elif [ "$1" = "2" ]; then
    python "$SCRIPT_DIR/src/hybrik/g1_teleop_mujoco.py"
else
    echo "Usage: $0 [1|2]"
    echo "  1: RealSense demo"
    echo "  2: G1 teleoperation"
fi