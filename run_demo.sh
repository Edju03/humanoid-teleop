#!/bin/bash

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate conda environment
conda activate tv

# Run direct control demo (works with simulated data if no camera)
python direct_control_demo.py