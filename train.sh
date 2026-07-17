#!/bin/bash
# Automatic MMHCL Training Script for Linux/Mac
# This script runs the automatic training with default settings

echo "========================================"
echo "MMHCL Automatic Training"
echo "========================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Run the automatic training script
python auto_train.py --dataset Clothing --gpu_id 0

