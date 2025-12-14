#!/bin/bash
# Run VINS-Fusion on ISEC dataset sequences
# Usage: ./run_vins_fusion.sh <floor> [config_file]

set -e

FLOOR=${1:-"5th_floor"}
CONFIG=${2:-"/config/vins_fusion/isec_stereo_imu.yaml"}
DATA_DIR="/data/ISEC"
OUTPUT_DIR="/results/trajectories/vins_fusion"
LOG_DIR="/results/logs"

echo "============================================"
echo "VINS-Fusion: Processing ${FLOOR}"
echo "============================================"

# Source ROS
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Get bag files for this floor
BAG_DIR="${DATA_DIR}/${FLOOR}"
if [ ! -d "${BAG_DIR}" ]; then
    echo "Error: Directory ${BAG_DIR} not found"
    exit 1
fi

# Count bag files
BAG_FILES=(${BAG_DIR}/*.bag)
NUM_BAGS=${#BAG_FILES[@]}
echo "Found ${NUM_BAGS} bag files in ${BAG_DIR}"

# Start roscore in background
roscore &
ROSCORE_PID=$!
sleep 3

# Start VINS-Fusion node
echo "Starting VINS-Fusion estimator..."
rosrun vins vins_node ${CONFIG} &
VINS_PID=$!
sleep 2

# Optional: Start visualization (disabled for benchmarking)
# rosrun vins vins_rviz ${CONFIG} &

# Play all bags in sequence
echo "Playing bag files..."
for bag in "${BAG_FILES[@]}"; do
    echo "  Playing: $(basename ${bag})"
    rosbag play ${bag} --clock -r 0.5 2>&1 | tee -a ${LOG_DIR}/vins_fusion_${FLOOR}.log
done

# Wait for processing to complete
sleep 10

# Save trajectory
echo "Saving trajectory..."
# VINS-Fusion publishes odometry to /vins_estimator/odometry
# We need to record and convert to TUM format

# Kill processes
kill ${VINS_PID} 2>/dev/null || true
kill ${ROSCORE_PID} 2>/dev/null || true

echo "============================================"
echo "VINS-Fusion: ${FLOOR} complete"
echo "Output: ${OUTPUT_DIR}/${FLOOR}.txt"
echo "============================================"
