#!/bin/bash
# SuMa++ Entrypoint Script
# Runs semantic LiDAR SLAM on ISEC dataset

set -e

# Source ROS environment
source /opt/ros/melodic/setup.bash
source /root/catkin_ws/devel/setup.bash

# Default values
SEQUENCE=${SEQUENCE:-"5th_floor"}
CONFIG=${CONFIG:-"/config/ouster_os128.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"/results/trajectories/suma_plus_plus"}

echo "============================================"
echo "SuMa++ Semantic LiDAR SLAM"
echo "============================================"
echo "Sequence: ${SEQUENCE}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Find bag file
BAG_FILE="/data/${SEQUENCE}/${SEQUENCE}.bag"
if [ ! -f "${BAG_FILE}" ]; then
    # Try alternative naming
    BAG_FILE=$(find /data/${SEQUENCE} -name "*.bag" | head -1)
fi

if [ -z "${BAG_FILE}" ] || [ ! -f "${BAG_FILE}" ]; then
    echo "ERROR: Could not find bag file for sequence ${SEQUENCE}"
    exit 1
fi

echo "Using bag file: ${BAG_FILE}"

# Run SuMa++ with semantic segmentation
# The visualizer runs in headless mode for batch processing
cd /root/catkin_ws/src/semantic_suma/build/bin

./suma \
    --config ${CONFIG} \
    --input ${BAG_FILE} \
    --output ${OUTPUT_DIR}/${SEQUENCE}.txt \
    --model ${RANGENET_MODEL_PATH} \
    --semantic \
    --no-gui

echo "============================================"
echo "SuMa++ complete. Output saved to:"
echo "  ${OUTPUT_DIR}/${SEQUENCE}.txt"
echo "============================================"
