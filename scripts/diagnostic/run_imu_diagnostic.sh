#!/bin/bash
# Run IMU diagnostic inside a ROS container
# Usage: ./run_imu_diagnostic.sh [--compare-all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Running IMU frame convention diagnostic..."
echo "Project: $PROJECT_DIR"

# Use the first bag file from 5th floor
BAG_FILE="/data/ISEC/5th_floor/2023-03-14-11-56-21_0.bag"

if [ "$1" == "--compare-all" ]; then
    ARGS="--compare-all"
else
    ARGS="--topic /imu/imu_uncompensated --samples 100"
fi

docker run --rm -it \
    -v ~/Dev/shared/datasets/ISEC:/data/ISEC:ro \
    -v "$PROJECT_DIR/scripts:/scripts:ro" \
    ros:noetic-perception \
    bash -c "pip3 install numpy -q && python3 /scripts/diagnostic/dump_imu_samples.py $BAG_FILE $ARGS"








