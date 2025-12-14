#!/bin/bash
# Run Basalt VIO on ISEC dataset in EuRoC format

FLOOR=$1
WITH_LC=${2:-""}

if [ -z "$FLOOR" ]; then
    echo "Usage: $0 <floor> [--with-loop-closure]"
    echo "  floor: 5th_floor, 1st_floor, 4th_floor, 2nd_floor"
    exit 1
fi

# Paths
DATA_DIR="/data/euroc/${FLOOR}"
CALIB="/config/basalt/isec_calib.json"
OUTPUT_DIR="/results/trajectories/basalt"
LOG_DIR="/results/logs"

# Select config based on loop closure flag
if [ "$WITH_LC" == "--with-loop-closure" ]; then
    ACTIVE_CONFIG="/config/basalt/isec_vio_config_with_lc.json"
    OUTPUT_FILE="${OUTPUT_DIR}/${FLOOR}_with_lc.txt"
    LOG_FILE="${LOG_DIR}/basalt_${FLOOR}_with_lc.log"
    LC_STATUS="enabled"
else
    ACTIVE_CONFIG="/config/basalt/isec_vio_config.json"
    OUTPUT_FILE="${OUTPUT_DIR}/${FLOOR}.txt"
    LOG_FILE="${LOG_DIR}/basalt_${FLOOR}.log"
    LC_STATUS="disabled"
fi

echo "============================================"
echo "Basalt VIO: Processing ${FLOOR}"
echo "Loop closure: ${LC_STATUS}"
echo "============================================"

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

echo "Running Basalt VIO..."
echo "Config: ${ACTIVE_CONFIG}"

# Run Basalt VIO with GUI disabled
cd /tmp
/opt/basalt/build/basalt_vio \
    --show-gui 0 \
    --dataset-path ${DATA_DIR} \
    --cam-calib ${CALIB} \
    --dataset-type euroc \
    --config-path ${ACTIVE_CONFIG} \
    --marg-data /tmp/basalt_marg_${FLOOR} \
    --save-trajectory tum \
    --use-imu 1 \
    2>&1 | tee ${LOG_FILE}

# Check for output trajectory
if [ -f "/tmp/trajectory.txt" ]; then
    mv /tmp/trajectory.txt ${OUTPUT_FILE}
    echo "============================================"
    echo "Trajectory saved to: ${OUTPUT_FILE}"
    LINES=$(wc -l < ${OUTPUT_FILE})
    echo "Poses: ${LINES}"
    echo "============================================"
else
    echo "ERROR: No trajectory output found"
    exit 1
fi
