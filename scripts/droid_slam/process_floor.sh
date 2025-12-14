#!/bin/bash
# Process a single floor with DROID-SLAM
# Usage: ./process_floor.sh <floor_name>

FLOOR=${1:-5th_floor}
DATA_DIR=/data/ISEC/${FLOOR}
WORK_DIR=/workspace/${FLOOR}
OUTPUT=/results/trajectories/droid_slam/${FLOOR}.txt

echo "=== Processing ${FLOOR} with DROID-SLAM ==="

# Create output directory
mkdir -p /results/trajectories/droid_slam
mkdir -p ${WORK_DIR}

# Step 1: Extract images (if not already done)
if [ ! -d "${WORK_DIR}/left" ]; then
    echo "Extracting stereo images..."
    python3 /opt/scripts/extract_stereo_images.py \
        ${DATA_DIR} ${WORK_DIR} \
        --left-topic /camera_array/cam1/image_raw \
        --right-topic /camera_array/cam3/image_raw
else
    echo "Images already extracted, skipping..."
fi

# Step 2: Run DROID-SLAM
echo "Running DROID-SLAM..."
python3 /opt/scripts/run_droid_slam.py ${WORK_DIR} ${OUTPUT} --stride 2

echo "=== Done! Output: ${OUTPUT} ==="
