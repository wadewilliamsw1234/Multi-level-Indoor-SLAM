#!/bin/bash
# Run LeGO-LOAM on remaining ISEC floor sequences
# Based on working run_lego_loam_full.sh from 5th_floor success
#
# Usage: 
#   ./run_remaining_floors.sh          # Run all remaining floors
#   ./run_remaining_floors.sh 1st      # Run specific floor

set -e

SLAM_DIR="/home/wadewilliams/Dev/ros1/slam-benchmark"
DATA_DIR="/home/wadewilliams/Dev/shared/datasets/ISEC"
RESULTS_DIR="$SLAM_DIR/results"

# Check arguments
if [ -n "$1" ]; then
    FLOORS=("${1}_floor")
else
    FLOORS=("1st_floor" "4th_floor" "2nd_floor")
fi

echo "========================================"
echo "LeGO-LOAM Processing: ${FLOORS[*]}"
echo "========================================"

for FLOOR in "${FLOORS[@]}"; do
    echo ""
    echo "========== Processing: $FLOOR =========="
    
    BAG_DIR="$DATA_DIR/$FLOOR"
    OUTPUT_BAG="$RESULTS_DIR/lego_loam_${FLOOR}.bag"
    
    # Check if already exists
    if [ -f "$OUTPUT_BAG" ]; then
        echo "Output exists: $OUTPUT_BAG"
        echo "Delete to reprocess, or skipping..."
        continue
    fi
    
    # Check bags exist
    BAG_COUNT=$(ls -1 "$BAG_DIR"/*.bag 2>/dev/null | wc -l)
    echo "Found $BAG_COUNT bags in $BAG_DIR"
    
    if [ "$BAG_COUNT" -eq 0 ]; then
        echo "ERROR: No bags found!"
        continue
    fi
    
    # Run in Docker container
    echo "Starting LeGO-LOAM container..."
    
    docker run --rm -it \
        --name lego_loam_run \
        -v "$DATA_DIR:/data/ISEC:ro" \
        -v "$RESULTS_DIR:/results" \
        --network=host \
        lego-loam:melodic \
        bash -c "
            set -e
            source /opt/ros/melodic/setup.bash
            source /catkin_ws/devel/setup.bash
            
            echo 'Starting roscore...'
            roscore &
            sleep 3
            
            echo 'Starting LeGO-LOAM nodes...'
            rosrun lego_loam imageProjection __name:=imageProjection &
            sleep 1
            rosrun lego_loam featureAssociation __name:=featureAssociation &
            sleep 1  
            rosrun lego_loam mapOptmization __name:=mapOptmization &
            sleep 1
            rosrun lego_loam transformFusion __name:=transformFusion &
            sleep 3
            
            echo 'Starting trajectory recording...'
            rosbag record -O /results/lego_loam_${FLOOR}.bag \
                /aft_mapped_to_init \
                /laser_odom_to_init \
                /integrated_to_init &
            RECORD_PID=\$!
            sleep 2
            
            echo 'Playing bags sequentially...'
            for bag in /data/ISEC/${FLOOR}/*.bag; do
                echo \"  Playing: \$(basename \$bag)\"
                rosbag play --clock \"\$bag\" -r 1.0 -q
            done
            
            echo 'Playback complete, waiting for processing...'
            sleep 5
            
            echo 'Stopping recording...'
            kill \$RECORD_PID 2>/dev/null || true
            sleep 2
            
            echo 'Done with ${FLOOR}!'
        "
    
    # Verify output
    if [ -f "$OUTPUT_BAG" ]; then
        SIZE=$(du -h "$OUTPUT_BAG" | cut -f1)
        echo "SUCCESS: $OUTPUT_BAG ($SIZE)"
    else
        echo "ERROR: Output bag not created!"
    fi
done

echo ""
echo "========================================"
echo "Processing complete!"
echo "========================================"
echo ""
echo "Output bags:"
ls -lh "$RESULTS_DIR"/lego_loam_*.bag 2>/dev/null

echo ""
echo "Next: Extract trajectories with:"
echo "  python3 scripts/extract_trajectory.py"
