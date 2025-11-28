#!/bin/bash
# Run ORB-SLAM3 with visualization via VNC
# Usage: ./run_orb_slam3_viz.sh [floor] [rate] [duration]

FLOOR=${1:-5th_floor}
RATE=${2:-0.3}
DURATION=${3:-180}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Starting ORB-SLAM3 Visualization Container"
echo "==========================================="
echo "Floor: ${FLOOR}"
echo "Playback rate: ${RATE}x"
echo "Duration: ${DURATION}s"
echo ""
echo "Connect VNC viewer to: localhost:5900"
echo "  Example: vncviewer localhost:5900"
echo ""

docker run --rm -it \
    -p 5900:5900 \
    -v ~/Dev/shared/datasets/ISEC:/data/ISEC:ro \
    -v ${PROJECT_ROOT}/results:/results \
    -v ${PROJECT_ROOT}/config:/config:ro \
    slam-benchmark/orb-slam3-viz:latest \
    /root/start_interactive.sh
