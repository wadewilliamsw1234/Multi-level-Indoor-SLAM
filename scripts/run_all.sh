#!/bin/bash
#
# SLAM Benchmark - Reproduce All Results
# =======================================
# This script reproduces all results from the SLAM benchmarking project.
#
# Prerequisites:
#   - Docker and nvidia-docker installed
#   - ISEC dataset at ~/Dev/shared/datasets/ISEC/
#   - Python 3.8+ with dependencies (pip install -r requirements.txt)
#
# Usage:
#   ./scripts/run_all.sh           # Run everything
#   ./scripts/run_all.sh --eval    # Only run evaluation (assumes trajectories exist)
#   ./scripts/run_all.sh --semantic # Only run semantic gating analysis
#
# Author: Wade Williams
# Course: EECE5554 Robotic Sensing and Navigation
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="${DATASET_DIR:-$HOME/Dev/shared/datasets/ISEC}"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo_step "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        echo_error "Docker not found. Please install Docker."
        exit 1
    fi

    if [ ! -d "$DATASET_DIR" ]; then
        echo_error "Dataset not found at $DATASET_DIR"
        echo "Set DATASET_DIR environment variable or download ISEC dataset."
        exit 1
    fi

    echo "  ✓ Docker available"
    echo "  ✓ Dataset found at $DATASET_DIR"
}

# Build Docker images
build_images() {
    echo_step "Building Docker images..."
    docker-compose build
}

# Run SLAM algorithms
run_lego_loam() {
    echo_step "Running LeGO-LOAM on all floors..."
    for floor in 5th 1st 4th 2nd; do
        echo "  Processing ${floor}_floor..."
        docker-compose run --rm lego-loam \
            roslaunch lego_loam run.launch \
            bag_path:=/data/${floor}_floor \
            output_path:=/results/trajectories/lego_loam/${floor}_floor.txt
    done
}

run_orb_slam3() {
    echo_step "Running ORB-SLAM3 on all floors..."
    for floor in 5th 1st 4th 2nd; do
        echo "  Processing ${floor}_floor..."
        docker-compose run --rm orb-slam3 \
            ./Examples/Stereo/stereo_euroc \
            ./Vocabulary/ORBvoc.txt \
            /config/ISEC_stereo.yaml \
            /data/${floor}_floor \
            /results/trajectories/orb_slam3/${floor}_floor.txt
    done
}

run_droid_slam() {
    echo_step "Running DROID-SLAM on all floors..."
    for floor in 5th 1st 4th 2nd; do
        echo "  Processing ${floor}_floor..."
        docker-compose run --rm droid-slam \
            python scripts/droid_slam/run_droid_slam_stereo.py \
            --datapath /data/${floor}_floor \
            --output /results/trajectories/droid_slam/${floor}_floor_stereo.txt
    done
}

# Run evaluation
run_evaluation() {
    echo_step "Running trajectory evaluation..."
    python scripts/evaluation/comprehensive_evaluation.py

    echo_step "Generating figures..."
    python scripts/evaluation/generate_figures.py
}

# Run semantic gating analysis
run_semantic_gating() {
    echo_step "Running semantic gating analysis..."

    echo "  Analyzing ORB-SLAM3..."
    python scripts/semantic_gating/orb_slam3_integration.py

    echo "  Analyzing DROID-SLAM..."
    python scripts/semantic_gating/droid_slam_integration.py

    echo "  Analyzing LeGO-LOAM..."
    python scripts/semantic_gating/lego_loam_integration.py

    echo_step "Semantic gating analysis complete!"
    echo "  Results saved to: results/semantic_gating/"
}

# Main execution
main() {
    case "${1:-all}" in
        --eval)
            check_prerequisites
            run_evaluation
            ;;
        --semantic)
            check_prerequisites
            run_semantic_gating
            ;;
        --slam)
            check_prerequisites
            build_images
            run_lego_loam
            run_orb_slam3
            run_droid_slam
            ;;
        all|*)
            check_prerequisites
            build_images
            run_lego_loam
            run_orb_slam3
            run_droid_slam
            run_evaluation
            run_semantic_gating
            ;;
    esac

    echo ""
    echo_step "All tasks complete!"
    echo ""
    echo "Results summary:"
    echo "  - Trajectories: results/trajectories/"
    echo "  - Metrics: results/metrics/"
    echo "  - Figures: results/figures/"
    echo "  - Semantic gating: results/semantic_gating/"
    echo ""
    echo "See results/BENCHMARK_RESULTS_SUMMARY.md for detailed analysis."
}

main "$@"
