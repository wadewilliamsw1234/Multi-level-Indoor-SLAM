# SLAM Benchmarking on Multi-Floor Indoor Environments

**Course:** EECE 5554 - Robotic Sensing and Navigation
**Author:** Wade Williams
**Date:** December 2024

## Project Overview

This project benchmarks state-of-the-art SLAM algorithms on the NUFR-M3F multi-floor indoor dataset (Kaveti et al., IEEE CASE 2023). We evaluate multiple algorithms across three categories—base SLAM, semantic SLAM, and our custom semantic gating pipeline—to address **perceptual aliasing**, a critical failure mode in buildings with structurally similar floors.

**Key Finding:** All tested algorithms (visual, deep learning, and LiDAR) suffer from perceptual aliasing in multi-floor environments. Our semantic gating pipeline prevents 62-75% of false loop closure candidates with negligible computational overhead.

## Implemented Algorithms

### Base SLAM (from Kaveti et al.)
| Algorithm | Type | Status | Description |
|-----------|------|--------|-------------|
| LeGO-LOAM | LiDAR | ✅ Running | Pseudo ground-truth |
| ORB-SLAM3 | Visual Stereo | ✅ Running | Feature-based SLAM |
| DROID-SLAM | Deep Learning | ✅ Running | Learned correlation volumes |
| Basalt | VIO | ⚠️ Experimental | Visual-inertial odometry |
| VINS-Fusion | VIO | ⚠️ Experimental | Visual-inertial SLAM |

### Semantic SLAM (State-of-the-Art)
| Algorithm | Type | Status | Key Feature |
|-----------|------|--------|-------------|
| Kimera + GNC | Semantic VIO | ✅ Ready | 70-80% outlier tolerance via Graduated Non-Convexity |
| S-Graphs 2.0 | Hierarchical | ✅ Ready | **Floor-level factor graph** - designed for multi-floor |
| SuMa++ | Semantic LiDAR | ✅ Ready | RangeNet++ dynamic object filtering |
| YOLOv8-ORB-SLAM3 | Dynamic Filtering | ✅ Ready | Real-time person/vehicle masking |

### Foundation Model VPR (Visual Place Recognition)
| Method | Backbone | VRAM | Best For |
|--------|----------|------|----------|
| MixVPR | ResNet-50 | ~2-3GB | Fast single-stage retrieval |
| SALAD | DINOv2 | ~3-4GB | Optimal transport aggregation |
| AnyLoc | DINOv2 | ~4GB | Universal, no training needed |
| **CricaVPR** | DINOv2 | ~4GB | **Perceptual aliasing robustness** (CVPR 2024) |

### Geometric Verification
| Method | Speed | Description |
|--------|-------|-------------|
| LightGlue | 150 FPS | Fast adaptive matching |
| SuperGlue | 15 FPS | Attention-based matching |
| LoFTR | 10 FPS | Dense detector-free matching |

## Quick Start

```bash
# Clone and enter repository
cd ~/Dev/ros1/slam-benchmark

# Run semantic gating analysis (our main contribution)
python scripts/semantic_gating/orb_slam3_integration.py
python scripts/semantic_gating/droid_slam_integration.py
python scripts/semantic_gating/lego_loam_integration.py

# View results
cat results/semantic_gating/semantic_gating_comparison.txt

# Run with CricaVPR (best for perceptual aliasing)
python -c "
from scripts.semantic_gating import SemanticPlaceRecognition
spr = SemanticPlaceRecognition(vpr_method='cricavpr', device='cuda')
"
```

## Repository Structure

```
slam-benchmark/
├── README.md                     # This file
├── docker-compose.yml            # Docker orchestration (12 services)
│
├── docker/                       # Dockerfiles for each algorithm
│   ├── Dockerfile.lego-loam      # LiDAR SLAM (pseudo ground-truth)
│   ├── Dockerfile.orb-slam3      # Visual stereo SLAM
│   ├── Dockerfile.droid-slam     # Deep learning SLAM
│   ├── Dockerfile.basalt         # Visual-inertial odometry
│   ├── Dockerfile.vins-fusion    # VIO (experimental)
│   ├── Dockerfile.kimera         # MIT Kimera VIO + RPGO (GNC)
│   ├── Dockerfile.s-graphs       # S-Graphs 2.0 (multi-floor specialized)
│   ├── Dockerfile.suma-plus-plus # Semantic LiDAR SLAM
│   ├── Dockerfile.yolo-orb-slam3 # Dynamic object filtering
│   └── Dockerfile.semantic-tools # VPR + geometric verification
│
├── config/                       # Algorithm configurations
│   ├── lego_loam/                # Ouster OS-128 config
│   ├── orb_slam3/                # Stereo camera calibration
│   ├── kimera/                   # Kimera + GNC parameters
│   ├── s_graphs/                 # S-Graphs 2.0 floor detection config
│   └── suma_plus_plus/           # Semantic LiDAR config
│
├── scripts/
│   ├── run_all.sh                # Reproduce all results
│   │
│   ├── semantic_gating/          # ⭐ OUR CONTRIBUTION
│   │   ├── floor_detector.py     # IMU-based elevator detection
│   │   ├── lidar_floor_tracker.py # LiDAR ground plane floor detection
│   │   ├── loop_closure_gate.py  # Semantic loop closure filtering
│   │   ├── place_recognition.py  # MixVPR, SALAD, AnyLoc, CricaVPR
│   │   ├── geometric_verification.py # LightGlue, SuperGlue, LoFTR
│   │   ├── orb_slam3_integration.py
│   │   ├── droid_slam_integration.py
│   │   └── lego_loam_integration.py
│   │
│   └── evaluation/
│       ├── comprehensive_evaluation.py
│       └── semantic_evaluation.py  # Semantic-specific metrics
│
├── results/
│   ├── trajectories/             # TUM-format outputs
│   ├── semantic_gating/          # Cross-floor analysis + visualizations
│   └── metrics/                  # JSON evaluation results
│
└── archive/                      # Deprecated/experimental code
```

## Running the Benchmarks

### Option 1: Complete Pipeline
```bash
./scripts/run_all.sh
```

### Option 2: Individual Algorithms
```bash
docker-compose build
docker-compose run --rm lego-loam         # LiDAR (ground truth)
docker-compose run --rm orb-slam3         # Visual
docker-compose run --rm droid-slam        # Deep Learning (GPU)
docker-compose run --rm kimera            # Semantic VIO + GNC
docker-compose run --rm s-graphs          # Multi-floor specialized
docker-compose run --rm suma-plus-plus    # Semantic LiDAR (GPU)
docker-compose run --rm yolo-orb-slam3    # Dynamic filtering (GPU)
```

### Option 3: Semantic Post-Processing
```bash
docker-compose run --rm semantic-tools /root/run_semantic_postprocess.sh all
```

## Results Summary

### Algorithm Performance (ATE RMSE vs LeGO-LOAM)

| Algorithm | 5th Floor | 1st Floor | 4th Floor | 2nd Floor | Mean |
|-----------|-----------|-----------|-----------|-----------|------|
| ORB-SLAM3 | 10.46m | 8.82m | 1.96m | 5.54m | **6.70m** |
| DROID-SLAM | 0.25m | 0.16m | 0.23m | 0.65m | **0.32m** |

### Semantic Gating Results

| Algorithm | Type | Cross-Floor Rate | False Positives Blocked |
|-----------|------|------------------|------------------------|
| LeGO-LOAM | LiDAR (ICP) | **75.3%** | 65,567 |
| ORB-SLAM3 | Visual (DBoW2) | 70.7% | 3,612,527 |
| DROID-SLAM | Deep Learning | 62.7% | 59,333 |

**Key Insight:** LiDAR-based methods are *more* susceptible to cross-floor aliasing than visual methods due to the geometric similarity of identical floor layouts.

## Semantic Gating Module

Our contribution addresses perceptual aliasing in multi-floor buildings through a multi-modal pipeline:

### Floor Detection (IMU + LiDAR Fusion)
```python
from scripts.semantic_gating import IMUFloorDetector, LiDARFloorTracker

# IMU-based elevator detection
imu_detector = IMUFloorDetector(z_accel_threshold=0.3, min_duration=2.0)
events = imu_detector.detect_elevator_events(timestamps, accel_x, accel_y, accel_z)
floor_labels = imu_detector.assign_floor_labels(timestamps, start_floor=0)

# LiDAR ground plane tracking
lidar_tracker = LiDARFloorTracker(floor_height=3.5)
estimate = lidar_tracker.process_scan(pointcloud)  # Returns z_height, floor_estimate
```

### Foundation Model Place Recognition
```python
from scripts.semantic_gating import SemanticPlaceRecognition, CricaVPR

# CricaVPR - BEST for perceptual aliasing (CVPR 2024)
spr = SemanticPlaceRecognition(vpr_method='cricavpr', device='cuda')

# Add images with floor labels
spr.add_image(image, timestamp, floor_label=1)

# Find loop closures with floor gating
matches = spr.find_loop_closures(enable_floor_gating=True)
```

### Geometric Verification
```python
from scripts.semantic_gating import LightGlue, SemanticGeometricVerifier

# Fast geometric verification
verifier = LightGlue(device='cuda')
result = verifier.match(image1, image2)
# Returns: num_matches, inlier_ratio, is_valid
```

## Key Innovations

### 1. Graduated Non-Convexity (GNC) for Outlier Rejection
Kimera-RPGO with GNC handles 70-80% outlier measurements—critical for multi-floor SLAM where perceptual aliasing creates correlated outliers that defeat standard RANSAC.

### 2. S-Graphs 2.0 Hierarchical Factor Graph
The **only** algorithm specifically designed for multi-floor SLAM with floor-level constraints:
- 4-layer factor graph: Keyframes → Walls → Rooms → Floors
- Floor-based loop closure gating built into optimization
- Stairway/elevator detection for floor transitions

### 3. CricaVPR for Perceptual Aliasing
CVPR 2024 method that explicitly addresses perceptual aliasing through cross-image correlation-aware features. Distinguishes similar-but-different places using local spatial structure.

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- 32GB+ RAM recommended
- 100GB+ disk space for dataset and Docker images

### Software
- Docker 20.10+
- nvidia-docker2 (for GPU support)
- Python 3.8+

### Dataset
Download the NUFR-M3F dataset and place at `~/Dev/shared/datasets/ISEC/`:
```
ISEC/
├── 1st_floor/          # Challenging: glass walls, reflections
├── 2nd_floor/          # Figure-eight trajectory
├── 4th_floor/          # Single loop
├── 5th_floor/          # Main test sequence (187m)
├── transit_1_to_4/     # Elevator traversal
├── transit_5_to_1/     # Elevator traversal
└── *.yaml              # Calibration files
```

## References

1. Kaveti, P., et al. "Challenges of Indoor SLAM: A Multi-Modal Multi-Floor Dataset for SLAM Evaluation." IEEE CASE 2023.
2. Shan, T., Englot, B. "LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping." IROS 2018.
3. Campos, C., et al. "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM." IEEE T-RO 2021.
4. Teed, Z., Deng, J. "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras." NeurIPS 2021.
5. Rosinol, A., et al. "Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization and Mapping." ICRA 2020.
6. Bavle, H., et al. "S-Graphs+: Real-time Localization and Mapping leveraging Hierarchical Representations." RA-L 2023.
7. Lu, F., et al. "CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition." CVPR 2024.
8. Yang, H., et al. "Graduated Non-Convexity for Robust Spatial Perception." IEEE T-RO 2020.

## License

MIT License - Academic use for EECE 5554 Final Project.
