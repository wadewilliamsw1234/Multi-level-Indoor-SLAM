# NUFR-M3F SLAM Benchmarking Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS: Noetic](https://img.shields.io/badge/ROS-Noetic-blue.svg)](http://wiki.ros.org/noetic)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)

A Docker-based benchmarking framework for evaluating state-of-the-art SLAM algorithms on the ISEC multi-floor indoor dataset. This project reproduces and extends the analysis from **"Challenges of Indoor SLAM: A Multi-Modal Multi-Floor Dataset for SLAM Evaluation"** (Kaveti et al., IEEE CASE 2023).

## 🎯 Project Goals

1. **Reproduce Paper Results**: Validate Table IV metrics for multiple SLAM algorithms
2. **Dockerized Pipeline**: Self-contained, reproducible evaluation environment
3. **Multi-Algorithm Comparison**: Side-by-side evaluation of visual, visual-inertial, LiDAR, and deep learning SLAM
4. **Highlight Failure Modes**: Demonstrate perceptual aliasing, visual degradation, and multi-floor challenges
5. **Generate Publication Figures**: Reproduce Figure 6 (perceptual aliasing) and Figure 7 (trajectory comparison)

## 📊 Current Results

### Implemented Algorithms

| Algorithm | Type | Status | 5th Floor | 1st Floor | 4th Floor | 2nd Floor |
|-----------|------|--------|-----------|-----------|-----------|-----------|
| **LeGO-LOAM** | LiDAR | ✅ Complete | 0.75m (0.40%) | 0.33m (0.45%) | 0.14m (0.21%) | 0.11m (0.08%) |
| **ORB-SLAM3** | Visual | ✅ Complete | 0.70m (0.46%) | ❌ 12.0m | 0.57m (0.97%) | 1.97m (1.84%) |
| **VINS-Fusion** | Visual-Inertial | 🔄 In Progress | - | - | - | - |
| **Basalt** | Visual-Inertial | 📋 Planned | - | - | - | - |
| **DROID-SLAM** | Deep Learning | 📋 Planned | - | - | - | - |

*Drift values shown as absolute error (percentage of trajectory length)*

### Paper Comparison

| Algorithm | Metric | Our Results | Paper Results | Assessment |
|-----------|--------|-------------|---------------|------------|
| LeGO-LOAM | 4th Floor ATE | **0.14m** | 0.79m | 5.6× better |
| LeGO-LOAM | 2nd Floor ATE | **0.11m** | 0.29m | 2.6× better |
| ORB-SLAM3 | 5th Floor ATE | 0.70m | 0.52m | Within 35% |
| ORB-SLAM3 | 1st Floor | Failed (tracking loss) | 0.95m | Expected* |

*\*The 1st floor contains challenging scenarios (glass, reflections, dynamic objects) that cause visual SLAM failure - this matches the paper's discussion of failure modes.*

## 🚀 Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU algorithms)
- ~50GB disk space for Docker images
- ISEC dataset (~500GB) from [NUFR-M3F Repository](https://github.com/neufieldrobotics/NUFR-M3F)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/slam-benchmark.git
cd slam-benchmark

# Create data directory and download/symlink ISEC dataset
mkdir -p data
ln -s /path/to/your/ISEC data/ISEC

# Build Docker images
make build

# Run LeGO-LOAM on all floors (uses LiDAR, most reliable)
docker run --rm -it \
    -v $(pwd)/data/ISEC:/data/ISEC:ro \
    -v $(pwd)/results:/results \
    --network=host \
    slam-benchmark/lego-loam:latest \
    /root/run_lego_loam_floor.sh 5th_floor

# Run ORB-SLAM3 on all floors (stereo visual)
docker run --rm -it \
    -v $(pwd)/data/ISEC:/data/ISEC:ro \
    -v $(pwd)/results:/results \
    -v $(pwd)/config:/config:ro \
    --network=host \
    slam-benchmark/orb-slam3:latest \
    /root/run_orb_slam3_all.sh

# Evaluate results
python3 scripts/evaluation/evaluate_all.py
```

## 📁 Project Structure

```
slam-benchmark/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── Makefile                      # Build automation
├── docker-compose.yml            # Container orchestration
├── .gitignore                    # Git ignore rules
│
├── docker/                       # Dockerfiles for each algorithm
│   ├── Dockerfile.ros-base       # Base ROS Noetic image
│   ├── Dockerfile.lego-loam      # LeGO-LOAM (LiDAR SLAM)
│   ├── Dockerfile.orb-slam3      # ORB-SLAM3 (Visual SLAM)
│   └── Dockerfile.evaluation     # Evaluation tools
│
├── config/                       # Algorithm configurations
│   ├── lego_loam/
│   │   └── isec_ouster.yaml      # Ouster OS-128 parameters
│   ├── orb_slam3/
│   │   └── ISEC_stereo.yaml      # Stereo camera calibration
│   ├── vins_fusion/
│   │   └── isec_stereo_imu.yaml  # Stereo-inertial config
│   ├── basalt/
│   │   └── isec_config.json      # VIO configuration
│   └── calibration_examples/     # Example calibration files
│
├── scripts/                      # Analysis and utility scripts
│   ├── extraction/               # Trajectory extraction
│   ├── evaluation/               # Metrics computation
│   ├── visualization/            # Plotting and figures
│   ├── utils/                    # Calibration, bag utilities
│   └── docker/                   # Container run scripts
│
├── results/                      # Generated outputs
│   ├── trajectories/             # TUM-format trajectory files
│   │   ├── lego_loam/
│   │   └── orb_slam3/
│   ├── figures/                  # Visualization outputs
│   ├── metrics/                  # Evaluation results (JSON)
│   └── logs/                     # Algorithm runtime logs
│
├── tests/                        # Unit tests
│
├── docs/                         # Additional documentation
│   └── ORB_SLAM3.md              # ORB-SLAM3 specific notes
│
└── data/                         # Dataset mount point (gitignored)
    └── ISEC/                     # Symlink to ISEC dataset
```

## 📖 Dataset

The ISEC dataset was collected at Northeastern University's Interdisciplinary Science and Engineering Complex using a mobile robot platform equipped with:

| Sensor | Model | Specifications |
|--------|-------|----------------|
| Cameras | FLIR Blackfly S (×7) | 720×540, 20Hz, global shutter |
| LiDAR | Ouster OS-128 | 128 channels, 45° VFOV, 10Hz |
| IMU | VectorNav VN-100 | 9-DOF, 200Hz |

### Floor Sequences

| Sequence | Duration | Length | Description |
|----------|----------|--------|-------------|
| 5th_floor | 437s | 187m | One loop + out-and-back corridor |
| 1st_floor | 126s | 65m | Open layout, glass walls, dynamic |
| 4th_floor | 131s | 66m | One loop, some dynamic content |
| 2nd_floor | 266s | 128m | Figure-eight with two loops |

### Key Challenges

- **Perceptual Aliasing**: Floors 2, 4, and 5 are visually nearly identical
- **Visual Degradation**: Featureless corridors, glass surfaces, reflections
- **Dynamic Content**: Moving people, especially on 1st floor
- **Elevator Transits**: Vision-only methods cannot track vertical motion

## 🔧 Algorithms

### ✅ LeGO-LOAM (Complete)

Lightweight LiDAR odometry optimized for ground vehicles. Adapted from Velodyne to Ouster OS-128:

- **Key Modifications**: N_SCAN: 16→128, vertical FOV: 30°→45°, ring field extraction
- **Performance**: Excellent on all floors, used as pseudo ground-truth
- **Docker Image**: `slam-benchmark/lego-loam:latest`

### ✅ ORB-SLAM3 (Complete)

Feature-based stereo visual SLAM using cam1 (left) and cam3 (right):

- **Configuration**: Older `Camera.fx` format (not `Camera1.fx`)
- **Performance**: 3/4 floors successful, 1st floor fails due to visual challenges
- **Loop Closure**: Disabled for fair comparison
- **Docker Image**: `slam-benchmark/orb-slam3:latest`

### 🔄 VINS-Fusion (In Progress)

Stereo visual-inertial odometry expected to handle 1st floor better:

- **Mode**: Stereo + IMU fusion
- **Config**: Ready at `config/vins_fusion/isec_stereo_imu.yaml`

### 📋 Basalt (Planned)

Visual-inertial odometry critical for **Figure 6** (perceptual aliasing):

- Must run with AND without loop closure
- Demonstrates incorrect cross-floor loop closures

### 📋 DROID-SLAM (Planned)

Deep learning-based SLAM, best performer in paper:

- Expected to handle textureless regions better
- Requires GPU (fits in 8GB VRAM)

## 📈 Evaluation Methodology

Following the paper's approach (Section V-A):

1. **Trajectory Alignment**: SE(3) alignment using initial segment
2. **Drift Metric**: Absolute Translational Error (ATE) at final position
3. **Ground Truth**: AprilTag markers at sequence start/end (paper) or LeGO-LOAM pseudo-GT (this project)
4. **Loop Closure**: Disabled for fair odometry comparison

```bash
# Run complete evaluation
python3 scripts/evaluation/evaluate_all.py

# Quick statistics
python3 scripts/evaluation/quick_traj_stats.py results/trajectories/lego_loam/

# Generate comparison plots
python3 scripts/visualization/plot_trajectory_2d.py \
    results/trajectories/lego_loam/5th_floor.txt \
    results/figures/lego_loam/5th_floor.png
```

## 🎨 Figures to Reproduce

### Figure 6: Perceptual Aliasing (Requires Basalt)

Demonstrates how visually similar floors cause incorrect loop closures:
- (a) Trajectory **without** loop closure - floors correctly separated
- (b) Trajectory **with** loop closure - floors incorrectly merged

### Figure 7: 5th Floor Trajectory Comparison

Overlay of all algorithm trajectories highlighting:
- **Region A**: Dynamic content causing visual SLAM artifacts
- **Region B**: Featureless corridor causing tracking failures

## 🛠️ Development

### Running Tests

```bash
# Unit tests
pytest tests/

# Test calibration converter
python3 -m pytest tests/test_calib_converter.py -v
```

### Adding a New Algorithm

1. Create `docker/Dockerfile.{algorithm}`
2. Add configuration to `config/{algorithm}/`
3. Create run script in `scripts/docker/`
4. Add evaluation entry in `scripts/evaluation/evaluate_all.py`
5. Update this README

### Building Individual Images

```bash
# Build specific algorithm
docker build -t slam-benchmark/lego-loam:latest -f docker/Dockerfile.lego-loam .
docker build -t slam-benchmark/orb-slam3:latest -f docker/Dockerfile.orb-slam3 .
```

## 📚 References

### Paper

```bibtex
@inproceedings{kaveti2023challenges,
  title={Challenges of Indoor SLAM: A Multi-Modal Multi-Floor Dataset for SLAM Evaluation},
  author={Kaveti, Pushyami and Gupta, Aniket and Giaya, Dennis and Karp, Madeline and 
          Keil, Colin and Nir, Jagatpreet and Zhang, Zhiyong and Singh, Hanumant},
  booktitle={IEEE International Conference on Automation Science and Engineering (CASE)},
  year={2023}
}
```

### Algorithm Repositories

- [LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM)
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
- [Basalt](https://gitlab.com/VladyslavUsenko/basalt)
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)

### Dataset

- [NUFR-M3F Dataset](https://github.com/neufieldrobotics/NUFR-M3F)

### Evaluation Tools

- [evo - Python package for trajectory evaluation](https://github.com/MichaelGrupp/evo)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Northeastern University Field Robotics Lab for the ISEC dataset
- Authors of the evaluated SLAM algorithms
- The ROS and robotics open-source community

---

**Status**: Active Development | **Last Updated**: November 2025

## 🖥️ Real-Time Visualization

This project includes VNC-based visualization for watching ORB-SLAM3 process data in real-time.

### Building the Visualization Image
```bash
# First, ensure the base ORB-SLAM3 image is built
docker build -t slam-benchmark/orb-slam3:latest -f docker/Dockerfile.orb-slam3 .

# Then build the visualization image
docker build -t slam-benchmark/orb-slam3-viz:latest -f docker/Dockerfile.orb-slam3-viz .
```

### Running Visualization
```bash
# Start the visualization container
docker run --rm -it \
    -p 5900:5900 \
    -v ~/Dev/shared/datasets/ISEC:/data/ISEC:ro \
    -v $(pwd)/results:/results \
    -v $(pwd)/config:/config:ro \
    slam-benchmark/orb-slam3-viz:latest

# In another terminal, connect via VNC
vncviewer localhost:5900
```

### Inside the Container

Once connected via VNC, run ORB-SLAM3 with visualization:
```bash
/root/run_orb_slam3_viewer.sh 5th_floor 0.3 180
```

Arguments:
- `floor`: Dataset floor (5th_floor, 1st_floor, 4th_floor, 2nd_floor)
- `rate`: Playback speed multiplier (0.3 = 30% speed, good for visualization)
- `duration`: Maximum playback duration in seconds

### What You'll See

Two windows appear side-by-side (1800x1350 each):

**Map Viewer (left):**
- Red dots: 3D map points triangulated from stereo
- Green rectangle: Current camera pose
- Blue line: Trajectory path traveled

**Current Frame (right):**
- Live camera feed from left stereo camera
- Green dots: ORB features being tracked
- Status bar: Map count, keyframes, map points, matches

### VNC Desktop Size

The VNC desktop is configured to 3600x1350 pixels to fit both windows side-by-side perfectly.
