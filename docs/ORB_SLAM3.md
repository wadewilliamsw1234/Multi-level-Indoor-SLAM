# ORB-SLAM3 for ISEC Dataset

This directory contains the Dockerfile and configuration for running ORB-SLAM3 
on the ISEC multi-floor indoor SLAM dataset.

## Quick Start

### 1. Build the Docker Image

```bash
cd ~/Dev/ros1/slam-benchmark

# Build ORB-SLAM3 image (takes ~15-20 minutes)
docker build -t slam-benchmark/orb-slam3:latest -f docker/Dockerfile.orb-slam3 .
```

### 2. Run on a Single Floor

```bash
# Run on 5th floor (largest sequence, ~7 minutes of data)
docker run --rm -it \
    -v ~/Dev/shared/datasets/ISEC:/data/ISEC:ro \
    -v ~/Dev/ros1/slam-benchmark/results:/results \
    -v ~/Dev/ros1/slam-benchmark/config:/config:ro \
    --network=host \
    slam-benchmark/orb-slam3:latest \
    /root/run_orb_slam3_floor.sh 5th_floor
```

### 3. Run on All Floors

```bash
docker run --rm -it \
    -v ~/Dev/shared/datasets/ISEC:/data/ISEC:ro \
    -v ~/Dev/ros1/slam-benchmark/results:/results \
    -v ~/Dev/ros1/slam-benchmark/config:/config:ro \
    --network=host \
    slam-benchmark/orb-slam3:latest \
    /root/run_orb_slam3_all.sh
```

### 4. Interactive Mode (for debugging)

```bash
docker run --rm -it \
    -v ~/Dev/shared/datasets/ISEC:/data/ISEC:ro \
    -v ~/Dev/ros1/slam-benchmark/results:/results \
    -v ~/Dev/ros1/slam-benchmark/config:/config:ro \
    --network=host \
    slam-benchmark/orb-slam3:latest \
    bash
```

Then inside the container:
```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
/root/run_orb_slam3_floor.sh 5th_floor
```

## Configuration

### Camera Setup
- **Stereo pair**: cam1 (left) + cam3 (right)
- **Baseline**: 0.3284m
- **Resolution**: 720×540 @ 20Hz
- **Model**: pinhole + radtan distortion

### Key Parameters (ISEC_stereo.yaml)
```yaml
Camera1.fx: 893.63
Camera1.fy: 893.97
Camera1.cx: 376.95
Camera1.cy: 266.57

Stereo.bf: 293.5  # baseline × fx
Stereo.ThDepth: 40.0

ORBextractor.nFeatures: 1500
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
```

### Loop Closure
Loop closure is **disabled** by default for fair comparison with the paper.
To enable, modify `ISEC_stereo.yaml`:
```yaml
LoopClosing.DetectLoopClosures: 1
```

## Expected Results (vs Paper Table IV)

| Floor | Length | Paper ATE(m) | Paper % |
|-------|--------|--------------|---------|
| 5th | 187m | 0.516m | 0.28% |
| 1st | 65m | 0.949m | 1.46% |
| 4th | 66m | 0.483m | 0.73% |
| 2nd | 128m | 0.310m | 0.24% |

## Output Files

Trajectories are saved in TUM format:
```
results/trajectories/orb_slam3/
├── 5th_floor.txt
├── 1st_floor.txt
├── 4th_floor.txt
└── 2nd_floor.txt
```

TUM format: `timestamp tx ty tz qx qy qz qw`

## Troubleshooting

### Build Fails on OpenCV
The Dockerfile patches CMakeLists.txt for OpenCV 4.x compatibility. If issues persist:
```bash
docker run --rm -it slam-benchmark/orb-slam3:latest bash
# Check OpenCV version
pkg-config --modversion opencv4
```

### Vocabulary Loading Slow
The ORB vocabulary (ORBvoc.txt) is ~1GB and takes 20-30 seconds to load.
This is normal - the script waits for initialization.

### Tracking Failures
ORB-SLAM3 may fail in:
- Featureless corridors (white walls)
- Glass surfaces with reflections
- Fast motion / motion blur

The paper notes these as challenging scenarios.

### Memory Issues
If running out of memory on large sequences:
- Process one floor at a time
- Reduce `ORBextractor.nFeatures` to 1000
- Use playback rate `-r 0.25` instead of `-r 0.5`

## Evaluation

To evaluate against LeGO-LOAM pseudo ground truth:
```bash
docker run --rm \
    -v ~/Dev/ros1/slam-benchmark/results:/results \
    python:3.10 \
    python3 /results/../scripts/evaluate_trajectories.py
```

Or install evo locally:
```bash
pip install evo
python3 scripts/evaluate_trajectories.py
```

## Files

```
docker/
└── Dockerfile.orb-slam3      # Docker build file

config/orb_slam3/
└── ISEC_stereo.yaml          # Camera calibration & ORB parameters

scripts/
├── run_orb_slam3_floor.sh    # Run on single floor
├── run_orb_slam3_all.sh      # Run on all floors
└── evaluate_trajectories.py  # Compute ATE metrics
```

## References

- ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
- Paper: "ORB-SLAM3: An Accurate Open-Source Library for Visual, 
  Visual-Inertial and Multi-Map SLAM" (Campos et al., IEEE T-RO 2021)
- ISEC Dataset Paper: "Challenges of Indoor SLAM: A Multi-Modal Multi-Floor 
  Dataset for SLAM Evaluation" (Kaveti et al., IEEE CASE 2023)
