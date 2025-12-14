# SLAM Benchmarking Quick Reference
## Run on your local machine at ~/Dev/ros1/slam-benchmark/

## 1. Dataset Verification
```bash
# Check bag files exist
ls -lh ~/Dev/shared/datasets/ISEC/5th_floor/*.bag | head -5

# Check topics in first bag
docker run --rm \
  -v ~/Dev/shared/datasets/ISEC:/data:ro \
  ros:noetic-perception \
  rosbag info /data/5th_floor/$(ls /data/5th_floor/*.bag | head -1 | xargs basename)

# Expected topics:
# /ouster/points (10Hz)
# /camera_array/cam1/image_raw (20Hz)
# /camera_array/cam3/image_raw (20Hz)
# /vectornav/imu (200Hz)
```

## 2. Run LeGO-LOAM (Reference)
```bash
# Create output directory
mkdir -p ~/Dev/ros1/slam-benchmark/results/trajectories

# Run LeGO-LOAM
docker run --rm \
  -v ~/Dev/shared/datasets/ISEC:/data:ro \
  -v ~/Dev/ros1/slam-benchmark/results:/output \
  slam-benchmark/lego-loam:latest \
  roslaunch lego_loam run.launch

# Save trajectory (from inside container or use your run script)
# Output: /output/trajectories/lego_loam_5th.txt
```

## 3. Run ORB-SLAM3 Stereo
```bash
# Ensure calibration has Camera2 params!
docker run --rm \
  -v ~/Dev/shared/datasets/ISEC:/data:ro \
  -v ~/Dev/ros1/slam-benchmark/results:/output \
  -v ~/Dev/ros1/slam-benchmark/config:/config:ro \
  slam-benchmark/orb-slam3:latest \
  roslaunch orb_slam3 stereo.launch \
    vocabulary:=/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    config:=/config/isec_stereo_orbslam3.yaml
```

## 4. Run DROID-SLAM
```bash
docker run --rm --gpus all \
  -v ~/Dev/shared/datasets/ISEC:/data:ro \
  -v ~/Dev/ros1/slam-benchmark/results:/output \
  slam-benchmark/droid-slam:latest \
  python demo.py \
    --imagedir /data/5th_floor/images \
    --calib /config/droid_calib.txt \
    --stride 1 \
    --output /output/trajectories/droid_slam_5th.txt
```

## 5. Evaluate Trajectories
```bash
# Using evo (install: pip install evo)
evo_ape tum \
  results/trajectories/lego_loam_5th.txt \
  results/trajectories/orb_slam3_5th.txt \
  --align --correct_scale --plot

# Compare all algorithms
python evaluate_trajectories.py \
  --compare-all results/trajectories/ \
  --output-dir results/figures/
```

## 6. Expected Results (Table IV, 5th Floor)
| Algorithm   | ATE (m) | ATE %  |
|-------------|---------|--------|
| LeGO-LOAM   | 0.395   | 0.21%  |
| ORB-SLAM3   | 0.516   | 0.28%  |
| DROID-SLAM  | 0.441*  | 0.24%  |
| VINS-Fusion | 1.120   | 0.60%  |

*Requires Sim(3) alignment

## 7. Common Issues & Fixes

### ORB-SLAM3 tracking lost
- Check Camera2 parameters exist in YAML
- Verify baseline is ~0.33m not ~0.16m
- Lower ORBextractor.iniThFAST if featureless areas

### DROID-SLAM scale wrong
- Use Sim(3) alignment: `--correct_scale` in evo
- Scale factor should be ~1.0 if stereo calibration correct

### Timestamps mismatch
- Use `--t_max_diff 0.1` in evo_ape
- Or resample to common frequency

### LeGO-LOAM channel error
- Ensure N_SCAN=128 in utility.h
- Rebuild Docker image after changes

## 8. Docker Image Status
```bash
docker images | grep slam-benchmark
# Expected:
# slam-benchmark/lego-loam   latest
# slam-benchmark/orb-slam3   latest  
# slam-benchmark/droid-slam  latest
```
