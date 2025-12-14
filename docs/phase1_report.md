# Phase 1: Project Review and Health Check Report

**Date**: December 3, 2025  
**Project**: SLAM Benchmark Study (ISEC Dataset)

---

## Executive Summary

Phase 1 comprehensive audit identified the **root cause of VINS-Fusion's massive drift**: a **frame convention mismatch** between the VectorNav VN-100 IMU (Z-down/NED-like) and VINS-Fusion's expected Z-up (FLU) convention. Multiple correction attempts were made, but a working solution has not yet been achieved. Additional debugging is recommended.

---

## 1. IMU Frame Convention Analysis

### Key Finding: VectorNav Uses Z-DOWN Convention

Diagnostic script (`scripts/diagnostic/dump_imu_samples.py`) confirmed:

| IMU Topic | Mean Accel Z | Convention |
|-----------|-------------|------------|
| `/imu/imu_compensated` | -9.87 m/s² | **Z-DOWN (NED)** |
| `/imu/imu_uncompensated` | -9.87 m/s² | **Z-DOWN (NED)** |
| `/ouster/imu` | +9.73 m/s² | **Z-UP (FLU)** ✅ |

The **Ouster LiDAR's built-in IMU** is already in FLU convention and could be an alternative data source.

### Evidence
```
============================================================
Frame Convention Analysis
============================================================

Dominant gravity axis: Z
Gravity sign in Z: NEGATIVE

⚠️  CONCLUSION: IMU uses Z-DOWN convention (NED-like)
   Frame transformation required for VINS-Fusion!
```

---

## 2. Correction Attempts & Results

### IMU Transformer Verification ✅
The `imu_ned_to_flu.py` node correctly transforms IMU data:
```
Original: ax=0.0098, ay=0.0536, az=-9.8577 (NED, gravity negative)
FLU:      ax=0.0098, ay=-0.0536, az=+9.8577 (FLU, gravity positive) ✅
```

### Test Results Summary

| Config | IMU Topic | Extrinsics | Noise | 30s Test | Full Test |
|--------|-----------|------------|-------|----------|-----------|
| Original | `/imu/imu_uncompensated` | Original | Calibrated | - | ~3340m drift |
| Ouster native FLU | `/ouster/imu` | Original | Higher | 28.79m ✅ | 15,439m ❌ |
| VectorNav FLU | `/imu/imu_flu` | Original | 100x larger | 47.19m ✅ | **675m** |
| VectorNav FLU | `/imu/imu_flu` | FLU-corrected | Calibrated | - | 8,524m ❌ |

### Key Insight: User's Guidance Was Correct
The user advised: **transform IMU data + keep original extrinsics**. This produces the best results:
- **VectorNav FLU + original extrinsics + 100x noise**: 675m vs 187m expected (3.6x error)
- This is **dramatically better** than the 50x errors from other attempts

### Remaining Issue: Long-term Drift
- Short tests (30s) look reasonable for both Ouster and VectorNav FLU
- Long tests accumulate drift without loop closure
- The 3.6x scale error on VectorNav suggests IMU noise parameters need further tuning

---

## 3. Docker Architecture Status

| Algorithm | Base Image | Status | Notes |
|-----------|-----------|--------|-------|
| LeGO-LOAM | ros:melodic-perception | ✅ Working | Pseudo ground truth |
| ORB-SLAM3 | ros:noetic-perception | ✅ Working | Fails on 1st floor |
| VINS-Fusion | ros:noetic-perception | ❌ Frame Issue | NED→FLU needed |
| DROID-SLAM | nvidia/cuda:11.8 | ⚠️ Untested | CUDA compatibility? |
| Basalt | ubuntu:22.04 | ⚠️ Pending | Needs EuRoC format |

---

## 4. Configuration Files Status

### Working Calibration Files
- ✅ `config/vins_fusion/cam0.yaml` - cam1 intrinsics (fx=893.6, image 720x540)
- ✅ `config/vins_fusion/cam1.yaml` - cam3 intrinsics (fx=890.4, image 720x540)
- ✅ Stereo baseline verified: **0.328m** (from extrinsics)

### Extrinsics
| File | body_T_cam0 | body_T_cam1 | Frame |
|------|-------------|-------------|-------|
| `isec_stereo_imu_config.yaml` | Original | Original | NED |
| `isec_stereo_imu_flu.yaml` | Transformed | Transformed | FLU |

### IMU Parameters (from ISEC calibration)
```yaml
acc_n: 0.0014126598501078217   # Accelerometer noise density
gyr_n: 7.77970988215584e-05    # Gyroscope noise density
acc_w: 1.9005701759499173e-05  # Accelerometer random walk
gyr_w: 3.3171207235534e-07     # Gyroscope random walk
```

---

## 5. Scripts and Tools Created

### Diagnostic Tools
1. **`scripts/diagnostic/dump_imu_samples.py`** - Analyzes IMU frame convention
2. **`scripts/diagnostic/run_imu_diagnostic.sh`** - Docker wrapper for IMU analysis

### Frame Correction Tools
3. **`scripts/vins_fusion/imu_ned_to_flu.py`** - ROS node to transform IMU data
4. **`scripts/utils/apply_frame_correction.py`** - Computes FLU-corrected extrinsics

### Test Scripts
5. **`scripts/run_vins_flu_test.sh`** - Test with FLU extrinsics only
6. **`scripts/run_vins_full_flu_test.sh`** - Full FLU correction test

---

## 6. Recommended Next Steps

### ✅ Working Config (Best So Far)
The best configuration is **VectorNav FLU + original extrinsics + 100x noise**:
- Config: `isec_flu_high_noise.yaml`
- IMU transformer: `imu_ned_to_flu.py`
- Result: 3.6x scale error (675m vs 187m expected)

### Option A: Further IMU Noise Tuning
Current 100x increase helps, but more tuning needed:
```yaml
# Try 200-500x larger for reduced IMU trust
acc_n: 0.28 - 0.7      # Currently 0.14
gyr_n: 0.015 - 0.039   # Currently 0.0078
```

### Option B: Enable Loop Closure
Would help reduce long-term drift. Requires:
1. VINS-Fusion vocabulary file for DBoW2
2. `loop_closure: 1` in config

### Option C: Ground Truth Comparison
Instead of judging absolute trajectory length:
1. Use LeGO-LOAM trajectory as pseudo ground truth
2. Evaluate VINS against it using ATE/RPE metrics
3. The 3.6x scale might be acceptable for relative evaluation

### Option D: Proceed to Other Algorithms
VINS-Fusion debugging has provided valuable insights:
- Frame conventions are critical
- IMU noise parameters significantly affect results
- Move to testing Basalt, DROID-SLAM, ORB-SLAM3

---

## 7. Files Changed/Created

```
config/vins_fusion/
├── isec_stereo_imu_flu.yaml          # FLU extrinsics + FLU IMU
├── isec_stereo_imu_flu_v2.yaml       # Original extrinsics + FLU IMU
├── isec_stereo_imu_estimate.yaml     # Enable auto-calibration

scripts/
├── diagnostic/
│   ├── dump_imu_samples.py           # IMU frame diagnostic
│   └── run_imu_diagnostic.sh         # Docker wrapper
├── vins_fusion/
│   └── imu_ned_to_flu.py             # IMU frame transformer
├── utils/
│   └── apply_frame_correction.py     # Extrinsic transformer
└── run_vins_*.sh                     # Test scripts

docs/
└── phase1_report.md                  # This report
```

---

## 8. Key Insights for Phase 2

1. **Frame conventions matter critically** for VIO - get this right first
2. **Ouster IMU** is a viable alternative if VectorNav integration proves difficult
3. **Basalt and DROID-SLAM** may be more robust to calibration issues
4. Consider **extrinsic estimation** during initial runs

---

## Appendix: IMU Transformation Mathematics

### NED to FLU Transformation
```python
R_flu_ned = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

# For IMU data:
accel_flu = R_flu_ned @ accel_ned
gyro_flu = R_flu_ned @ gyro_ned

# For extrinsics T = [R|t]:
R_flu = R_flu_ned @ R_ned
t_flu = R_flu_ned @ t_ned
```

### Verification
- Stereo baseline preserved: 0.328m ✅
- Camera optical axis still points forward ✅
- Frame convention consistent ❌ (still debugging)

