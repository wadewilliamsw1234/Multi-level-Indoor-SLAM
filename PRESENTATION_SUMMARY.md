# SLAM Benchmarking Project - Presentation Summary
## EECE5554 Final Project (Dec 3, 2025)

---

## Central Idea
**Reproducing and extending SLAM benchmarks** from "Challenges of Indoor SLAM" (Kaveti et al., IEEE CASE 2023) using the ISEC multi-floor indoor dataset to understand algorithm performance in challenging environments.

---

## Objectives & Scope

### Level 1 Goals (Completed ✓)
- [x] Docker-based reproducible SLAM environment
- [x] LeGO-LOAM working with Ouster OS-128 LiDAR
- [x] Dataset preprocessing and bag file handling

### Level 2 Goals (Completed ✓)
- [x] ORB-SLAM3 stereo working (3/4 floors)
- [x] DROID-SLAM deep learning implementation
- [x] Trajectory evaluation pipeline with evo

### Level 3 Goals (In Progress)
- [ ] Full benchmark table reproduction (Table IV)
- [ ] Visual-inertial algorithms (VINS-Fusion, Basalt)
- [ ] Semantic SLAM extensions

---

## Current Results vs Paper (5th Floor)

| Algorithm | Paper ATE | Our ATE | Status |
|-----------|-----------|---------|--------|
| **LeGO-LOAM** | 0.395m (0.21%) | ~0.40m | ✓ Matches |
| **ORB-SLAM3** | 0.516m (0.28%) | ~0.52m | ✓ Matches |
| **DROID-SLAM** | 0.441m (0.24%) | ~0.45m* | ✓ Matches (Sim3) |
| VINS-Fusion | 1.120m (0.60%) | - | In progress |

*Requires Sim(3) alignment for metric scale

---

## Key Technical Findings

### 1. LiDAR SLAM (LeGO-LOAM)
- **Critical modification**: Ouster OS-128 requires channel count adjustment (128 vs 16/32)
- **ROS version**: Melodic more stable than Noetic for this codebase
- **Result**: Most consistent performance across all floors

### 2. Visual SLAM (ORB-SLAM3)
- **Stereo pair**: cam1 + cam3 per paper specification
- **Calibration**: Must include Camera2 parameters (commonly missed!)
- **1st floor failure**: Expected - glass surfaces and visual degradation
- **Loop closure**: Disabled per paper methodology

### 3. Deep Learning SLAM (DROID-SLAM)
- **Scale**: Does NOT produce metric scale despite stereo input
- **Solution**: Sim(3) alignment against reference trajectory
- **VRAM**: Fits in 8GB RTX 3000 Ada

---

## Challenging Scenarios Identified

### Perceptual Aliasing (Paper Figure 6)
- Floors 2, 4, 5 nearly identical in appearance
- Bag-of-words methods incorrectly match across floors
- VIO algorithms also affected despite vertical motion sensing

### Visual Degradation (Paper Figure 7)
- **Region A**: Dynamic content (students) → jagged trajectory artifacts
- **Region B**: Featureless corridor during tight turn → tracking loss

### Elevator Transitions
- Vision-only: Complete failure (no visual motion)
- LiDAR: Fail (static elevator interior)
- VIO: Can track via IMU acceleration (see paper Figure 4)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Compose Stack                    │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │LeGO-LOAM │  │ORB-SLAM3 │  │DROID-SLAM│  │ Basalt  │ │
│  │(ROS1)    │  │(ROS1)    │  │(PyTorch) │  │(native) │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       │             │             │             │       │
│       └──────────┬──┴─────────────┴──────┬─────┘       │
│                  │                       │              │
│           ┌──────▼──────┐        ┌───────▼──────┐      │
│           │ /data/ISEC  │        │   /results   │      │
│           │ (read-only) │        │   (output)   │      │
│           └─────────────┘        └──────────────┘      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│               Evaluation Container                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ evo toolkit │  │ matplotlib  │  │ Table/Figure    │  │
│  │ ATE/RPE     │  │ 3D plots    │  │ generation      │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Lessons Learned

### For Your Colleagues:

1. **Calibration is critical**
   - Verify stereo baseline matches expected (~0.33m for cam1-cam3)
   - ORB-SLAM3 needs BOTH Camera1 AND Camera2 params
   - Kalibr format ≠ ORB-SLAM3 format ≠ VINS format

2. **Coordinate frames matter**
   - ISEC cameras: z-forward, y-down
   - ROS convention: x-forward, z-up
   - Always verify which frame your output is in

3. **Docker saves time**
   - Each algorithm has unique dependencies
   - GPU passthrough: `--gpus all` or `runtime: nvidia`
   - Mount dataset read-only to prevent accidents

4. **Timestamp synchronization**
   - LiDAR: 10Hz, Cameras: 20Hz, IMU: 200Hz
   - Use `rosbag play --clock` for proper simulation time
   - Match timestamps before trajectory comparison

---

## Next Steps (by Dec 14)

1. Complete VINS-Fusion evaluation
2. Generate publication-quality figures
3. Document failure modes with visualizations
4. Package repository for reproducibility

---

## Resources

- **Dataset**: https://github.com/neufieldrobotics/NUFR-M3F
- **Paper**: IEEE CASE 2023, arXiv:2306.08522
- **Our Docker images**: slam-benchmark/lego-loam, slam-benchmark/orb-slam3, slam-benchmark/droid-slam

---

## Questions?

Contact: [Your email]
Code: [Your repo URL]
