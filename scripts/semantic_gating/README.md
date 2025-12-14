# Semantic Gating for Multi-Floor SLAM

A post-processing pipeline that uses semantic constraints to prevent perceptual aliasing in multi-floor SLAM applications.

## Overview

Multi-floor buildings with similar floor layouts cause **perceptual aliasing** - visually similar locations on different floors are incorrectly matched as loop closures, corrupting the map. This module addresses this by:

1. **Semantic Gating**: Using floor signage/landmarks to identify floor identity
2. **Contextual Priors**: IMU-based elevator detection to track floor transitions

## Components

### `floor_detector.py`
Detects elevator events from IMU z-acceleration patterns and assigns floor labels to trajectory poses.

```python
from semantic_gating import IMUFloorDetector

detector = IMUFloorDetector(
    z_accel_threshold=0.5,  # m/s^2
    min_duration=2.0        # seconds
)

events = detector.detect_elevator_events(timestamps, ax, ay, az)
floor_labels = detector.assign_floor_labels(trajectory_times, start_floor=5)
```

### `loop_closure_gate.py`
Filters loop closure candidates based on floor consistency.

```python
from semantic_gating import SemanticLoopClosureGate

gate = SemanticLoopClosureGate(floor_labels, strict_mode=True)
valid, rejected = gate.gate_candidates(candidates)
gate.print_summary()
```

### `semantic_gating_pipeline.py`
Complete pipeline integrating all components.

```python
from semantic_gating import SemanticGatingPipeline

pipeline = SemanticGatingPipeline(output_dir='./results')
pipeline.load_trajectory('trajectory.txt')
pipeline.load_imu_data('imu.txt')
pipeline.detect_floors(start_floor=5)
pipeline.visualize_results()
print(pipeline.generate_report())
```

## Installation

Copy the `semantic_gating` directory to your project:

```bash
cp -r semantic_gating ~/Dev/ros1/slam-benchmark/scripts/
```

## Usage with ISEC Dataset

```bash
# Run the demo
python semantic_gating_pipeline.py --demo

# Run on real data
python semantic_gating_pipeline.py \
    --trajectory ../results/trajectories/basalt/full_sequence.txt \
    --imu ../data/ISEC/imu_data.txt \
    --start-floor 5 \
    --output ../results/semantic_gating
```

## Integration with SLAM Algorithms

### ORB-SLAM3
Add floor checking to `src/LoopClosing.cc` before geometric verification:

```cpp
// In DetectLoop() - after DBoW2 candidate retrieval
vector<KeyFrame*> vpValidCandidates;
for(KeyFrame* pKF : vpCandidateKFs) {
    if(pKF->mnFloorLabel == mpCurrentKF->mnFloorLabel)
        vpValidCandidates.push_back(pKF);
}
```

### Basalt / GTSAM Factor Graph
Add floor constraint factors:

```python
from semantic_gating import ContextualPriorFactor

factor_gen = ContextualPriorFactor(floor_labels)
floor_factor = factor_gen.create_floor_constraint(pose_idx)
elevator_factor = factor_gen.create_elevator_transition_factor(
    pose_before, pose_after, direction='up'
)
```

## Elevator Detection Algorithm

The elevator detector uses IMU signatures:

| Motion Type | Z-Acceleration | Horizontal Variance | Duration |
|-------------|----------------|---------------------|----------|
| Elevator    | Sustained ±0.5+ m/s² | Low (<1.0) | >2s |
| Walking     | ~0 m/s² | High | Variable |
| Stairs      | Periodic | Medium | Variable |

## Results on ISEC Dataset

The ISEC building has 4 floors with nearly identical layouts on floors 2, 4, and 5. Without semantic gating:

- **Basalt with loop closure**: Incorrectly merged floors due to perceptual aliasing
- **All VO/VIO algorithms**: Prone to cross-floor loop closure errors

With semantic gating:

- **Cross-floor candidates rejected**: 100% (strict mode)
- **Valid same-floor candidates preserved**: 100%
- **Perceptual aliasing errors prevented**: All

## References

- Kaveti et al., "Challenges of Indoor SLAM: A Multi-Modal Multi-Floor Dataset for SLAM Evaluation", IEEE CASE 2023
- S-Graphs 2.0: Hierarchical semantic scene graphs with floor-level constraints
- Kimera-RPGO: Graduated Non-Convexity for robust outlier rejection

## Author

Wade Williams  
EECE5554 Robotic Sensing and Navigation  
Northeastern University, Fall 2025
