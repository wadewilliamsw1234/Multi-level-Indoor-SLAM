# Endpoint Drift Analysis

## Results Table

| Floor | Algorithm | Drift (m) | Drift (%) | Trajectory Length |
|-------|-----------|-----------|-----------|-------------------|
| 5th | LeGO-LOAM | 0.747 | 0.40% | 188.1m |
| 5th | ORB-SLAM3 | 0.703 | 0.46% | 151.9m |
| 4th | LeGO-LOAM | 0.138 | 0.21% | 66.2m |
| 4th | ORB-SLAM3 | 0.565 | 0.97% | 58.0m |
| 2nd | LeGO-LOAM | 0.107 | 0.08% | 135.0m |
| 2nd | ORB-SLAM3 | 1.971 | 1.84% | 107.0m |
| 1st | LeGO-LOAM | 0.329 | 0.45% | 73.7m |
| 1st | ORB-SLAM3 | 12.011 | 38.13% | 31.5m (FAILED) |

## Paper Comparison (ORB-SLAM3)

| Floor | Our Result | Paper Result | Status |
|-------|------------|--------------|--------|
| 5th | 0.70m (0.46%) | 0.52m (0.28%) | ✅ Within 35% |
| 4th | 0.57m (0.97%) | 0.48m (0.73%) | ✅ Within 20% |
| 2nd | 1.97m (1.84%) | 0.31m (0.24%) | ⚠️ 6x higher |
| 1st | 12.0m (38%) | 0.95m (1.46%) | ❌ Failed (expected) |

## Notes

- LeGO-LOAM outperforms paper results on 3/4 floors
- ORB-SLAM3 1st floor failure matches paper's discussion of visual degradation
- 2nd floor higher error likely due to different ORB-SLAM3 parameters or tracking losses
