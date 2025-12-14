# Archived Files

This directory contains files that are no longer part of the active codebase but are preserved for reference and to document our iteration process.

## Directory Structure

### `superseded/`
Files replaced by better implementations:
- **evaluation scripts**: Earlier versions of trajectory evaluation (replaced by `comprehensive_evaluation.py`)
- **config backups**: Old configuration files before parameter tuning
- **extraction scripts**: Duplicate/older versions of data extraction utilities

### `experimental/`
Features we tried but didn't produce usable results:
- **VINS-Fusion configs**: Multiple IMU calibration attempts that failed due to coordinate frame issues
- **Debug scripts**: One-off debugging utilities for troubleshooting

### `debug/`
Debug sessions and temporary outputs:
- **debug_session_***: Timestamped debug session artifacts
- **debug scripts**: Shell scripts used for troubleshooting

### `intermediate/`
Intermediate outputs that are no longer needed:
- **trajectory variants**: Scaled, corrected, and timestamped versions (canonical versions kept in `results/trajectories/`)
- **VINS trajectories**: Failed algorithm outputs preserved for documentation

## Why Archive Instead of Delete?

1. **Academic integrity**: Shows our iteration process and failed experiments
2. **Reproducibility**: Someone may want to understand why certain approaches didn't work
3. **Reference**: Contains useful code snippets that may be reused

## Active Codebase

The active codebase is in the parent directory. Key entry points:
- `scripts/run_all.sh` - Reproduce all results
- `scripts/semantic_gating/` - Our Level 3 contribution (floor-based loop closure gating)
- `scripts/evaluation/comprehensive_evaluation.py` - Main evaluation script

---
*Archived: December 2024*
*Course: EECE5554 Robotic Sensing and Navigation*
