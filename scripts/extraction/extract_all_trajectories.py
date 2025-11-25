#!/usr/bin/env python3
"""
Extract trajectories from all LeGO-LOAM bags and compute statistics.
Compares results with paper Table IV.

Usage: python3 extract_all_trajectories.py
"""

import subprocess
import sys
from pathlib import Path
import math

RESULTS_DIR = Path("/home/wadewilliams/Dev/ros1/slam-benchmark/results")
TRAJ_DIR = RESULTS_DIR / "trajectories" / "lego_loam"

# Paper values for comparison (Table IV)
PAPER_VALUES = {
    "5th_floor": {"length_m": 187, "ate_m": 0.395, "ate_pct": 0.21},
    "1st_floor": {"length_m": 65, "ate_m": 0.256, "ate_pct": 0.39},
    "4th_floor": {"length_m": 66, "ate_m": 0.789, "ate_pct": 1.20},
    "2nd_floor": {"length_m": 128, "ate_m": 0.286, "ate_pct": 0.22},
}


def analyze_trajectory(traj_path):
    """Analyze existing trajectory file."""
    poses = []
    with open(traj_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                poses.append([float(p) for p in parts[:4]])
    
    if not poses:
        return None
    
    # Calculate length
    length = 0
    for i in range(1, len(poses)):
        dx = poses[i][1] - poses[i-1][1]
        dy = poses[i][2] - poses[i-1][2]
        dz = poses[i][3] - poses[i-1][3]
        length += math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Calculate drift (start to end distance)
    dx = poses[-1][1] - poses[0][1]
    dy = poses[-1][2] - poses[0][2]
    dz = poses[-1][3] - poses[0][3]
    drift = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Duration
    duration = poses[-1][0] - poses[0][0]
    
    return {
        "poses": len(poses),
        "length_m": length,
        "drift_m": drift,
        "drift_pct": (drift / length * 100) if length > 0 else 0,
        "duration_s": duration,
    }


def main():
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 65)
    print("LeGO-LOAM Trajectory Analysis - Comparison with Paper")
    print("=" * 65)
    
    floors = ["5th_floor", "1st_floor", "4th_floor", "2nd_floor"]
    results = {}
    
    for floor in floors:
        print(f"\n--- {floor} ---")
        
        traj_path = TRAJ_DIR / f"{floor}.txt"
        
        if not traj_path.exists():
            print(f"  NOT FOUND: {traj_path}")
            print(f"  Run: python3 scripts/extract_trajectory.py (for this floor's bag)")
            continue
        
        stats = analyze_trajectory(traj_path)
        if stats:
            results[floor] = stats
            paper = PAPER_VALUES.get(floor, {})
            
            print(f"  Poses:    {stats['poses']}")
            print(f"  Length:   {stats['length_m']:.1f}m (paper: {paper.get('length_m', '?')}m)")
            print(f"  Our ATE:  {stats['drift_m']:.3f}m / {stats['drift_pct']:.2f}%")
            print(f"  Paper:    {paper.get('ate_m', '?')}m / {paper.get('ate_pct', '?')}%")
            print(f"  Duration: {stats['duration_s']:.1f}s")
    
    # Summary table
    if results:
        print("\n" + "=" * 65)
        print("SUMMARY: LeGO-LOAM Results vs Paper Table IV")
        print("=" * 65)
        print(f"{'Sequence':<12} {'Length':<14} {'Our ATE':<18} {'Paper ATE':<15}")
        print("-" * 65)
        
        for floor in floors:
            if floor in results:
                r = results[floor]
                p = PAPER_VALUES.get(floor, {})
                length_match = "✓" if abs(r['length_m'] - p.get('length_m', 0)) < 10 else "?"
                print(f"{floor:<12} {r['length_m']:>5.0f}m {length_match} ({p.get('length_m', '?'):>3}m)  "
                      f"{r['drift_m']:.3f}m ({r['drift_pct']:.2f}%)   "
                      f"{p.get('ate_m', '?')}m ({p.get('ate_pct', '?')}%)")
        
        print("-" * 65)
        print("Note: Our ATE may differ due to parameter tuning & hardware differences")
    
    print(f"\nTrajectory files: {TRAJ_DIR}")


if __name__ == "__main__":
    main()
