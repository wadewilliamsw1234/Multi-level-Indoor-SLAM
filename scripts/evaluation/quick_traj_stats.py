#!/usr/bin/env python3
import numpy as np
import sys
from pathlib import Path

def analyze_trajectory(traj_file):
    """Compute basic statistics for a TUM format trajectory."""
    try:
        data = np.loadtxt(traj_file)
        if data.shape[1] < 4:
            print(f"ERROR: Invalid format in {traj_file}")
            return None
        
        timestamps = data[:, 0]
        positions = data[:, 1:4]
        
        # Compute trajectory length (sum of segment distances)
        diffs = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(segment_lengths)
        
        # Time span
        duration = timestamps[-1] - timestamps[0]
        
        # Start and end positions
        start_pos = positions[0]
        end_pos = positions[-1]
        drift = np.linalg.norm(end_pos - start_pos)
        
        # Position bounds
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        extent = max_pos - min_pos
        
        return {
            'num_poses': len(timestamps),
            'duration': duration,
            'length': total_length,
            'drift': drift,
            'drift_pct': (drift / total_length * 100) if total_length > 0 else 0,
            'extent': extent,
            'start': start_pos,
            'end': end_pos
        }
    except Exception as e:
        print(f"ERROR reading {traj_file}: {e}")
        return None

if __name__ == "__main__":
    traj_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    
    # Expected from paper Table IV
    paper_results = {
        '5th_floor': {'length': 187, 'ate': 0.516, 'ate_pct': 0.28},
        '1st_floor': {'length': 65, 'ate': 0.949, 'ate_pct': 1.46},
        '4th_floor': {'length': 66, 'ate': 0.483, 'ate_pct': 0.73},
        '2nd_floor': {'length': 128, 'ate': 0.310, 'ate_pct': 0.24}
    }
    
    print("="*80)
    print("ORB-SLAM3 Trajectory Analysis")
    print("="*80)
    
    for floor in ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']:
        traj_file = traj_dir / f"{floor}.txt"
        
        print(f"\n{floor.upper()}:")
        print("-" * 60)
        
        if not traj_file.exists():
            print(f"  ❌ FILE NOT FOUND: {traj_file}")
            continue
        
        stats = analyze_trajectory(traj_file)
        if stats is None:
            continue
        
        print(f"  Poses:          {stats['num_poses']:,}")
        print(f"  Duration:       {stats['duration']:.1f} seconds")
        print(f"  Traj Length:    {stats['length']:.1f} m")
        print(f"  Start->End:     {stats['drift']:.2f} m")
        print(f"  Drift %:        {stats['drift_pct']:.2f}%")
        print(f"  Extent (XYZ):   [{stats['extent'][0]:.1f}, {stats['extent'][1]:.1f}, {stats['extent'][2]:.1f}] m")
        
        # Compare with paper
        if floor in paper_results:
            paper = paper_results[floor]
            print(f"\n  📊 PAPER COMPARISON:")
            print(f"     Paper length:    {paper['length']} m")
            print(f"     Our length:      {stats['length']:.1f} m  (diff: {abs(stats['length'] - paper['length']):.1f} m)")
            print(f"     Paper ATE:       {paper['ate']:.3f} m ({paper['ate_pct']:.2f}%)")
            print(f"     Our drift:       {stats['drift']:.3f} m ({stats['drift_pct']:.2f}%)")
            
            # Rough assessment
            length_match = abs(stats['length'] - paper['length']) < paper['length'] * 0.2
            if length_match:
                print(f"     ✅ Length matches paper (within 20%)")
            else:
                print(f"     ⚠️  Length differs significantly from paper")

