#!/usr/bin/env python3
"""
SLAM Benchmarking Evaluation Script
Computes drift metrics and compares to paper Table IV values.
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Paper Table IV reference values
PAPER_RESULTS = {
    'orb_slam3': {
        '5th_floor': {'ate_m': 0.516, 'ate_pct': 0.28, 'length': 187},
        '1st_floor': {'ate_m': 0.949, 'ate_pct': 1.46, 'length': 65},
        '4th_floor': {'ate_m': 0.483, 'ate_pct': 0.73, 'length': 66},
        '2nd_floor': {'ate_m': 0.310, 'ate_pct': 0.24, 'length': 128},
    },
    'lego_loam': {
        '5th_floor': {'ate_m': 0.395, 'ate_pct': 0.21, 'length': 187},
        '1st_floor': {'ate_m': 0.256, 'ate_pct': 0.39, 'length': 65},
        '4th_floor': {'ate_m': 0.789, 'ate_pct': 1.20, 'length': 66},
        '2nd_floor': {'ate_m': 0.286, 'ate_pct': 0.22, 'length': 128},
    },
}

def load_tum_trajectory(filepath):
    """Load TUM format trajectory file."""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1:4], data[:, 4:8]

def compute_trajectory_length(positions):
    """Compute total trajectory length."""
    diffs = np.diff(positions, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

def compute_drift(positions):
    """Compute drift = distance from start to end."""
    return np.linalg.norm(positions[-1] - positions[0])

def analyze_trajectory(traj_file):
    """Compute all metrics for a trajectory."""
    timestamps, positions, _ = load_tum_trajectory(traj_file)
    length = compute_trajectory_length(positions)
    drift = compute_drift(positions)
    drift_pct = (drift / length * 100) if length > 0 else 0
    
    return {
        'num_poses': len(timestamps),
        'duration_s': float(timestamps[-1] - timestamps[0]),
        'length_m': float(length),
        'drift_m': float(drift),
        'drift_pct': float(drift_pct),
    }

def main():
    results_dir = Path(__file__).parent.parent.parent / 'results'
    traj_dir = results_dir / 'trajectories'
    
    algorithms = [d.name for d in traj_dir.iterdir() if d.is_dir()]
    floors = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']
    
    print("\n" + "="*100)
    print("SLAM BENCHMARKING RESULTS - Comparison with Kaveti et al. (IEEE CASE 2023)")
    print("="*100)
    
    all_results = {}
    
    for algo in sorted(algorithms):
        print(f"\n{'─'*100}")
        print(f"  {algo.upper().replace('_', '-')}")
        print(f"{'─'*100}")
        print(f"{'Floor':<12} {'Poses':>8} {'Length':>10} {'Drift':>10} {'Drift%':>8} │ {'Paper ATE':>10} {'Paper%':>8} │ {'Status':<12}")
        print(f"{'─'*12}─{'─'*8}─{'─'*10}─{'─'*10}─{'─'*8}─┼─{'─'*10}─{'─'*8}─┼─{'─'*12}")
        
        all_results[algo] = {}
        
        for floor in floors:
            traj_file = traj_dir / algo / f'{floor}.txt'
            if not traj_file.exists():
                print(f"{floor:<12} {'N/A':>8}")
                continue
            
            try:
                r = analyze_trajectory(traj_file)
                all_results[algo][floor] = r
                
                # Get paper values
                paper = PAPER_RESULTS.get(algo, {}).get(floor, {})
                paper_ate = paper.get('ate_m', '-')
                paper_pct = paper.get('ate_pct', '-')
                
                # Determine status
                if paper_ate != '-':
                    ratio = r['drift_m'] / paper_ate
                    if ratio <= 1.5:
                        status = '✅ EXCELLENT'
                    elif ratio <= 2.0:
                        status = '✓ GOOD'
                    elif ratio <= 5.0:
                        status = '⚠️ ACCEPTABLE'
                    else:
                        status = '❌ POOR'
                else:
                    status = '? NO REF'
                
                paper_ate_str = f"{paper_ate:.3f}" if isinstance(paper_ate, float) else str(paper_ate)
                paper_pct_str = f"{paper_pct:.2f}%" if isinstance(paper_pct, float) else str(paper_pct)
                
                print(f"{floor:<12} {r['num_poses']:>8} {r['length_m']:>10.1f} {r['drift_m']:>10.3f} {r['drift_pct']:>7.2f}% │ {paper_ate_str:>10} {paper_pct_str:>8} │ {status}")
                
            except Exception as e:
                print(f"{floor:<12} ERROR: {e}")
    
    print("\n" + "="*100)
    print("Status: ✅ ≤1.5x paper | ✓ ≤2x paper | ⚠️ ≤5x paper | ❌ >5x paper")
    print("="*100)
    
    # Save JSON
    metrics_dir = results_dir / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    with open(metrics_dir / 'evaluation_results.json', 'w') as f:
        json.dump({'generated': datetime.now().isoformat(), 'results': all_results}, f, indent=2)
    print(f"\nResults saved to: {metrics_dir / 'evaluation_results.json'}")

if __name__ == '__main__':
    main()
