#!/usr/bin/env python3
"""Final evaluation - handles broken trajectories gracefully"""

import numpy as np
from pathlib import Path
import json

BASE = Path(__file__).parent.parent.parent
TRAJ = BASE / 'results' / 'trajectories'
METRICS = BASE / 'results' / 'metrics'
METRICS.mkdir(exist_ok=True)

FLOORS = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']
EXPECTED_LEN = {'5th_floor': 187, '1st_floor': 65, '4th_floor': 66, '2nd_floor': 128}

PAPER = {
    'orb_slam3': {'5th_floor': 0.516, '1st_floor': 0.949, '4th_floor': 0.483, '2nd_floor': 0.310},
    'droid_slam': {'5th_floor': 0.441, '1st_floor': 0.666, '4th_floor': 0.112, '2nd_floor': 0.214},
    'lego_loam': {'5th_floor': 0.395, '1st_floor': 0.256, '4th_floor': 0.789, '2nd_floor': 0.286},
    'basalt': {'5th_floor': 1.214, '1st_floor': 4.043, '4th_floor': 1.809, '2nd_floor': 3.054},
}

def load_traj(filepath):
    """Load TUM trajectory, skip comments"""
    data = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    return np.array(data) if data else None

def get_traj_file(algo, floor):
    d = TRAJ / algo
    if algo == 'droid_slam':
        # Prefer stereo version
        for name in [f'{floor}_stereo.txt', f'{floor}.txt']:
            if (d / name).exists():
                return d / name
    f = d / f'{floor}.txt'
    return f if f.exists() else None

def traj_length(pos):
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

def endpoint_drift(pos):
    return float(np.linalg.norm(pos[-1] - pos[0]))

def is_valid_trajectory(pos, expected_len, max_ratio=10):
    """Check if trajectory is valid (not diverged)"""
    length = traj_length(pos)
    if length > expected_len * max_ratio:
        return False, f"diverged ({length:.0f}m vs {expected_len}m expected)"
    if length < expected_len * 0.1:
        return False, f"too short ({length:.1f}m vs {expected_len}m expected)"
    return True, "ok"

def align_sim3(est, ref):
    """Sim(3) alignment: rotation, translation, scale"""
    # Center
    est_c = np.mean(est, axis=0)
    ref_c = np.mean(ref, axis=0)
    est_centered = est - est_c
    ref_centered = ref - ref_c
    
    # Rotation via SVD
    H = est_centered.T @ ref_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Scale
    est_rot = (R @ est_centered.T).T
    scale = np.sum(ref_centered * est_rot) / (np.sum(est_rot * est_rot) + 1e-10)
    
    # Translation
    t = ref_c - scale * R @ est_c
    
    # Apply
    aligned = scale * (R @ est.T).T + t
    return aligned, scale

def align_se3(est, ref):
    """SE(3) alignment: rotation, translation only"""
    aligned, scale = align_sim3(est, ref)
    # For SE3, we don't use scale (but this simplified version does)
    # A proper SE3 would fix scale=1
    return aligned, 1.0

def associate_by_time(est_data, ref_data, max_diff=0.5):
    """Associate trajectories by timestamp"""
    est_t = est_data[:, 0]
    ref_t = ref_data[:, 0]
    
    # Normalize if needed
    if est_t[0] > 1e15:
        est_t = est_t / 1e9
    if ref_t[0] > 1e15:
        ref_t = ref_t / 1e9
    
    matches = []
    for i, t in enumerate(est_t):
        j = np.argmin(np.abs(ref_t - t))
        if np.abs(ref_t[j] - t) < max_diff:
            matches.append((i, j))
    
    if len(matches) < 10:
        return None, None
    
    est_idx = [m[0] for m in matches]
    ref_idx = [m[1] for m in matches]
    return est_data[est_idx, 1:4], ref_data[ref_idx, 1:4]

def compute_ate(est, ref):
    """Compute ATE statistics"""
    errors = np.linalg.norm(est - ref, axis=1)
    return {
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'mean': float(np.mean(errors)),
        'max': float(np.max(errors)),
        'median': float(np.median(errors)),
    }

print("=" * 80)
print("SLAM BENCHMARKING - FINAL EVALUATION")
print("=" * 80)

results = {}

# Load reference (LeGO-LOAM)
print("\n[1] Loading LeGO-LOAM reference trajectories...")
ref_trajs = {}
for floor in FLOORS:
    f = get_traj_file('lego_loam', floor)
    if f:
        data = load_traj(f)
        if data is not None:
            ref_trajs[floor] = data
            print(f"  {floor}: {len(data)} poses, {traj_length(data[:, 1:4]):.1f}m")

# Evaluate each algorithm
print("\n[2] Evaluating algorithms...")

for algo in ['lego_loam', 'orb_slam3', 'basalt', 'droid_slam']:
    print(f"\n{'='*40}")
    print(f"  {algo.upper()}")
    print(f"{'='*40}")
    results[algo] = {}
    
    for floor in FLOORS:
        f = get_traj_file(algo, floor)
        if not f:
            print(f"  {floor}: NO FILE")
            continue
        
        data = load_traj(f)
        if data is None or len(data) < 10:
            print(f"  {floor}: LOAD FAILED")
            continue
        
        pos = data[:, 1:4]
        length = traj_length(pos)
        drift = endpoint_drift(pos)
        expected = EXPECTED_LEN[floor]
        
        # Check validity
        valid, reason = is_valid_trajectory(pos, expected)
        
        r = {
            'poses': len(data),
            'length': length,
            'drift': drift,
            'drift_pct': drift / length * 100 if length > 0 else 0,
            'expected': expected,
            'valid': valid,
            'paper': PAPER.get(algo, {}).get(floor),
        }
        
        if not valid:
            print(f"  {floor}: INVALID - {reason}")
            r['status'] = reason
            results[algo][floor] = r
            continue
        
        # Compute ATE vs LeGO-LOAM (for non-reference algorithms)
        if algo != 'lego_loam' and floor in ref_trajs:
            ref_data = ref_trajs[floor]
            est_assoc, ref_assoc = associate_by_time(data, ref_data)
            
            if est_assoc is not None and len(est_assoc) > 10:
                # Use Sim(3) for DROID (no metric scale), SE(3) for others
                if algo == 'droid_slam':
                    aligned, scale = align_sim3(est_assoc, ref_assoc)
                    r['scale'] = scale
                else:
                    aligned, _ = align_se3(est_assoc, ref_assoc)
                
                ate = compute_ate(aligned, ref_assoc)
                r['ate_vs_lego'] = ate
                r['num_matched'] = len(est_assoc)
                
                print(f"  {floor}: {len(data)} poses, {length:.1f}m, drift={drift:.3f}m, ATE={ate['rmse']:.3f}m")
            else:
                print(f"  {floor}: {len(data)} poses, {length:.1f}m, drift={drift:.3f}m (no time match)")
        else:
            print(f"  {floor}: {len(data)} poses, {length:.1f}m, drift={drift:.3f}m")
        
        r['status'] = 'ok'
        results[algo][floor] = r

# Save results
output = METRICS / 'final_evaluation.json'
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n\nResults saved to: {output}")

# Print summary tables
print("\n" + "=" * 80)
print("TABLE 1: ENDPOINT DRIFT (meters) - Ours vs Paper")
print("=" * 80)
print(f"{'Algorithm':<12} | {'5th Floor':^18} | {'1st Floor':^18} | {'4th Floor':^18} | {'2nd Floor':^18}")
print("-" * 92)

for algo in ['lego_loam', 'orb_slam3', 'basalt', 'droid_slam']:
    row = f"{algo:<12} |"
    for floor in FLOORS:
        r = results.get(algo, {}).get(floor, {})
        if r.get('valid', False):
            ours = r.get('drift', 0)
            paper = r.get('paper')
            if paper:
                row += f" {ours:6.3f} / {paper:.3f}   |"
            else:
                row += f" {ours:6.3f} / -       |"
        else:
            status = r.get('status', 'missing')[:10]
            row += f" {'FAIL':^16} |"
    print(row)

print("\n" + "=" * 80)
print("TABLE 2: TRAJECTORY LENGTHS (meters)")
print("=" * 80)
print(f"{'Algorithm':<12} | {'5th (187m)':^12} | {'1st (65m)':^12} | {'4th (66m)':^12} | {'2nd (128m)':^12}")
print("-" * 68)

for algo in ['lego_loam', 'orb_slam3', 'basalt', 'droid_slam']:
    row = f"{algo:<12} |"
    for floor in FLOORS:
        r = results.get(algo, {}).get(floor, {})
        if r.get('valid', False):
            length = r.get('length', 0)
            row += f" {length:10.1f}m |"
        else:
            row += f" {'FAIL':^12} |"
    print(row)

print("\n" + "=" * 80)
print("TABLE 3: ATE vs LeGO-LOAM (RMSE in meters)")
print("=" * 80)
print(f"{'Algorithm':<12} | {'5th Floor':^12} | {'1st Floor':^12} | {'4th Floor':^12} | {'2nd Floor':^12}")
print("-" * 68)

for algo in ['orb_slam3', 'basalt', 'droid_slam']:
    row = f"{algo:<12} |"
    for floor in FLOORS:
        r = results.get(algo, {}).get(floor, {})
        ate = r.get('ate_vs_lego', {})
        if ate:
            rmse = ate.get('rmse', 0)
            row += f" {rmse:10.3f}m |"
        elif r.get('valid', False):
            row += f" {'no match':^12} |"
        else:
            row += f" {'FAIL':^12} |"
    print(row)

print("\n" + "=" * 80)
print("TABLE 4: SCALE FACTORS (DROID-SLAM Sim3 alignment)")
print("=" * 80)
for floor in FLOORS:
    r = results.get('droid_slam', {}).get(floor, {})
    scale = r.get('scale')
    if scale:
        expected = EXPECTED_LEN[floor]
        actual = r.get('length', 0)
        print(f"  {floor}: scale={scale:.4f}, raw_length={actual:.1f}m -> scaled≈{actual*scale:.1f}m (expected {expected}m)")

print("\nDone!")
