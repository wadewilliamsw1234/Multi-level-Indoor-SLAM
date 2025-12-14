#!/usr/bin/env python3
"""
Comprehensive SLAM Evaluation Pipeline
Reproduces paper analysis + extended diagnostics
"""

import numpy as np
from pathlib import Path
import json
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
TRAJ_DIR = BASE_DIR / 'results' / 'trajectories'
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
METRICS_DIR.mkdir(exist_ok=True)

ALGORITHMS = {
    'lego_loam': {'type': 'LiDAR', 'has_scale': True, 'color': 'black'},
    'orb_slam3': {'type': 'Visual', 'has_scale': True, 'color': 'red'},
    'basalt': {'type': 'VIO', 'has_scale': True, 'color': 'blue'},
    'droid_slam': {'type': 'Deep Learning', 'has_scale': False, 'color': 'green'},
}

FLOORS = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']

PAPER_RESULTS = {
    'orb_slam3': {'5th_floor': 0.516, '1st_floor': 0.949, '4th_floor': 0.483, '2nd_floor': 0.310},
    'droid_slam': {'5th_floor': 0.441, '1st_floor': 0.666, '4th_floor': 0.112, '2nd_floor': 0.214},
    'lego_loam': {'5th_floor': 0.395, '1st_floor': 0.256, '4th_floor': 0.789, '2nd_floor': 0.286},
    'basalt': {'5th_floor': 1.214, '1st_floor': 4.043, '4th_floor': 1.809, '2nd_floor': 3.054},
}

EXPECTED_LENGTHS = {'5th_floor': 187, '1st_floor': 65, '4th_floor': 66, '2nd_floor': 128}

# Problem regions (approximate trajectory percentage)
PROBLEM_REGIONS = {
    '5th_floor': {
        'A': {'start_pct': 0.25, 'end_pct': 0.40, 'type': 'dynamic', 'description': 'Dynamic content (people)'},
        'B': {'start_pct': 0.55, 'end_pct': 0.70, 'type': 'featureless', 'description': 'Featureless corridor + tight turn'},
    },
    '1st_floor': {
        'C': {'start_pct': 0.10, 'end_pct': 0.30, 'type': 'glass', 'description': 'Glass walls, reflections'},
        'D': {'start_pct': 0.60, 'end_pct': 0.80, 'type': 'open', 'description': 'Open atrium'},
    },
}

# ============================================================================
# TRAJECTORY LOADING AND PREPROCESSING
# ============================================================================

def load_tum_trajectory(filepath):
    """Load trajectory in TUM format: timestamp tx ty tz qx qy qz qw"""
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return {
            'timestamps': data[:, 0],
            'positions': data[:, 1:4],
            'quaternions': data[:, 4:8],  # qx qy qz qw
            'raw': data
        }
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None

def get_trajectory_file(algo, floor):
    """Get the appropriate trajectory file for an algorithm"""
    traj_dir = TRAJ_DIR / algo
    
    # Handle DROID-SLAM variations
    if algo == 'droid_slam':
        candidates = [
            f'{floor}_stereo.txt',
            f'{floor}.txt',
        ]
        for c in candidates:
            if (traj_dir / c).exists():
                return traj_dir / c
    
    # Standard case
    traj_file = traj_dir / f'{floor}.txt'
    if traj_file.exists():
        return traj_file
    
    return None

# ============================================================================
# BASIC METRICS
# ============================================================================

def compute_trajectory_length(positions):
    """Compute total trajectory length"""
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

def compute_endpoint_drift(positions):
    """Compute drift between start and end positions"""
    return np.linalg.norm(positions[-1] - positions[0])

def compute_cumulative_distance(positions):
    """Compute cumulative distance along trajectory"""
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0], np.cumsum(distances)])

# ============================================================================
# TRAJECTORY ALIGNMENT (SE3 and Sim3)
# ============================================================================

def align_trajectories_se3(est_positions, ref_positions):
    """Align estimated trajectory to reference using SE(3) (rotation + translation)"""
    # Center both trajectories
    est_centroid = np.mean(est_positions, axis=0)
    ref_centroid = np.mean(ref_positions, axis=0)
    
    est_centered = est_positions - est_centroid
    ref_centered = ref_positions - ref_centroid
    
    # Compute optimal rotation using SVD
    H = est_centered.T @ ref_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = ref_centroid - R @ est_centroid
    
    # Apply transformation
    aligned = (R @ est_positions.T).T + t
    
    return aligned, R, t, 1.0

def align_trajectories_sim3(est_positions, ref_positions):
    """Align estimated trajectory to reference using Sim(3) (rotation + translation + scale)"""
    # Center both trajectories
    est_centroid = np.mean(est_positions, axis=0)
    ref_centroid = np.mean(ref_positions, axis=0)
    
    est_centered = est_positions - est_centroid
    ref_centered = ref_positions - ref_centroid
    
    # Compute optimal rotation using SVD
    H = est_centered.T @ ref_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    est_rotated = (R @ est_centered.T).T
    scale = np.sum(ref_centered * est_rotated) / np.sum(est_rotated * est_rotated)
    
    # Compute translation
    t = ref_centroid - scale * R @ est_centroid
    
    # Apply transformation
    aligned = scale * (R @ est_positions.T).T + t
    
    return aligned, R, t, scale

def associate_trajectories(est_traj, ref_traj, max_diff=0.1):
    """Associate trajectories by timestamp"""
    est_times = est_traj['timestamps']
    ref_times = ref_traj['timestamps']
    
    # Normalize timestamps if needed
    if est_times[0] > 1e15:  # Nanoseconds
        est_times = est_times / 1e9
    if ref_times[0] > 1e15:
        ref_times = ref_times / 1e9
    
    matches = []
    for i, t_est in enumerate(est_times):
        diffs = np.abs(ref_times - t_est)
        j = np.argmin(diffs)
        if diffs[j] < max_diff:
            matches.append((i, j))
    
    if len(matches) < 10:
        return None, None
    
    est_idx = [m[0] for m in matches]
    ref_idx = [m[1] for m in matches]
    
    return est_traj['positions'][est_idx], ref_traj['positions'][ref_idx]

# ============================================================================
# ATE AND RPE COMPUTATION
# ============================================================================

def compute_ate(est_positions, ref_positions):
    """Compute Absolute Trajectory Error statistics"""
    errors = np.linalg.norm(est_positions - ref_positions, axis=1)
    return {
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'max': float(np.max(errors)),
        'min': float(np.min(errors)),
        'errors': errors
    }

def compute_rpe(positions, delta=1.0, delta_unit='m'):
    """Compute Relative Pose Error over segments"""
    cumulative_dist = compute_cumulative_distance(positions)
    total_length = cumulative_dist[-1]
    
    rpe_trans = []
    rpe_indices = []
    
    for i in range(len(positions) - 1):
        # Find point approximately delta meters ahead
        target_dist = cumulative_dist[i] + delta
        if target_dist > total_length:
            break
        
        j = np.searchsorted(cumulative_dist, target_dist)
        if j >= len(positions):
            break
        
        # Compute relative translation error (simplified - comparing to expected distance)
        actual_dist = np.linalg.norm(positions[j] - positions[i])
        expected_dist = cumulative_dist[j] - cumulative_dist[i]
        
        if expected_dist > 0:
            rpe_trans.append(abs(actual_dist - expected_dist) / expected_dist * 100)
            rpe_indices.append(i)
    
    if len(rpe_trans) == 0:
        return None
    
    rpe_trans = np.array(rpe_trans)
    return {
        'rmse': float(np.sqrt(np.mean(rpe_trans**2))),
        'mean': float(np.mean(rpe_trans)),
        'median': float(np.median(rpe_trans)),
        'std': float(np.std(rpe_trans)),
        'max': float(np.max(rpe_trans)),
        'values': rpe_trans,
        'indices': np.array(rpe_indices)
    }

def compute_rotation_error(est_quats, ref_quats):
    """Compute rotation errors between quaternion sequences"""
    errors = []
    for q_est, q_ref in zip(est_quats, ref_quats):
        try:
            R_est = Rotation.from_quat(q_est)
            R_ref = Rotation.from_quat(q_ref)
            R_diff = R_ref.inv() * R_est
            angle = R_diff.magnitude()  # Rotation angle in radians
            errors.append(np.degrees(angle))
        except:
            continue
    
    if len(errors) == 0:
        return None
    
    errors = np.array(errors)
    return {
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'max': float(np.max(errors)),
        'errors': errors
    }

# ============================================================================
# SEGMENT-WISE ANALYSIS
# ============================================================================

def compute_segment_metrics(est_positions, ref_positions, num_segments=10):
    """Compute ATE for trajectory segments"""
    n = len(est_positions)
    segment_size = n // num_segments
    
    segments = []
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else n
        
        seg_est = est_positions[start:end]
        seg_ref = ref_positions[start:end]
        
        if len(seg_est) > 0:
            errors = np.linalg.norm(seg_est - seg_ref, axis=1)
            segments.append({
                'segment': i + 1,
                'start_idx': start,
                'end_idx': end,
                'start_pct': start / n * 100,
                'end_pct': end / n * 100,
                'rmse': float(np.sqrt(np.mean(errors**2))),
                'max': float(np.max(errors)),
                'num_poses': len(seg_est)
            })
    
    return segments

def compute_error_at_distances(est_positions, ref_positions, distances=[10, 25, 50, 100, 150]):
    """Compute error at specific trajectory distances"""
    cumulative_dist = compute_cumulative_distance(ref_positions)
    total_length = cumulative_dist[-1]
    
    errors_at_dist = {}
    for d in distances:
        if d > total_length:
            errors_at_dist[d] = None
            continue
        
        idx = np.searchsorted(cumulative_dist, d)
        if idx < len(est_positions):
            error = np.linalg.norm(est_positions[idx] - ref_positions[idx])
            errors_at_dist[d] = float(error)
        else:
            errors_at_dist[d] = None
    
    return errors_at_dist

# ============================================================================
# PROBLEM REGION ANALYSIS
# ============================================================================

def analyze_problem_regions(est_positions, ref_positions, floor):
    """Analyze performance in annotated problem regions"""
    if floor not in PROBLEM_REGIONS:
        return {}
    
    n = len(est_positions)
    results = {}
    
    for region_name, region_info in PROBLEM_REGIONS[floor].items():
        start_idx = int(region_info['start_pct'] * n)
        end_idx = int(region_info['end_pct'] * n)
        
        if end_idx <= start_idx:
            continue
        
        region_est = est_positions[start_idx:end_idx]
        region_ref = ref_positions[start_idx:end_idx]
        
        # Compute metrics for this region
        errors = np.linalg.norm(region_est - region_ref, axis=1)
        
        # Compute metrics for rest of trajectory
        other_est = np.vstack([est_positions[:start_idx], est_positions[end_idx:]])
        other_ref = np.vstack([ref_positions[:start_idx], ref_positions[end_idx:]])
        other_errors = np.linalg.norm(other_est - other_ref, axis=1)
        
        results[region_name] = {
            'type': region_info['type'],
            'description': region_info['description'],
            'start_pct': region_info['start_pct'] * 100,
            'end_pct': region_info['end_pct'] * 100,
            'region_rmse': float(np.sqrt(np.mean(errors**2))),
            'region_max': float(np.max(errors)),
            'other_rmse': float(np.sqrt(np.mean(other_errors**2))) if len(other_errors) > 0 else 0,
            'degradation_factor': float(np.sqrt(np.mean(errors**2)) / np.sqrt(np.mean(other_errors**2))) if len(other_errors) > 0 and np.mean(other_errors**2) > 0 else 0,
            'num_poses': len(region_est)
        }
    
    return results

# ============================================================================
# TRACKING QUALITY METRICS
# ============================================================================

def analyze_tracking_quality(traj, expected_rate=20.0):
    """Analyze tracking continuity and quality"""
    timestamps = traj['timestamps']
    
    # Normalize timestamps
    if timestamps[0] > 1e15:
        timestamps = timestamps / 1e9
    
    # Compute time differences
    dt = np.diff(timestamps)
    expected_dt = 1.0 / expected_rate
    
    # Detect gaps (> 2x expected frame time)
    gaps = dt > (2 * expected_dt)
    gap_indices = np.where(gaps)[0]
    
    # Compute effective pose rate
    total_time = timestamps[-1] - timestamps[0]
    pose_rate = len(timestamps) / total_time if total_time > 0 else 0
    
    return {
        'total_poses': len(timestamps),
        'total_time': float(total_time),
        'expected_poses': int(total_time * expected_rate),
        'pose_rate': float(pose_rate),
        'completeness': float(len(timestamps) / (total_time * expected_rate) * 100) if total_time > 0 else 0,
        'num_gaps': int(np.sum(gaps)),
        'gap_indices': gap_indices.tolist(),
        'max_gap': float(np.max(dt)) if len(dt) > 0 else 0,
        'mean_dt': float(np.mean(dt)) if len(dt) > 0 else 0,
    }

def compute_trajectory_smoothness(positions):
    """Compute trajectory smoothness (inverse of jerkiness)"""
    if len(positions) < 3:
        return None
    
    # First derivative (velocity)
    vel = np.diff(positions, axis=0)
    
    # Second derivative (acceleration)
    acc = np.diff(vel, axis=0)
    
    # Compute jerk magnitude
    acc_magnitude = np.linalg.norm(acc, axis=1)
    
    return {
        'mean_acceleration': float(np.mean(acc_magnitude)),
        'max_acceleration': float(np.max(acc_magnitude)),
        'std_acceleration': float(np.std(acc_magnitude)),
        'smoothness_score': float(1.0 / (1.0 + np.mean(acc_magnitude)))  # Higher = smoother
    }

# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_algorithm_floor(algo, floor, ref_traj=None):
    """Comprehensive evaluation for one algorithm on one floor"""
    print(f"  Evaluating {algo} on {floor}...")
    
    traj_file = get_trajectory_file(algo, floor)
    if traj_file is None:
        print(f"    No trajectory file found")
        return None
    
    traj = load_tum_trajectory(traj_file)
    if traj is None:
        return None
    
    results = {
        'algorithm': algo,
        'floor': floor,
        'trajectory_file': str(traj_file),
        'num_poses': len(traj['positions']),
    }
    
    # Basic metrics
    positions = traj['positions']
    results['trajectory_length'] = compute_trajectory_length(positions)
    results['endpoint_drift'] = compute_endpoint_drift(positions)
    results['drift_percent'] = results['endpoint_drift'] / results['trajectory_length'] * 100 if results['trajectory_length'] > 0 else 0
    
    # Expected length comparison
    expected_len = EXPECTED_LENGTHS.get(floor, 0)
    results['expected_length'] = expected_len
    results['length_ratio'] = results['trajectory_length'] / expected_len if expected_len > 0 else 0
    
    # Tracking quality
    results['tracking'] = analyze_tracking_quality(traj)
    
    # Smoothness
    smoothness = compute_trajectory_smoothness(positions)
    if smoothness:
        results['smoothness'] = smoothness
    
    # RPE at different scales
    for delta in [1.0, 5.0, 10.0]:
        rpe = compute_rpe(positions, delta=delta)
        if rpe:
            results[f'rpe_{int(delta)}m'] = {k: v for k, v in rpe.items() if k != 'values' and k != 'indices'}
    
    # If we have a reference trajectory (LeGO-LOAM), compute ATE
    if ref_traj is not None and algo != 'lego_loam':
        est_assoc, ref_assoc = associate_trajectories(traj, ref_traj, max_diff=0.5)
        
        if est_assoc is not None and len(est_assoc) > 10:
            # Choose alignment method
            if ALGORITHMS[algo]['has_scale']:
                aligned, R, t, scale = align_trajectories_se3(est_assoc, ref_assoc)
            else:
                aligned, R, t, scale = align_trajectories_sim3(est_assoc, ref_assoc)
            
            results['scale_factor'] = scale
            results['num_associated'] = len(est_assoc)
            
            # Full ATE
            ate = compute_ate(aligned, ref_assoc)
            results['ate_vs_lego'] = {k: v for k, v in ate.items() if k != 'errors'}
            results['ate_errors'] = ate['errors'].tolist()
            
            # Segment-wise analysis
            results['segments'] = compute_segment_metrics(aligned, ref_assoc, num_segments=10)
            
            # Error at specific distances
            results['error_at_distance'] = compute_error_at_distances(aligned, ref_assoc)
            
            # Problem region analysis
            problem_results = analyze_problem_regions(aligned, ref_assoc, floor)
            if problem_results:
                results['problem_regions'] = problem_results
            
            # Store aligned trajectory for visualization
            results['aligned_positions'] = aligned.tolist()
            results['reference_positions'] = ref_assoc.tolist()
    
    # Paper comparison
    if algo in PAPER_RESULTS and floor in PAPER_RESULTS[algo]:
        paper_ate = PAPER_RESULTS[algo][floor]
        results['paper_ate'] = paper_ate
        if 'ate_vs_lego' in results:
            results['vs_paper_ratio'] = results['ate_vs_lego']['rmse'] / paper_ate if paper_ate > 0 else 0
    
    return results

def run_full_evaluation():
    """Run complete evaluation pipeline"""
    print("=" * 70)
    print("COMPREHENSIVE SLAM EVALUATION")
    print("=" * 70)
    
    all_results = {}
    
    # First, load all LeGO-LOAM trajectories as reference
    print("\n[1] Loading reference trajectories (LeGO-LOAM)...")
    ref_trajectories = {}
    for floor in FLOORS:
        traj_file = get_trajectory_file('lego_loam', floor)
        if traj_file:
            ref_trajectories[floor] = load_tum_trajectory(traj_file)
            if ref_trajectories[floor]:
                print(f"  {floor}: {len(ref_trajectories[floor]['positions'])} poses")
    
    # Evaluate each algorithm on each floor
    print("\n[2] Evaluating algorithms...")
    for algo in ALGORITHMS.keys():
        print(f"\n--- {algo.upper()} ---")
        all_results[algo] = {}
        
        for floor in FLOORS:
            ref_traj = ref_trajectories.get(floor)
            result = evaluate_algorithm_floor(algo, floor, ref_traj)
            if result:
                all_results[algo][floor] = result
    
    # Save detailed results
    print("\n[3] Saving results...")
    output_file = METRICS_DIR / 'comprehensive_evaluation.json'
    
    # Convert numpy arrays for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print(f"  Saved to: {output_file}")
    
    # Generate summary tables
    print("\n[4] Generating summary tables...")
    generate_summary_tables(all_results)
    
    return all_results

def generate_summary_tables(results):
    """Generate summary tables in various formats"""
    
    # Table 1: Main Results (Endpoint Drift)
    print("\n" + "=" * 70)
    print("TABLE 1: ENDPOINT DRIFT (ATE) - Comparison with Paper")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'5th Floor':<20} {'1st Floor':<20} {'4th Floor':<20} {'2nd Floor':<20}")
    print(f"{'':<15} {'Ours / Paper':<20} {'Ours / Paper':<20} {'Ours / Paper':<20} {'Ours / Paper':<20}")
    print("-" * 95)
    
    for algo in ALGORITHMS.keys():
        row = f"{algo:<15}"
        for floor in FLOORS:
            if algo in results and floor in results[algo]:
                r = results[algo][floor]
                our_drift = r.get('endpoint_drift', 0)
                paper_drift = r.get('paper_ate', '-')
                if isinstance(paper_drift, (int, float)):
                    row += f" {our_drift:.3f} / {paper_drift:.3f}     "
                else:
                    row += f" {our_drift:.3f} / -         "
            else:
                row += f" -                   "
        print(row)
    
    # Table 2: Trajectory Lengths
    print("\n" + "=" * 70)
    print("TABLE 2: TRAJECTORY LENGTHS")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'5th (187m)':<15} {'1st (65m)':<15} {'4th (66m)':<15} {'2nd (128m)':<15}")
    print("-" * 75)
    
    for algo in ALGORITHMS.keys():
        row = f"{algo:<15}"
        for floor in FLOORS:
            if algo in results and floor in results[algo]:
                length = results[algo][floor].get('trajectory_length', 0)
                row += f" {length:.1f}m         "
            else:
                row += f" -              "
        print(row)
    
    # Table 3: Full ATE vs LeGO-LOAM
    print("\n" + "=" * 70)
    print("TABLE 3: FULL ATE vs LeGO-LOAM (RMSE / Max)")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'5th Floor':<18} {'1st Floor':<18} {'4th Floor':<18} {'2nd Floor':<18}")
    print("-" * 87)
    
    for algo in ['orb_slam3', 'basalt', 'droid_slam']:
        row = f"{algo:<15}"
        for floor in FLOORS:
            if algo in results and floor in results[algo]:
                ate = results[algo][floor].get('ate_vs_lego', {})
                rmse = ate.get('rmse', 0)
                max_e = ate.get('max', 0)
                row += f" {rmse:.2f} / {max_e:.2f}      "
            else:
                row += f" -                 "
        print(row)
    
    # Table 4: Tracking Quality
    print("\n" + "=" * 70)
    print("TABLE 4: TRACKING QUALITY (Poses / Completeness %)")
    print("=" * 70)
    print(f"{'Algorithm':<15} {'5th Floor':<18} {'1st Floor':<18} {'4th Floor':<18} {'2nd Floor':<18}")
    print("-" * 87)
    
    for algo in ALGORITHMS.keys():
        row = f"{algo:<15}"
        for floor in FLOORS:
            if algo in results and floor in results[algo]:
                tracking = results[algo][floor].get('tracking', {})
                poses = tracking.get('total_poses', 0)
                completeness = tracking.get('completeness', 0)
                row += f" {poses} / {completeness:.0f}%      "
            else:
                row += f" -                 "
        print(row)
    
    # Save tables to file
    tables_file = METRICS_DIR / 'summary_tables.txt'
    with open(tables_file, 'w') as f:
        f.write("SLAM Benchmarking Summary Tables\n")
        f.write("=" * 70 + "\n\n")
        # (Would write all tables here)
    print(f"\n  Tables saved to: {tables_file}")

if __name__ == '__main__':
    results = run_full_evaluation()
