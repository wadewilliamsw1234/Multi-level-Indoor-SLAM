#!/usr/bin/env python3
"""
Evaluation Script
=================

Computes trajectory metrics (ATE, RPE) for SLAM algorithm outputs.
Generates Table IV from the paper and comparison metrics.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import evo package for trajectory evaluation
try:
    from evo.core import trajectory, sync, metrics
    from evo.tools import file_interface
    HAS_EVO = True
except ImportError:
    HAS_EVO = False
    print("Warning: evo package not available. Install with: pip install evo")


# Dataset specifications from paper
SEQUENCES = {
    '5th_floor': {'duration': 437.86, 'length': 187, 'description': 'One loop, one out and back'},
    '1st_floor': {'duration': 125.58, 'length': 65, 'description': 'One loop, open layout'},
    '4th_floor': {'duration': 131.00, 'length': 66, 'description': 'One loop, some dynamic content'},
    '2nd_floor': {'duration': 266.00, 'length': 128, 'description': 'Two loops in figure eight'},
}

# Paper results for comparison (Table IV)
PAPER_RESULTS = {
    'ORB-SLAM3': {
        '5th_floor': (0.516, 0.28),
        '1st_floor': (0.949, 1.46),
        '4th_floor': (0.483, 0.73),
        '2nd_floor': (0.310, 0.24),
    },
    'SVO': {
        '5th_floor': (0.626, 0.33),
        '1st_floor': (0.720, 1.11),
        '4th_floor': (0.482, 0.73),
        '2nd_floor': (0.371, 0.29),
    },
    'VINS-Fusion': {
        '5th_floor': (1.120, 0.60),
        '1st_floor': (2.265, 3.48),
        'full_dataset': (15.844, 2.03),
    },
    'Basalt': {
        '5th_floor': (1.214, 0.65),
        '1st_floor': (4.043, 6.22),
        '4th_floor': (1.809, 2.74),
        '2nd_floor': (3.054, 2.39),
        'full_dataset': (1.753, 0.22),
    },
    'DROID-SLAM': {
        '5th_floor': (0.441, 0.24),
        '1st_floor': (0.666, 1.02),
        '4th_floor': (0.112, 0.17),
        '2nd_floor': (0.214, 0.17),
    },
    'LeGO-LOAM': {
        '5th_floor': (0.395, 0.21),
        '1st_floor': (0.256, 0.39),
        '4th_floor': (0.789, 1.20),
        '2nd_floor': (0.286, 0.22),
    },
}


@dataclass
class TrajectoryMetrics:
    """Computed trajectory metrics."""
    algorithm: str
    sequence: str
    ate_m: float
    ate_percent: float
    rpe_trans: Optional[float] = None
    rpe_rot: Optional[float] = None
    trajectory_length: Optional[float] = None
    num_poses: int = 0


def load_tum_trajectory(filepath: Path) -> Optional[np.ndarray]:
    """
    Load trajectory from TUM format file.
    
    TUM format: timestamp tx ty tz qx qy qz qw
    
    Args:
        filepath: Path to trajectory file
        
    Returns:
        Nx8 numpy array or None if file doesn't exist
    """
    if not filepath.exists():
        return None
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                values = [float(v) for v in line.split()]
                if len(values) >= 8:
                    data.append(values[:8])
    
    if not data:
        return None
    
    return np.array(data)


def compute_trajectory_length(traj: np.ndarray) -> float:
    """
    Compute total length of trajectory.
    
    Args:
        traj: Nx8 trajectory array (TUM format)
        
    Returns:
        Total trajectory length in meters
    """
    positions = traj[:, 1:4]  # tx, ty, tz
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))


def compute_ate(
    est_traj: np.ndarray,
    ref_traj: np.ndarray,
    align: bool = True,
    correct_scale: bool = False,
) -> Tuple[float, np.ndarray]:
    """
    Compute Absolute Trajectory Error.
    
    Args:
        est_traj: Estimated trajectory (Nx8 TUM format)
        ref_traj: Reference trajectory (Nx8 TUM format)
        align: Whether to align trajectories
        correct_scale: Whether to correct scale (for monocular)
        
    Returns:
        Tuple of (ATE in meters, aligned estimated trajectory)
    """
    if not HAS_EVO:
        # Fallback: simple final position error
        final_est = est_traj[-1, 1:4]
        final_ref = ref_traj[-1, 1:4]
        ate = np.linalg.norm(final_est - final_ref)
        return ate, est_traj
    
    # Use evo package for proper alignment and ATE computation
    traj_ref = file_interface.read_tum_trajectory_file(ref_traj)
    traj_est = file_interface.read_tum_trajectory_file(est_traj)
    
    # Synchronize timestamps
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    
    # Align trajectories
    if align:
        traj_est_aligned = trajectory.align_trajectory(
            traj_est, traj_ref,
            correct_scale=correct_scale,
            correct_only_scale=False
        )
    else:
        traj_est_aligned = traj_est
    
    # Compute ATE
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    
    ate = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    
    return ate, traj_est_aligned


def compute_final_drift(
    est_traj: np.ndarray,
    ref_traj: np.ndarray,
    align_segment_ratio: float = 0.1,
) -> float:
    """
    Compute final position drift using initial segment alignment.
    
    This follows the paper's methodology (Equation 1):
    - Align trajectories using initial segment
    - Compute translational error at final position
    
    Args:
        est_traj: Estimated trajectory (Nx8 TUM format)
        ref_traj: Reference trajectory (Nx8 TUM format)
        align_segment_ratio: Ratio of trajectory to use for alignment
        
    Returns:
        Final drift in meters
    """
    # Get alignment segment
    n_align = int(len(est_traj) * align_segment_ratio)
    n_align = max(n_align, 10)  # At least 10 poses
    
    est_align = est_traj[:n_align, 1:4]
    ref_align = ref_traj[:n_align, 1:4]
    
    # Compute alignment transformation (SE3)
    # Using Procrustes analysis
    est_centroid = np.mean(est_align, axis=0)
    ref_centroid = np.mean(ref_align, axis=0)
    
    est_centered = est_align - est_centroid
    ref_centered = ref_align - ref_centroid
    
    # SVD for rotation
    H = est_centered.T @ ref_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation
    t = ref_centroid - R @ est_centroid
    
    # Apply transformation to estimated trajectory
    est_positions = est_traj[:, 1:4]
    est_aligned = (R @ est_positions.T).T + t
    
    # Compute final drift
    final_est = est_aligned[-1]
    final_ref = ref_traj[-1, 1:4]
    drift = np.linalg.norm(final_est - final_ref)
    
    return drift


def evaluate_trajectory(
    est_path: Path,
    ref_path: Path,
    algorithm: str,
    sequence: str,
    trajectory_length: Optional[float] = None,
) -> TrajectoryMetrics:
    """
    Evaluate a single trajectory against reference.
    
    Args:
        est_path: Path to estimated trajectory file
        ref_path: Path to reference trajectory file
        algorithm: Algorithm name
        sequence: Sequence name
        trajectory_length: Known trajectory length (optional)
        
    Returns:
        TrajectoryMetrics object
    """
    est_traj = load_tum_trajectory(est_path)
    ref_traj = load_tum_trajectory(ref_path)
    
    if est_traj is None:
        print(f"Warning: Could not load {est_path}")
        return TrajectoryMetrics(
            algorithm=algorithm,
            sequence=sequence,
            ate_m=float('nan'),
            ate_percent=float('nan'),
        )
    
    if ref_traj is None:
        print(f"Warning: Could not load {ref_path}")
        return TrajectoryMetrics(
            algorithm=algorithm,
            sequence=sequence,
            ate_m=float('nan'),
            ate_percent=float('nan'),
        )
    
    # Compute trajectory length if not provided
    if trajectory_length is None:
        trajectory_length = compute_trajectory_length(ref_traj)
    
    # Compute ATE
    ate_m = compute_final_drift(est_traj, ref_traj)
    ate_percent = (ate_m / trajectory_length) * 100 if trajectory_length > 0 else 0
    
    return TrajectoryMetrics(
        algorithm=algorithm,
        sequence=sequence,
        ate_m=ate_m,
        ate_percent=ate_percent,
        trajectory_length=trajectory_length,
        num_poses=len(est_traj),
    )


def find_trajectories(results_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Find all trajectory files in results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Nested dict: {algorithm: {sequence: path}}
    """
    trajectories = {}
    traj_dir = results_dir / 'trajectories'
    
    if not traj_dir.exists():
        return trajectories
    
    for algo_dir in traj_dir.iterdir():
        if algo_dir.is_dir():
            algorithm = algo_dir.name
            trajectories[algorithm] = {}
            
            for traj_file in algo_dir.glob('*.txt'):
                sequence = traj_file.stem
                trajectories[algorithm][sequence] = traj_file
    
    return trajectories


def evaluate_all(
    results_dir: Path,
    ground_truth_dir: Optional[Path] = None,
) -> List[TrajectoryMetrics]:
    """
    Evaluate all trajectories in results directory.
    
    Args:
        results_dir: Path to results directory
        ground_truth_dir: Path to ground truth (default: results_dir/ground_truth)
        
    Returns:
        List of TrajectoryMetrics
    """
    if ground_truth_dir is None:
        ground_truth_dir = results_dir / 'ground_truth'
    
    trajectories = find_trajectories(results_dir)
    all_metrics = []
    
    for algorithm, sequences in trajectories.items():
        for sequence, est_path in sequences.items():
            # Look for ground truth
            ref_path = ground_truth_dir / f'{sequence}.txt'
            
            # Use LeGO-LOAM as pseudo ground truth if available
            if not ref_path.exists():
                lego_path = results_dir / 'trajectories' / 'lego_loam' / f'{sequence}.txt'
                if lego_path.exists() and algorithm != 'lego_loam':
                    ref_path = lego_path
            
            if not ref_path.exists():
                print(f"Warning: No ground truth for {sequence}")
                continue
            
            # Get known trajectory length
            traj_length = SEQUENCES.get(sequence, {}).get('length')
            
            metrics = evaluate_trajectory(
                est_path, ref_path, algorithm, sequence, traj_length
            )
            all_metrics.append(metrics)
            
            print(f"{algorithm}/{sequence}: ATE={metrics.ate_m:.3f}m ({metrics.ate_percent:.2f}%)")
    
    return all_metrics


def generate_table_iv(
    metrics: List[TrajectoryMetrics],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate Table IV (algorithm comparison) from metrics.
    
    Args:
        metrics: List of TrajectoryMetrics
        output_path: Path to save CSV (optional)
        
    Returns:
        Pandas DataFrame with results
    """
    # Organize by algorithm and sequence
    data = {}
    for m in metrics:
        if m.algorithm not in data:
            data[m.algorithm] = {}
        data[m.algorithm][m.sequence] = (m.ate_m, m.ate_percent)
    
    # Create DataFrame
    sequences = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor', 'full_dataset']
    rows = []
    
    for algorithm in sorted(data.keys()):
        row = {'Algorithm': algorithm}
        for seq in sequences:
            if seq in data[algorithm]:
                ate_m, ate_pct = data[algorithm][seq]
                row[f'{seq}_ATE(m)'] = f'{ate_m:.3f}'
                row[f'{seq}_%'] = f'{ate_pct:.2f}%'
            else:
                row[f'{seq}_ATE(m)'] = '-'
                row[f'{seq}_%'] = '-'
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        # Also save LaTeX version
        latex_path = output_path.with_suffix('.tex')
        df.to_latex(latex_path, index=False)
    
    return df


def compare_with_paper(
    metrics: List[TrajectoryMetrics],
) -> pd.DataFrame:
    """
    Compare results with paper values.
    
    Args:
        metrics: List of TrajectoryMetrics
        
    Returns:
        DataFrame with comparison
    """
    rows = []
    
    for m in metrics:
        paper_result = PAPER_RESULTS.get(m.algorithm, {}).get(m.sequence)
        
        row = {
            'Algorithm': m.algorithm,
            'Sequence': m.sequence,
            'Our ATE(m)': m.ate_m,
            'Our %': m.ate_percent,
        }
        
        if paper_result:
            paper_ate, paper_pct = paper_result
            row['Paper ATE(m)'] = paper_ate
            row['Paper %'] = paper_pct
            row['Diff ATE(m)'] = m.ate_m - paper_ate
            row['Diff %'] = abs(m.ate_m - paper_ate) / paper_ate * 100 if paper_ate > 0 else 0
        else:
            row['Paper ATE(m)'] = '-'
            row['Paper %'] = '-'
            row['Diff ATE(m)'] = '-'
            row['Diff %'] = '-'
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_metrics_json(
    metrics: List[TrajectoryMetrics],
    output_path: Path,
):
    """Save metrics to JSON file."""
    data = []
    for m in metrics:
        data.append({
            'algorithm': m.algorithm,
            'sequence': m.sequence,
            'ate_m': m.ate_m,
            'ate_percent': m.ate_percent,
            'trajectory_length': m.trajectory_length,
            'num_poses': m.num_poses,
        })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Evaluate SLAM trajectories')
    
    parser.add_argument('--results-dir', type=Path, default=Path('/results'),
                       help='Results directory')
    parser.add_argument('--ground-truth', type=Path,
                       help='Ground truth directory')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all trajectories')
    parser.add_argument('--table-only', action='store_true',
                       help='Only generate Table IV')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with paper results')
    parser.add_argument('--output', '-o', type=Path,
                       help='Output file path')
    
    args = parser.parse_args()
    
    if args.all or args.table_only:
        metrics = evaluate_all(args.results_dir, args.ground_truth)
        
        if not metrics:
            print("No trajectories found to evaluate.")
            return
        
        # Generate Table IV
        output_csv = args.output or args.results_dir / 'metrics' / 'table_iv.csv'
        df = generate_table_iv(metrics, output_csv)
        print("\nTable IV:")
        print(df.to_string(index=False))
        
        # Save detailed JSON
        json_path = args.results_dir / 'metrics' / 'detailed_metrics.json'
        save_metrics_json(metrics, json_path)
        
        if args.compare:
            comparison = compare_with_paper(metrics)
            print("\nComparison with Paper:")
            print(comparison.to_string(index=False))
            
            comp_path = args.results_dir / 'metrics' / 'paper_comparison.csv'
            comparison.to_csv(comp_path, index=False)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
