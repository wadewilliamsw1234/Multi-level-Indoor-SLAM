#!/usr/bin/env python3
"""
evaluate_trajectories.py - Compute ATE metrics for SLAM trajectories
Compares algorithm trajectories against LeGO-LOAM pseudo ground truth
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Check for evo package
try:
    from evo.core import trajectory, sync, metrics
    from evo.tools import file_interface
    from evo.core.metrics import PoseRelation, Unit
except ImportError:
    print("Installing evo package...")
    os.system("pip install evo --break-system-packages -q")
    from evo.core import trajectory, sync, metrics
    from evo.tools import file_interface
    from evo.core.metrics import PoseRelation, Unit

@dataclass
class TrajectoryResult:
    """Results for a single trajectory evaluation"""
    floor: str
    algorithm: str
    ate_m: float
    ate_percent: float
    trajectory_length: float
    pose_count: int
    success: bool
    error_msg: str = ""


def load_trajectory(filepath: Path) -> Optional[trajectory.PoseTrajectory3D]:
    """Load a TUM format trajectory file"""
    try:
        return file_interface.read_tum_trajectory_file(str(filepath))
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def compute_trajectory_length(traj: trajectory.PoseTrajectory3D) -> float:
    """Compute total trajectory length in meters"""
    positions = traj.positions_xyz
    length = 0.0
    for i in range(1, len(positions)):
        diff = positions[i] - positions[i-1]
        length += np.linalg.norm(diff)
    return length


def compute_ate(est_traj: trajectory.PoseTrajectory3D, 
                ref_traj: trajectory.PoseTrajectory3D,
                align: bool = True) -> Tuple[float, float, float]:
    """
    Compute Absolute Trajectory Error
    Returns: (ATE in meters, trajectory length, ATE as percentage)
    """
    # Synchronize trajectories by timestamp
    ref_synced, est_synced = sync.associate_trajectories(ref_traj, est_traj)
    
    # Align trajectories (SE3 alignment)
    if align:
        est_synced.align(ref_synced, correct_scale=False)
    
    # Compute trajectory length from reference
    traj_length = compute_trajectory_length(ref_synced)
    
    # Compute ATE - we want final position error (drift)
    # Get final positions
    ref_final = ref_synced.positions_xyz[-1]
    est_final = est_synced.positions_xyz[-1]
    
    # ATE = Euclidean distance at final position
    ate = np.linalg.norm(ref_final - est_final)
    ate_percent = (ate / traj_length) * 100 if traj_length > 0 else 0
    
    return ate, traj_length, ate_percent


def evaluate_algorithm(algorithm: str, 
                       floors: List[str],
                       results_dir: Path,
                       gt_dir: Path) -> List[TrajectoryResult]:
    """Evaluate an algorithm across all floors"""
    results = []
    
    algo_dir = results_dir / "trajectories" / algorithm
    
    for floor in floors:
        print(f"\n  Evaluating {algorithm} on {floor}...")
        
        # Load estimated trajectory
        est_file = algo_dir / f"{floor}.txt"
        if not est_file.exists():
            results.append(TrajectoryResult(
                floor=floor,
                algorithm=algorithm,
                ate_m=0.0,
                ate_percent=0.0,
                trajectory_length=0.0,
                pose_count=0,
                success=False,
                error_msg=f"File not found: {est_file}"
            ))
            print(f"    SKIP: Trajectory file not found")
            continue
        
        est_traj = load_trajectory(est_file)
        if est_traj is None:
            results.append(TrajectoryResult(
                floor=floor,
                algorithm=algorithm,
                ate_m=0.0,
                ate_percent=0.0,
                trajectory_length=0.0,
                pose_count=0,
                success=False,
                error_msg="Failed to load trajectory"
            ))
            continue
        
        # Load ground truth (LeGO-LOAM)
        gt_file = gt_dir / f"{floor}.txt"
        if not gt_file.exists():
            print(f"    SKIP: Ground truth not found: {gt_file}")
            results.append(TrajectoryResult(
                floor=floor,
                algorithm=algorithm,
                ate_m=0.0,
                ate_percent=0.0,
                trajectory_length=0.0,
                pose_count=len(est_traj.timestamps),
                success=False,
                error_msg=f"Ground truth not found: {gt_file}"
            ))
            continue
        
        ref_traj = load_trajectory(gt_file)
        if ref_traj is None:
            results.append(TrajectoryResult(
                floor=floor,
                algorithm=algorithm,
                ate_m=0.0,
                ate_percent=0.0,
                trajectory_length=0.0,
                pose_count=len(est_traj.timestamps),
                success=False,
                error_msg="Failed to load ground truth"
            ))
            continue
        
        try:
            ate, traj_length, ate_percent = compute_ate(est_traj, ref_traj)
            
            results.append(TrajectoryResult(
                floor=floor,
                algorithm=algorithm,
                ate_m=ate,
                ate_percent=ate_percent,
                trajectory_length=traj_length,
                pose_count=len(est_traj.timestamps),
                success=True
            ))
            
            print(f"    ATE: {ate:.3f}m ({ate_percent:.2f}%)")
            print(f"    Trajectory length: {traj_length:.1f}m")
            print(f"    Poses: {len(est_traj.timestamps)}")
            
        except Exception as e:
            results.append(TrajectoryResult(
                floor=floor,
                algorithm=algorithm,
                ate_m=0.0,
                ate_percent=0.0,
                trajectory_length=0.0,
                pose_count=len(est_traj.timestamps),
                success=False,
                error_msg=str(e)
            ))
            print(f"    ERROR: {e}")
    
    return results


def generate_table(results: List[TrajectoryResult], 
                   paper_results: Dict[str, Dict[str, Tuple[float, float]]]) -> str:
    """Generate comparison table (markdown format)"""
    
    # Group by algorithm
    by_algo = {}
    for r in results:
        if r.algorithm not in by_algo:
            by_algo[r.algorithm] = {}
        by_algo[r.algorithm][r.floor] = r
    
    floors = ["5th_floor", "1st_floor", "4th_floor", "2nd_floor"]
    
    lines = []
    lines.append("# SLAM Algorithm Comparison - ISEC Dataset")
    lines.append("")
    lines.append("## Results vs Paper (Table IV)")
    lines.append("")
    lines.append("| Algorithm | Floor | Our ATE(m) | Our % | Paper ATE(m) | Paper % |")
    lines.append("|-----------|-------|------------|-------|--------------|---------|")
    
    for algo, algo_results in sorted(by_algo.items()):
        for floor in floors:
            r = algo_results.get(floor)
            paper = paper_results.get(algo, {}).get(floor, (None, None))
            
            if r and r.success:
                our_ate = f"{r.ate_m:.3f}"
                our_pct = f"{r.ate_percent:.2f}%"
            else:
                our_ate = "-"
                our_pct = "-"
            
            if paper[0] is not None:
                paper_ate = f"{paper[0]:.3f}"
                paper_pct = f"{paper[1]:.2f}%"
            else:
                paper_ate = "-"
                paper_pct = "-"
            
            lines.append(f"| {algo} | {floor} | {our_ate} | {our_pct} | {paper_ate} | {paper_pct} |")
    
    return "\n".join(lines)


def main():
    results_dir = Path("/results")
    gt_dir = results_dir / "trajectories" / "lego_loam"
    
    floors = ["5th_floor", "1st_floor", "4th_floor", "2nd_floor"]
    
    # Paper results for comparison (Table IV)
    paper_results = {
        "orb_slam3": {
            "5th_floor": (0.516, 0.28),
            "1st_floor": (0.949, 1.46),
            "4th_floor": (0.483, 0.73),
            "2nd_floor": (0.310, 0.24),
        },
        "lego_loam": {
            "5th_floor": (0.395, 0.21),
            "1st_floor": (0.256, 0.39),
            "4th_floor": (0.789, 1.20),
            "2nd_floor": (0.286, 0.22),
        }
    }
    
    print("=" * 60)
    print("SLAM Trajectory Evaluation")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Ground truth: {gt_dir}")
    
    all_results = []
    
    # Find all algorithm directories
    traj_dir = results_dir / "trajectories"
    if traj_dir.exists():
        for algo_dir in sorted(traj_dir.iterdir()):
            if algo_dir.is_dir() and algo_dir.name != "lego_loam":
                print(f"\n{'='*60}")
                print(f"Evaluating: {algo_dir.name}")
                print("=" * 60)
                
                algo_results = evaluate_algorithm(
                    algo_dir.name, floors, results_dir, gt_dir
                )
                all_results.extend(algo_results)
    
    # Generate and save results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    table = generate_table(all_results, paper_results)
    print(table)
    
    # Save to file
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    with open(metrics_dir / "comparison_table.md", "w") as f:
        f.write(table)
    
    # Save detailed JSON
    json_results = []
    for r in all_results:
        json_results.append({
            "floor": r.floor,
            "algorithm": r.algorithm,
            "ate_m": r.ate_m,
            "ate_percent": r.ate_percent,
            "trajectory_length": r.trajectory_length,
            "pose_count": r.pose_count,
            "success": r.success,
            "error_msg": r.error_msg
        })
    
    with open(metrics_dir / "detailed_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  {metrics_dir / 'comparison_table.md'}")
    print(f"  {metrics_dir / 'detailed_results.json'}")


if __name__ == "__main__":
    main()
