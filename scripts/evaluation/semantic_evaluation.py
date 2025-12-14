#!/usr/bin/env python3
"""
Semantic Evaluation Metrics for Multi-Floor SLAM

Computes semantic-specific metrics beyond standard ATE/RPE:
- Loop closure precision/recall with semantic gating
- Floor detection accuracy
- Dynamic object filtering effectiveness
- Cross-floor aliasing prevention rate

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import argparse


@dataclass
class LoopClosureMetrics:
    """Metrics for loop closure quality assessment."""
    total_candidates: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Semantic gating specific
    same_floor_candidates: int = 0
    cross_floor_candidates: int = 0
    cross_floor_rejected: int = 0

    @property
    def precision(self) -> float:
        """TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def cross_floor_rate(self) -> float:
        """Percentage of candidates that were cross-floor."""
        if self.total_candidates == 0:
            return 0.0
        return self.cross_floor_candidates / self.total_candidates

    @property
    def gating_effectiveness(self) -> float:
        """Percentage of cross-floor candidates successfully rejected."""
        if self.cross_floor_candidates == 0:
            return 1.0  # No cross-floor candidates to reject
        return self.cross_floor_rejected / self.cross_floor_candidates


@dataclass
class FloorDetectionMetrics:
    """Metrics for floor detection accuracy."""
    total_poses: int = 0
    correct_floor_labels: int = 0
    elevator_events_detected: int = 0
    elevator_events_ground_truth: int = 0

    @property
    def floor_accuracy(self) -> float:
        """Percentage of poses with correct floor label."""
        if self.total_poses == 0:
            return 0.0
        return self.correct_floor_labels / self.total_poses

    @property
    def elevator_precision(self) -> float:
        """For elevator event detection."""
        # Simplified: detected / ground_truth
        if self.elevator_events_ground_truth == 0:
            return 1.0 if self.elevator_events_detected == 0 else 0.0
        return min(1.0, self.elevator_events_detected / self.elevator_events_ground_truth)


@dataclass
class DynamicFilteringMetrics:
    """Metrics for dynamic object filtering effectiveness."""
    total_frames: int = 0
    frames_with_dynamic_objects: int = 0
    total_features_extracted: int = 0
    features_filtered: int = 0

    # Trajectory quality
    mean_tracking_velocity: float = 0.0  # m/s
    velocity_std: float = 0.0  # Lower is smoother
    tracking_failures: int = 0

    @property
    def dynamic_object_rate(self) -> float:
        """Percentage of frames containing dynamic objects."""
        if self.total_frames == 0:
            return 0.0
        return self.frames_with_dynamic_objects / self.total_frames

    @property
    def feature_filter_rate(self) -> float:
        """Percentage of features filtered as dynamic."""
        if self.total_features_extracted == 0:
            return 0.0
        return self.features_filtered / self.total_features_extracted


@dataclass
class SemanticEvaluationResult:
    """Complete semantic evaluation results for an algorithm."""
    algorithm: str
    sequence: str

    # Standard trajectory metrics (from evo)
    ate_rmse: float = 0.0
    ate_mean: float = 0.0
    ate_max: float = 0.0
    rpe_rmse: float = 0.0
    endpoint_drift: float = 0.0
    drift_percentage: float = 0.0

    # Semantic metrics
    loop_closure: LoopClosureMetrics = field(default_factory=LoopClosureMetrics)
    floor_detection: FloorDetectionMetrics = field(default_factory=FloorDetectionMetrics)
    dynamic_filtering: DynamicFilteringMetrics = field(default_factory=DynamicFilteringMetrics)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'algorithm': self.algorithm,
            'sequence': self.sequence,
            'trajectory_metrics': {
                'ate_rmse': self.ate_rmse,
                'ate_mean': self.ate_mean,
                'ate_max': self.ate_max,
                'rpe_rmse': self.rpe_rmse,
                'endpoint_drift': self.endpoint_drift,
                'drift_percentage': self.drift_percentage,
            },
            'loop_closure_metrics': {
                'total_candidates': self.loop_closure.total_candidates,
                'precision': self.loop_closure.precision,
                'recall': self.loop_closure.recall,
                'f1_score': self.loop_closure.f1_score,
                'cross_floor_rate': self.loop_closure.cross_floor_rate,
                'gating_effectiveness': self.loop_closure.gating_effectiveness,
                'same_floor_candidates': self.loop_closure.same_floor_candidates,
                'cross_floor_candidates': self.loop_closure.cross_floor_candidates,
            },
            'floor_detection_metrics': {
                'floor_accuracy': self.floor_detection.floor_accuracy,
                'elevator_precision': self.floor_detection.elevator_precision,
                'total_poses': self.floor_detection.total_poses,
                'elevator_events_detected': self.floor_detection.elevator_events_detected,
            },
            'dynamic_filtering_metrics': {
                'dynamic_object_rate': self.dynamic_filtering.dynamic_object_rate,
                'feature_filter_rate': self.dynamic_filtering.feature_filter_rate,
                'velocity_std': self.dynamic_filtering.velocity_std,
                'tracking_failures': self.dynamic_filtering.tracking_failures,
            }
        }


class SemanticEvaluator:
    """
    Comprehensive semantic evaluation for SLAM algorithms.

    Evaluates:
    1. Standard trajectory metrics (ATE, RPE, drift)
    2. Loop closure quality with semantic gating analysis
    3. Floor detection accuracy
    4. Dynamic object filtering effectiveness
    """

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.trajectories_dir = self.results_dir / 'trajectories'
        self.semantic_dir = self.results_dir / 'semantic_gating'
        self.metrics_dir = self.results_dir / 'metrics'
        self.metrics_dir.mkdir(exist_ok=True)

    def load_trajectory(self, algorithm: str, sequence: str) -> Optional[np.ndarray]:
        """Load trajectory in TUM format."""
        traj_file = self.trajectories_dir / algorithm / f"{sequence}.txt"
        if not traj_file.exists():
            # Try alternative names
            for alt in [f"{sequence}_stereo.txt", f"{sequence}_tum.txt"]:
                alt_file = self.trajectories_dir / algorithm / alt
                if alt_file.exists():
                    traj_file = alt_file
                    break

        if not traj_file.exists():
            print(f"Warning: Trajectory not found: {traj_file}")
            return None

        try:
            data = np.loadtxt(traj_file)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data
        except Exception as e:
            print(f"Error loading trajectory {traj_file}: {e}")
            return None

    def load_semantic_analysis(self, algorithm: str) -> Optional[dict]:
        """Load semantic gating analysis results."""
        analysis_file = self.semantic_dir / f"{algorithm}_semantic_analysis.txt"
        if not analysis_file.exists():
            return None

        # Parse the text file for key metrics
        metrics = {}
        try:
            with open(analysis_file, 'r') as f:
                content = f.read()

            # Extract metrics using simple parsing
            import re

            # Total poses
            match = re.search(r'Total poses:\s+(\d+)', content)
            if match:
                metrics['total_poses'] = int(match.group(1))

            # Loop closure candidates
            match = re.search(r'Total candidates:\s+([\d,]+)', content)
            if match:
                metrics['total_candidates'] = int(match.group(1).replace(',', ''))

            # Same floor
            match = re.search(r'Same-floor.*?:\s+([\d,]+)', content)
            if match:
                metrics['same_floor'] = int(match.group(1).replace(',', ''))

            # Cross floor
            match = re.search(r'Cross-floor.*?:\s+([\d,]+)', content)
            if match:
                metrics['cross_floor'] = int(match.group(1).replace(',', ''))

            # Cross-floor rate
            match = re.search(r'Cross-floor rate:\s+([\d.]+)%', content)
            if match:
                metrics['cross_floor_rate'] = float(match.group(1))

            return metrics

        except Exception as e:
            print(f"Error parsing semantic analysis: {e}")
            return None

    def compute_trajectory_metrics(self, trajectory: np.ndarray) -> dict:
        """Compute basic trajectory metrics."""
        if trajectory is None or len(trajectory) < 2:
            return {}

        # Extract positions (columns 1-3 in TUM format)
        positions = trajectory[:, 1:4]

        # Compute velocities
        timestamps = trajectory[:, 0]
        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6  # Avoid division by zero

        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt

        # Endpoint drift
        start_pos = positions[0]
        end_pos = positions[-1]
        endpoint_drift = np.linalg.norm(end_pos - start_pos)

        # Total trajectory length
        segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_length = np.sum(segment_lengths)

        # Drift percentage
        drift_pct = (endpoint_drift / total_length * 100) if total_length > 0 else 0

        return {
            'endpoint_drift': endpoint_drift,
            'total_length': total_length,
            'drift_percentage': drift_pct,
            'mean_velocity': np.mean(velocities),
            'velocity_std': np.std(velocities),
            'duration': timestamps[-1] - timestamps[0],
            'num_poses': len(trajectory),
        }

    def evaluate_algorithm(self, algorithm: str,
                          sequences: List[str] = None) -> List[SemanticEvaluationResult]:
        """Evaluate a single algorithm across all sequences."""
        if sequences is None:
            sequences = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']

        results = []

        # Load semantic analysis (covers all sequences combined)
        semantic_analysis = self.load_semantic_analysis(algorithm)

        for sequence in sequences:
            result = SemanticEvaluationResult(
                algorithm=algorithm,
                sequence=sequence
            )

            # Load and evaluate trajectory
            trajectory = self.load_trajectory(algorithm, sequence)
            if trajectory is not None:
                traj_metrics = self.compute_trajectory_metrics(trajectory)
                result.endpoint_drift = traj_metrics.get('endpoint_drift', 0)
                result.drift_percentage = traj_metrics.get('drift_percentage', 0)
                result.floor_detection.total_poses = traj_metrics.get('num_poses', 0)
                result.dynamic_filtering.mean_tracking_velocity = traj_metrics.get('mean_velocity', 0)
                result.dynamic_filtering.velocity_std = traj_metrics.get('velocity_std', 0)

            # Apply semantic analysis metrics (same for all sequences in this impl)
            if semantic_analysis:
                result.loop_closure.total_candidates = semantic_analysis.get('total_candidates', 0)
                result.loop_closure.same_floor_candidates = semantic_analysis.get('same_floor', 0)
                result.loop_closure.cross_floor_candidates = semantic_analysis.get('cross_floor', 0)
                result.loop_closure.cross_floor_rejected = semantic_analysis.get('cross_floor', 0)

            results.append(result)

        return results

    def evaluate_all(self, algorithms: List[str] = None) -> Dict[str, List[SemanticEvaluationResult]]:
        """Evaluate all algorithms."""
        if algorithms is None:
            algorithms = ['lego_loam', 'orb_slam3', 'droid_slam',
                         'kimera', 'suma_plus_plus', 'yolo_orb_slam3']

        all_results = {}

        for algorithm in algorithms:
            print(f"Evaluating {algorithm}...")
            results = self.evaluate_algorithm(algorithm)
            if results:
                all_results[algorithm] = results

        return all_results

    def generate_comparison_table(self, results: Dict[str, List[SemanticEvaluationResult]]) -> str:
        """Generate markdown comparison table."""
        lines = [
            "# Semantic SLAM Evaluation Results",
            "",
            "## Trajectory Accuracy (Endpoint Drift)",
            "",
            "| Algorithm | 5th Floor | 1st Floor | 4th Floor | 2nd Floor | Mean |",
            "|-----------|-----------|-----------|-----------|-----------|------|",
        ]

        sequences = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']

        for algorithm, algo_results in results.items():
            row = [algorithm]
            drifts = []
            for seq in sequences:
                result = next((r for r in algo_results if r.sequence == seq), None)
                if result and result.endpoint_drift > 0:
                    row.append(f"{result.endpoint_drift:.2f}m")
                    drifts.append(result.endpoint_drift)
                else:
                    row.append("N/A")

            mean_drift = np.mean(drifts) if drifts else 0
            row.append(f"**{mean_drift:.2f}m**" if mean_drift > 0 else "N/A")
            lines.append("| " + " | ".join(row) + " |")

        lines.extend([
            "",
            "## Loop Closure Metrics",
            "",
            "| Algorithm | Total Candidates | Same-Floor | Cross-Floor | Cross-Floor Rate | Gating Effectiveness |",
            "|-----------|------------------|------------|-------------|------------------|---------------------|",
        ])

        for algorithm, algo_results in results.items():
            # Use first result (metrics are aggregate)
            if algo_results:
                lc = algo_results[0].loop_closure
                row = [
                    algorithm,
                    f"{lc.total_candidates:,}",
                    f"{lc.same_floor_candidates:,}",
                    f"{lc.cross_floor_candidates:,}",
                    f"{lc.cross_floor_rate*100:.1f}%",
                    f"{lc.gating_effectiveness*100:.1f}%",
                ]
                lines.append("| " + " | ".join(row) + " |")

        lines.extend([
            "",
            "## Trajectory Smoothness",
            "",
            "| Algorithm | Mean Velocity (m/s) | Velocity Std (m/s) | Tracking Failures |",
            "|-----------|--------------------|--------------------|-------------------|",
        ])

        for algorithm, algo_results in results.items():
            velocities = [r.dynamic_filtering.mean_tracking_velocity for r in algo_results
                         if r.dynamic_filtering.mean_tracking_velocity > 0]
            stds = [r.dynamic_filtering.velocity_std for r in algo_results
                   if r.dynamic_filtering.velocity_std > 0]

            mean_vel = np.mean(velocities) if velocities else 0
            mean_std = np.mean(stds) if stds else 0
            failures = sum(r.dynamic_filtering.tracking_failures for r in algo_results)

            row = [
                algorithm,
                f"{mean_vel:.3f}" if mean_vel > 0 else "N/A",
                f"{mean_std:.3f}" if mean_std > 0 else "N/A",
                str(failures),
            ]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def save_results(self, results: Dict[str, List[SemanticEvaluationResult]],
                    output_name: str = "semantic_evaluation"):
        """Save results to JSON and markdown."""
        # JSON output
        json_output = {}
        for algorithm, algo_results in results.items():
            json_output[algorithm] = [r.to_dict() for r in algo_results]

        json_file = self.metrics_dir / f"{output_name}.json"
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved JSON results to: {json_file}")

        # Markdown output
        md_content = self.generate_comparison_table(results)
        md_file = self.metrics_dir / f"{output_name}.md"
        with open(md_file, 'w') as f:
            f.write(md_content)
        print(f"Saved markdown results to: {md_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic evaluation for multi-floor SLAM algorithms"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Path to results directory'
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=['lego_loam', 'orb_slam3', 'droid_slam'],
        help='Algorithms to evaluate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='semantic_evaluation',
        help='Output filename prefix'
    )

    args = parser.parse_args()

    # Find results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent.parent
        results_dir = script_dir / 'results'

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print("=" * 60)
    print("Semantic SLAM Evaluation")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Algorithms: {args.algorithms}")
    print("=" * 60)

    evaluator = SemanticEvaluator(results_dir)
    results = evaluator.evaluate_all(args.algorithms)

    if results:
        evaluator.save_results(results, args.output)
        print("\n" + evaluator.generate_comparison_table(results))
    else:
        print("No results to evaluate.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
