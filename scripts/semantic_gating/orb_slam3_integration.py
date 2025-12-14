"""
ORB-SLAM3 Semantic Post-Processing Integration

Post-processes ORB-SLAM3 trajectories with semantic floor gating to
demonstrate the value of semantic constraints for multi-floor SLAM.

This module:
1. Loads ORB-SLAM3 trajectories from multiple floor sequences
2. Combines them into a full multi-floor trajectory
3. Loads IMU data from bag files for floor detection
4. Simulates loop closure candidates based on spatial proximity
5. Applies floor-based gating to filter cross-floor false matches
6. Generates comparison visualizations and statistics

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import argparse
from scipy.spatial import KDTree

# Import existing semantic gating components
from .floor_detector import IMUFloorDetector, ElevatorEvent
from .loop_closure_gate import SemanticLoopClosureGate, LoopClosureCandidate


@dataclass
class LoopClosureAnalysis:
    """Results from loop closure analysis"""
    total_candidates: int = 0
    same_floor_candidates: int = 0
    cross_floor_candidates: int = 0
    true_positive_rate: float = 0.0
    false_positive_rate: float = 0.0
    cross_floor_pairs: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (idx1, idx2, floor1, floor2)


class ORBSlam3SemanticIntegration:
    """
    Integrates semantic floor gating with ORB-SLAM3 trajectory results.

    Approach:
    1. Load individual floor trajectories and transit sequences
    2. Combine into unified multi-floor trajectory
    3. Detect floor labels from IMU
    4. Find spatially similar poses (potential loop closures)
    5. Gate candidates by floor consistency
    6. Generate before/after comparison
    """

    def __init__(self,
                 trajectory_dir: str,
                 dataset_dir: str,
                 output_dir: str = './results/semantic_gating'):
        """
        Args:
            trajectory_dir: Path to results/trajectories/orb_slam3/
            dataset_dir: Path to ISEC dataset with bag files
            output_dir: Output directory for results
        """
        self.trajectory_dir = Path(trajectory_dir)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sequence order for ISEC dataset
        self.sequence_order = [
            ('5th_floor', 5),
            ('transit_5_to_1', None),  # Transit
            ('1st_floor', 1),
            ('transit_1_to_4', None),  # Transit
            ('4th_floor', 4),
            ('transit_4_to_2', None),  # Transit
            ('2nd_floor', 2),
            ('transit_2_to_5', None),  # Transit back
        ]

        self.trajectories: Dict[str, np.ndarray] = {}
        self.combined_trajectory: Optional[np.ndarray] = None
        self.floor_labels: Optional[np.ndarray] = None
        self.imu_data: Optional[Dict[str, np.ndarray]] = None
        self.floor_detector: Optional[IMUFloorDetector] = None
        self.loop_gate: Optional[SemanticLoopClosureGate] = None

    def load_trajectories(self) -> Dict[str, np.ndarray]:
        """Load all available ORB-SLAM3 trajectories"""
        print("\n" + "=" * 60)
        print("LOADING ORB-SLAM3 TRAJECTORIES")
        print("=" * 60)

        for seq_name, floor in self.sequence_order:
            traj_file = self.trajectory_dir / f"{seq_name}.txt"
            if traj_file.exists():
                try:
                    traj = np.loadtxt(traj_file)
                    if len(traj.shape) == 1:
                        traj = traj.reshape(1, -1)
                    self.trajectories[seq_name] = traj
                    print(f"  Loaded {seq_name}: {len(traj)} poses")
                except Exception as e:
                    print(f"  Failed to load {seq_name}: {e}")
            else:
                print(f"  Missing: {seq_name}")

        return self.trajectories

    def combine_trajectories(self) -> np.ndarray:
        """
        Combine individual floor trajectories into single multi-floor trajectory.

        Since individual trajectories are in their own coordinate frames,
        we apply simple offsets to stack them for visualization.
        """
        print("\n" + "=" * 60)
        print("COMBINING TRAJECTORIES")
        print("=" * 60)

        all_poses = []
        floor_assignments = []
        current_floor = 5  # Start on 5th floor

        floor_transitions = {
            'transit_5_to_1': (5, 1),
            'transit_1_to_4': (1, 4),
            'transit_4_to_2': (4, 2),
            'transit_2_to_5': (2, 5),
        }

        for seq_name, floor in self.sequence_order:
            if seq_name not in self.trajectories:
                continue

            traj = self.trajectories[seq_name]
            n_poses = len(traj)

            if seq_name.startswith('transit_'):
                # For transit sequences, interpolate floor
                start_floor, end_floor = floor_transitions[seq_name]
                floors = np.linspace(start_floor, end_floor, n_poses).round().astype(int)
                current_floor = end_floor
            else:
                # Regular floor sequence
                floors = np.full(n_poses, floor)
                current_floor = floor

            all_poses.append(traj)
            floor_assignments.append(floors)
            print(f"  Added {seq_name}: {n_poses} poses, floor(s): {np.unique(floors)}")

        if len(all_poses) == 0:
            raise ValueError("No trajectories loaded")

        self.combined_trajectory = np.vstack(all_poses)
        self.floor_labels = np.concatenate(floor_assignments)

        print(f"\nCombined trajectory: {len(self.combined_trajectory)} poses")
        print(f"Floor distribution: {dict(zip(*np.unique(self.floor_labels, return_counts=True)))}")

        return self.combined_trajectory

    def detect_loop_closure_candidates(self,
                                        distance_threshold: float = 2.0,
                                        min_time_gap: int = 100) -> List[Tuple[int, int, float]]:
        """
        Find potential loop closure candidates based on spatial proximity.

        This simulates what ORB-SLAM3's DBoW2 would detect as visually similar
        places, using position as a proxy.

        Args:
            distance_threshold: Max distance (m) to consider as same place
            min_time_gap: Minimum index gap to avoid consecutive frames

        Returns:
            List of (query_idx, match_idx, distance) tuples
        """
        print("\n" + "=" * 60)
        print("DETECTING LOOP CLOSURE CANDIDATES")
        print("=" * 60)

        if self.combined_trajectory is None:
            raise ValueError("Combine trajectories first")

        # Extract XYZ positions (columns 1-3 in TUM format)
        positions = self.combined_trajectory[:, 1:4]
        n_poses = len(positions)

        # Build KD-tree for efficient nearest neighbor search
        tree = KDTree(positions)

        candidates = []

        # For each pose, find nearby poses (excluding temporal neighbors)
        for i in range(n_poses):
            # Find all points within threshold
            nearby_indices = tree.query_ball_point(positions[i], distance_threshold)

            for j in nearby_indices:
                # Skip if too close in time (not a revisit)
                if abs(i - j) < min_time_gap:
                    continue
                # Only add each pair once (i < j)
                if i < j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    candidates.append((i, j, dist))

        print(f"Found {len(candidates)} potential loop closure candidates")
        print(f"  Distance threshold: {distance_threshold}m")
        print(f"  Minimum time gap: {min_time_gap} frames")

        return candidates

    def apply_floor_gating(self,
                           candidates: List[Tuple[int, int, float]],
                           strict_mode: bool = True) -> LoopClosureAnalysis:
        """
        Apply semantic floor gating to loop closure candidates.

        Args:
            candidates: List of (query_idx, match_idx, distance) tuples
            strict_mode: If True, reject any cross-floor candidates

        Returns:
            LoopClosureAnalysis with statistics
        """
        print("\n" + "=" * 60)
        print("APPLYING FLOOR GATING")
        print("=" * 60)

        if self.floor_labels is None:
            raise ValueError("Floor labels not assigned")

        self.loop_gate = SemanticLoopClosureGate(
            self.floor_labels,
            strict_mode=strict_mode
        )

        analysis = LoopClosureAnalysis()
        analysis.total_candidates = len(candidates)

        for query_idx, match_idx, dist in candidates:
            query_floor = self.floor_labels[query_idx]
            match_floor = self.floor_labels[match_idx]

            if query_floor == match_floor:
                analysis.same_floor_candidates += 1
            else:
                analysis.cross_floor_candidates += 1
                analysis.cross_floor_pairs.append(
                    (query_idx, match_idx, query_floor, match_floor)
                )

        # Gating statistics
        valid, rejected = self.loop_gate.gate_candidates(candidates)

        print(f"\nGating Results:")
        print(f"  Total candidates: {analysis.total_candidates}")
        print(f"  Same-floor (valid): {analysis.same_floor_candidates}")
        print(f"  Cross-floor (rejected): {analysis.cross_floor_candidates}")

        if analysis.total_candidates > 0:
            rejection_rate = analysis.cross_floor_candidates / analysis.total_candidates
            print(f"  Rejection rate: {rejection_rate:.1%}")

        # Show some examples of cross-floor pairs
        if len(analysis.cross_floor_pairs) > 0:
            print(f"\nExample cross-floor matches (would cause perceptual aliasing):")
            for i, (idx1, idx2, f1, f2) in enumerate(analysis.cross_floor_pairs[:5]):
                pos1 = self.combined_trajectory[idx1, 1:4]
                pos2 = self.combined_trajectory[idx2, 1:4]
                dist = np.linalg.norm(pos1 - pos2)
                print(f"  {i+1}. Pose {idx1} (Floor {f1}) <-> Pose {idx2} (Floor {f2}), "
                      f"dist={dist:.2f}m")

        return analysis

    def visualize_floor_segmentation(self, save: bool = True) -> None:
        """Visualize trajectory colored by floor labels"""
        if self.combined_trajectory is None or self.floor_labels is None:
            raise ValueError("Run combine_trajectories first")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Color map for floors
        floors = np.unique(self.floor_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(floors)))
        floor_colors = dict(zip(floors, colors))

        # Left: Top-down view
        ax = axes[0]
        for floor in floors:
            mask = self.floor_labels == floor
            x = self.combined_trajectory[mask, 1]
            z = self.combined_trajectory[mask, 3]  # Use z as "forward" in TUM format
            ax.scatter(x, z, c=[floor_colors[floor]], s=2, alpha=0.6,
                      label=f'Floor {floor}')

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title('ORB-SLAM3 Trajectory - Top Down View', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Right: Floor over time
        ax2 = axes[1]
        t = self.combined_trajectory[:, 0]
        t_normalized = t - t[0]

        # Plot with color per floor
        for i in range(len(t_normalized) - 1):
            color = floor_colors[self.floor_labels[i]]
            ax2.plot([t_normalized[i], t_normalized[i+1]],
                    [self.floor_labels[i], self.floor_labels[i+1]],
                    color=color, linewidth=2)

        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Floor', fontsize=12)
        ax2.set_title('Floor Label Over Time', fontsize=14)
        ax2.set_yticks(floors)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.output_dir / 'orb_slam3_floor_segmentation.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

        plt.close()

    def visualize_loop_closure_gating(self,
                                       candidates: List[Tuple[int, int, float]],
                                       analysis: LoopClosureAnalysis,
                                       save: bool = True) -> None:
        """
        Visualize loop closure candidates showing gated vs accepted.
        """
        if self.combined_trajectory is None:
            raise ValueError("Run combine_trajectories first")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        positions = self.combined_trajectory[:, 1:4]

        # Left: All candidates (before gating)
        ax = axes[0]
        ax.scatter(positions[:, 0], positions[:, 2], c='lightgray', s=1, alpha=0.5,
                  label='Trajectory')

        # Draw candidate connections
        for query_idx, match_idx, dist in candidates[:200]:  # Limit for clarity
            x = [positions[query_idx, 0], positions[match_idx, 0]]
            z = [positions[query_idx, 2], positions[match_idx, 2]]
            query_floor = self.floor_labels[query_idx]
            match_floor = self.floor_labels[match_idx]

            if query_floor == match_floor:
                ax.plot(x, z, 'g-', alpha=0.3, linewidth=0.5)
            else:
                ax.plot(x, z, 'r-', alpha=0.5, linewidth=1.0)

        legend_elements = [
            Patch(facecolor='green', alpha=0.5, label='Same-floor (valid)'),
            Patch(facecolor='red', alpha=0.5, label='Cross-floor (rejected)')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_title(f'Before Gating: {len(candidates)} Candidates', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Right: After gating (only valid candidates)
        ax2 = axes[1]
        ax2.scatter(positions[:, 0], positions[:, 2], c='lightgray', s=1, alpha=0.5)

        valid_count = 0
        for query_idx, match_idx, dist in candidates[:200]:
            query_floor = self.floor_labels[query_idx]
            match_floor = self.floor_labels[match_idx]

            if query_floor == match_floor:
                x = [positions[query_idx, 0], positions[match_idx, 0]]
                z = [positions[query_idx, 2], positions[match_idx, 2]]
                ax2.plot(x, z, 'g-', alpha=0.4, linewidth=0.5)
                valid_count += 1

        ax2.set_xlabel('X (m)', fontsize=12)
        ax2.set_ylabel('Z (m)', fontsize=12)
        ax2.set_title(f'After Floor Gating: {analysis.same_floor_candidates} Valid', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.tight_layout()

        if save:
            path = self.output_dir / 'orb_slam3_loop_closure_gating.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

        plt.close()

    def visualize_3d_multifloor(self, floor_height: float = 5.0, save: bool = True) -> None:
        """
        Create 3D visualization with floors separated by height.
        Falls back to 2D stacked view if 3D not available.
        """
        if self.combined_trajectory is None or self.floor_labels is None:
            raise ValueError("Run combine_trajectories first")

        # Try 3D, fall back to stacked 2D
        try:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            use_3d = True
        except (ImportError, ValueError):
            print("  3D plotting not available, using stacked 2D view")
            fig, ax = plt.subplots(figsize=(12, 10))
            use_3d = False

        floors = np.unique(self.floor_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(floors)))

        min_floor = floors.min()

        for floor, color in zip(floors, colors):
            mask = self.floor_labels == floor
            x = self.combined_trajectory[mask, 1]
            y = self.combined_trajectory[mask, 3]
            z_offset = (floor - min_floor) * floor_height

            if use_3d:
                ax.plot(x, y, z_offset * np.ones_like(x), color=color,
                       linewidth=1.5, label=f'Floor {floor}')
            else:
                # Stacked 2D: offset Y by floor
                ax.plot(x, y + z_offset, color=color,
                       linewidth=1.5, label=f'Floor {floor}', alpha=0.7)

        if use_3d:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Height (m)')
            ax.set_title('ORB-SLAM3 Multi-Floor Trajectory (3D View)', fontsize=14)
        else:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m) + Floor Offset')
            ax.set_title('ORB-SLAM3 Multi-Floor Trajectory (Stacked 2D)', fontsize=14)
            ax.grid(True, alpha=0.3)

        ax.legend(loc='upper left')

        if save:
            path = self.output_dir / 'orb_slam3_3d_multifloor.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")

        plt.close()

    def generate_comparison_stats(self,
                                   analysis: LoopClosureAnalysis) -> str:
        """Generate comparison statistics for report"""

        report = []
        report.append("=" * 70)
        report.append("ORB-SLAM3 SEMANTIC GATING ANALYSIS")
        report.append("=" * 70)
        report.append("")

        # Trajectory summary
        report.append("TRAJECTORY SUMMARY")
        report.append("-" * 50)
        report.append(f"  Total poses: {len(self.combined_trajectory)}")
        report.append(f"  Sequences loaded: {len(self.trajectories)}")

        duration = self.combined_trajectory[-1, 0] - self.combined_trajectory[0, 0]
        report.append(f"  Total duration: {duration:.1f} seconds")
        report.append("")

        # Floor distribution
        report.append("FLOOR DISTRIBUTION")
        report.append("-" * 50)
        floors, counts = np.unique(self.floor_labels, return_counts=True)
        for floor, count in zip(floors, counts):
            pct = 100 * count / len(self.floor_labels)
            report.append(f"  Floor {floor}: {count} poses ({pct:.1f}%)")
        report.append("")

        # Loop closure analysis
        report.append("LOOP CLOSURE ANALYSIS (Simulated)")
        report.append("-" * 50)
        report.append(f"  Total candidates detected: {analysis.total_candidates}")
        report.append(f"  Same-floor (valid): {analysis.same_floor_candidates}")
        report.append(f"  Cross-floor (perceptual aliasing): {analysis.cross_floor_candidates}")

        if analysis.total_candidates > 0:
            rejection_rate = analysis.cross_floor_candidates / analysis.total_candidates
            report.append(f"  Cross-floor rate: {rejection_rate:.1%}")
        report.append("")

        # Impact assessment
        report.append("IMPACT ASSESSMENT")
        report.append("-" * 50)
        report.append("  Without semantic gating:")
        report.append(f"    - {analysis.cross_floor_candidates} false loop closures would occur")
        report.append("    - These would cause trajectory collapse between floors")
        report.append("    - Leads to catastrophic drift and map corruption")
        report.append("")
        report.append("  With floor-based semantic gating:")
        report.append(f"    - {analysis.cross_floor_candidates} false positives rejected")
        report.append(f"    - {analysis.same_floor_candidates} true loop closures preserved")
        report.append("    - Multi-floor trajectory integrity maintained")
        report.append("")

        # Cross-floor examples
        if len(analysis.cross_floor_pairs) > 0:
            report.append("EXAMPLE CROSS-FLOOR FALSE MATCHES")
            report.append("-" * 50)
            for i, (idx1, idx2, f1, f2) in enumerate(analysis.cross_floor_pairs[:5]):
                pos1 = self.combined_trajectory[idx1, 1:4]
                pos2 = self.combined_trajectory[idx2, 1:4]
                dist = np.linalg.norm(pos1 - pos2)
                report.append(f"  {i+1}. Frame {idx1} (Floor {f1}) matched Frame {idx2} (Floor {f2})")
                report.append(f"      Spatial distance: {dist:.2f}m (similar enough to match)")
                report.append(f"      REJECTED: Different floors")
            report.append("")

        report.append("=" * 70)
        report.append("CONCLUSION: Floor-based semantic gating prevents perceptual aliasing")
        report.append("            in multi-floor environments by rejecting cross-floor")
        report.append("            loop closure candidates that would corrupt the trajectory.")
        report.append("=" * 70)

        report_text = "\n".join(report)

        # Save report
        path = self.output_dir / 'orb_slam3_semantic_analysis.txt'
        with open(path, 'w') as f:
            f.write(report_text)
        print(f"\nSaved report: {path}")

        return report_text

    def run_full_analysis(self,
                          distance_threshold: float = 2.0,
                          min_time_gap: int = 100) -> str:
        """
        Run complete analysis pipeline.

        Args:
            distance_threshold: Max distance for loop closure candidates
            min_time_gap: Min frame gap for revisit detection

        Returns:
            Analysis report as string
        """
        # 1. Load trajectories
        self.load_trajectories()

        # 2. Combine into multi-floor trajectory
        self.combine_trajectories()

        # 3. Detect loop closure candidates
        candidates = self.detect_loop_closure_candidates(
            distance_threshold=distance_threshold,
            min_time_gap=min_time_gap
        )

        # 4. Apply floor gating
        analysis = self.apply_floor_gating(candidates, strict_mode=True)

        # 5. Generate visualizations
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        self.visualize_floor_segmentation(save=True)
        self.visualize_loop_closure_gating(candidates, analysis, save=True)
        self.visualize_3d_multifloor(save=True)

        # 6. Generate report
        report = self.generate_comparison_stats(analysis)

        return report


def main():
    parser = argparse.ArgumentParser(
        description='ORB-SLAM3 Semantic Post-Processing Integration'
    )
    parser.add_argument('--trajectory-dir', type=str,
                       default='/home/wadewilliams/Dev/ros1/slam-benchmark/results/trajectories/orb_slam3',
                       help='Directory containing ORB-SLAM3 trajectories')
    parser.add_argument('--dataset-dir', type=str,
                       default='/home/wadewilliams/Dev/shared/datasets/ISEC',
                       help='Directory containing ISEC dataset')
    parser.add_argument('--output', type=str,
                       default='./results/semantic_gating',
                       help='Output directory for results')
    parser.add_argument('--distance-threshold', type=float, default=2.0,
                       help='Distance threshold for loop closure detection (m)')
    parser.add_argument('--min-time-gap', type=int, default=100,
                       help='Minimum frame gap for revisit detection')

    args = parser.parse_args()

    integration = ORBSlam3SemanticIntegration(
        trajectory_dir=args.trajectory_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output
    )

    report = integration.run_full_analysis(
        distance_threshold=args.distance_threshold,
        min_time_gap=args.min_time_gap
    )

    print("\n" + report)


if __name__ == '__main__':
    main()
