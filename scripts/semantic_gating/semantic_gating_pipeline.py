"""
Semantic Gating Pipeline for Multi-Floor SLAM

Integrates IMU-based floor detection with loop closure gating
to prevent perceptual aliasing in buildings with similar floor layouts.

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation

Usage:
    python semantic_gating_pipeline.py --trajectory <path> --imu <path> --output <dir>
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List
import argparse

from .floor_detector import IMUFloorDetector, ElevatorEvent
from .loop_closure_gate import SemanticLoopClosureGate, LoopClosureCandidate


class SemanticGatingPipeline:
    """
    Complete pipeline for semantic-aware multi-floor SLAM post-processing.
    
    Pipeline stages:
    1. Load trajectory and IMU data
    2. Detect elevator events from IMU
    3. Assign floor labels to trajectory poses
    4. Gate loop closure candidates by floor consistency
    5. Visualize results and generate reports
    """
    
    def __init__(self, output_dir: str = './results/semantic_gating'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trajectory = None
        self.imu_data = None
        self.floor_detector = None
        self.loop_gate = None
        self.floor_labels = None
        
    def load_trajectory(self, path: str) -> np.ndarray:
        """Load trajectory in TUM format: timestamp x y z qx qy qz qw"""
        self.trajectory = np.loadtxt(path)
        print(f"Loaded trajectory: {len(self.trajectory)} poses")
        return self.trajectory
    
    def load_imu_data(self, path: str) -> np.ndarray:
        """Load IMU data: timestamp ax ay az gx gy gz"""
        self.imu_data = np.loadtxt(path)
        print(f"Loaded IMU data: {len(self.imu_data)} samples")
        return self.imu_data
    
    def detect_floors(self, 
                      start_floor: int = 5,
                      z_threshold: float = 0.5,
                      min_duration: float = 2.0) -> np.ndarray:
        """
        Run floor detection on loaded data.
        
        Args:
            start_floor: Initial floor number
            z_threshold: Z-acceleration threshold for elevator detection
            min_duration: Minimum elevator duration in seconds
            
        Returns:
            Array of floor labels for each trajectory pose
        """
        if self.imu_data is None:
            raise ValueError("Load IMU data first with load_imu_data()")
        
        self.floor_detector = IMUFloorDetector(
            z_accel_threshold=z_threshold,
            min_duration=min_duration
        )
        
        # Extract IMU components
        t = self.imu_data[:, 0]
        ax, ay, az = self.imu_data[:, 1], self.imu_data[:, 2], self.imu_data[:, 3]
        
        # Detect elevator events
        events = self.floor_detector.detect_elevator_events(t, ax, ay, az)
        print(f"Detected {len(events)} elevator events")
        
        for i, event in enumerate(events):
            print(f"  Event {i+1}: t={event.start_time:.1f}s, "
                  f"direction={event.direction}, duration={event.duration:.1f}s")
        
        # Assign floor labels to trajectory
        traj_times = self.trajectory[:, 0]
        self.floor_labels = self.floor_detector.assign_floor_labels(
            traj_times, start_floor
        )
        
        unique_floors = np.unique(self.floor_labels)
        print(f"Floor labels assigned: {unique_floors}")
        
        return self.floor_labels
    
    def create_loop_closure_gate(self, strict_mode: bool = True):
        """Initialize the loop closure gating system"""
        if self.floor_labels is None:
            raise ValueError("Run detect_floors() first")
        
        self.loop_gate = SemanticLoopClosureGate(
            self.floor_labels, 
            strict_mode=strict_mode
        )
        return self.loop_gate
    
    def gate_candidates(self, 
                        candidates: List[tuple]) -> tuple:
        """
        Filter loop closure candidates.
        
        Args:
            candidates: List of (query_idx, match_idx, score) tuples
            
        Returns:
            (valid_candidates, rejected_candidates)
        """
        if self.loop_gate is None:
            self.create_loop_closure_gate()
        
        return self.loop_gate.gate_candidates(candidates)
    
    def visualize_results(self, save: bool = True):
        """Generate visualization of floor-labeled trajectory"""
        if self.trajectory is None or self.floor_labels is None:
            raise ValueError("Run pipeline first")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Top-down view colored by floor
        ax = axes[0]
        floors = np.unique(self.floor_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(floors)))
        
        for floor, color in zip(floors, colors):
            mask = self.floor_labels == floor
            ax.plot(self.trajectory[mask, 1], self.trajectory[mask, 3],
                   color=color, label=f'Floor {floor}', linewidth=1.5)
        
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('z (m)', fontsize=12)
        ax.set_aspect('equal')
        ax.legend(loc='upper left')
        ax.set_title('Trajectory with Floor Labels', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Floor label over time
        ax2 = axes[1]
        traj_times = self.trajectory[:, 0]
        # Normalize time to start at 0
        t_normalized = traj_times - traj_times[0]
        ax2.plot(t_normalized, self.floor_labels, 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Floor', fontsize=12)
        ax2.set_title('Floor Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks(np.unique(self.floor_labels))
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / 'floor_segmentation.png'
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")
        
        plt.show()
    
    def visualize_3d(self, floor_height: float = 5.0, save: bool = True):
        """Generate 3D visualization with floor separation"""
        if self.trajectory is None or self.floor_labels is None:
            raise ValueError("Run pipeline first")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        floors = np.unique(self.floor_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(floors)))
        
        for floor, color in zip(floors, colors):
            mask = self.floor_labels == floor
            z_offset = (floor - floors.min()) * floor_height
            ax.plot(self.trajectory[mask, 1], 
                   self.trajectory[mask, 3],
                   z_offset * np.ones(mask.sum()),
                   color=color, label=f'Floor {floor}', linewidth=1.5)
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Floor Height (m)')
        ax.set_title('Multi-Floor Trajectory (Separated)', fontsize=14)
        ax.legend()
        
        if save:
            path = self.output_dir / 'floor_3d.png'
            plt.savefig(path, dpi=150)
            print(f"Saved: {path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate summary report"""
        if self.floor_detector is None:
            return "Pipeline not run yet"
        
        report = []
        report.append("=" * 60)
        report.append("SEMANTIC GATING PIPELINE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Trajectory info
        report.append("TRAJECTORY INFORMATION")
        report.append("-" * 40)
        report.append(f"  Total poses: {len(self.trajectory)}")
        duration = self.trajectory[-1, 0] - self.trajectory[0, 0]
        report.append(f"  Duration: {duration:.1f} seconds")
        report.append("")
        
        # Floor detection results
        report.append("FLOOR DETECTION RESULTS")
        report.append("-" * 40)
        report.append(f"  Elevator events detected: {len(self.floor_detector.events)}")
        for i, event in enumerate(self.floor_detector.events):
            report.append(f"    Event {i+1}: {event.direction} at t={event.start_time:.1f}s "
                         f"(duration={event.duration:.1f}s)")
        report.append("")
        
        # Floor distribution
        report.append("FLOOR DISTRIBUTION")
        report.append("-" * 40)
        floors, counts = np.unique(self.floor_labels, return_counts=True)
        for floor, count in zip(floors, counts):
            pct = 100 * count / len(self.floor_labels)
            report.append(f"  Floor {floor}: {count} poses ({pct:.1f}%)")
        report.append("")
        
        # Loop closure gating (if run)
        if self.loop_gate is not None:
            stats = self.loop_gate.get_stats()
            report.append("LOOP CLOSURE GATING")
            report.append("-" * 40)
            report.append(f"  Total candidates: {stats['total_candidates']}")
            report.append(f"  Accepted: {stats['accepted']}")
            report.append(f"  Rejected (cross-floor): {stats['rejected_cross_floor']}")
            if stats['total_candidates'] > 0:
                report.append(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")
            report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        path = self.output_dir / 'semantic_gating_report.txt'
        with open(path, 'w') as f:
            f.write(report_text)
        print(f"Saved: {path}")
        
        return report_text


def run_demo():
    """Run demo with synthetic data"""
    print("=" * 60)
    print("SEMANTIC GATING PIPELINE - DEMO")
    print("=" * 60)
    
    pipeline = SemanticGatingPipeline(output_dir='/tmp/semantic_gating_demo')
    
    # Create synthetic trajectory (simulating ISEC 5th floor)
    n_poses = 5000
    t = np.linspace(0, 300, n_poses)  # 5 minutes
    
    # Circular-ish trajectory
    theta = np.linspace(0, 2*np.pi, n_poses)
    x = 20 * np.cos(theta) + np.random.normal(0, 0.1, n_poses)
    y = np.zeros(n_poses)  # Height stays constant on single floor
    z = 30 * np.sin(theta) + np.random.normal(0, 0.1, n_poses)
    
    # Create trajectory array
    pipeline.trajectory = np.column_stack([t, x, y, z, 
                                           np.zeros(n_poses),  # qx
                                           np.zeros(n_poses),  # qy
                                           np.zeros(n_poses),  # qz
                                           np.ones(n_poses)])  # qw
    
    # Create synthetic IMU data
    imu_rate = 200  # Hz
    n_imu = int(300 * imu_rate)
    t_imu = np.linspace(0, 300, n_imu)
    ax = np.random.normal(0, 0.1, n_imu)
    ay = np.random.normal(0, 0.1, n_imu)
    az = np.random.normal(9.81, 0.1, n_imu)
    gx = np.random.normal(0, 0.01, n_imu)
    gy = np.random.normal(0, 0.01, n_imu)
    gz = np.random.normal(0, 0.01, n_imu)
    
    # Inject elevator events
    # Event 1: t=100-105s (going down from 5 to 1)
    mask1 = (t_imu >= 100) & (t_imu <= 105)
    az[mask1] -= 0.8
    
    # Event 2: t=200-204s (going up from 1 to 4)
    mask2 = (t_imu >= 200) & (t_imu <= 204)
    az[mask2] += 0.7
    
    pipeline.imu_data = np.column_stack([t_imu, ax, ay, az, gx, gy, gz])
    
    # Run floor detection
    pipeline.floor_detector = IMUFloorDetector()
    events = pipeline.floor_detector.detect_elevator_events(t_imu, ax, ay, az)
    pipeline.floor_labels = pipeline.floor_detector.assign_floor_labels(
        pipeline.trajectory[:, 0], start_floor=5
    )
    
    print(f"\nDetected {len(events)} elevator events")
    print(f"Floor labels: {np.unique(pipeline.floor_labels)}")
    
    # Create loop closure gate and test candidates
    pipeline.create_loop_closure_gate(strict_mode=True)
    
    # Simulate some loop closure candidates
    candidates = [
        (100, 4500, 0.85),   # Same floor - valid
        (500, 2500, 0.92),   # Cross-floor - reject!
        (1000, 1500, 0.88),  # Same floor - valid
        (200, 3000, 0.91),   # Cross-floor - reject!
    ]
    
    valid, rejected = pipeline.gate_candidates(candidates)
    
    print(f"\nLoop closure gating:")
    print(f"  Valid: {len(valid)}")
    print(f"  Rejected: {len(rejected)}")
    
    # Generate report
    print("\n" + pipeline.generate_report())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Semantic Gating Pipeline for Multi-Floor SLAM'
    )
    parser.add_argument('--trajectory', type=str, help='Path to trajectory file (TUM format)')
    parser.add_argument('--imu', type=str, help='Path to IMU data file')
    parser.add_argument('--output', type=str, default='./results/semantic_gating',
                       help='Output directory')
    parser.add_argument('--start-floor', type=int, default=5, help='Starting floor number')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.trajectory and args.imu:
        pipeline = SemanticGatingPipeline(output_dir=args.output)
        pipeline.load_trajectory(args.trajectory)
        pipeline.load_imu_data(args.imu)
        pipeline.detect_floors(start_floor=args.start_floor)
        pipeline.visualize_results()
        pipeline.visualize_3d()
        print(pipeline.generate_report())
    else:
        print("Use --demo for demo, or provide --trajectory and --imu paths")
        parser.print_help()
