#!/usr/bin/env python3
"""
Test LiDAR Floor Tracker on Real ISEC Dataset

Validates multi-modal floor detection by:
1. Extracting point clouds from Ouster OS-128 LiDAR
2. Running LiDARFloorTracker RANSAC ground plane fitting
3. Extracting IMU data and running IMUFloorDetector
4. Comparing LiDAR vs IMU floor estimates
5. Creating visualizations

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from .lidar_floor_tracker import LiDARFloorTracker, FloorEstimate
from .floor_detector import IMUFloorDetector

# Try to import rosbags with typestore
ROSBAGS_AVAILABLE = False
Reader = None
typestore = None

try:
    from rosbags.rosbag1 import Reader as _Reader
    from rosbags.typesys import Stores, get_typestore
    Reader = _Reader
    typestore = get_typestore(Stores.ROS1_NOETIC)
    ROSBAGS_AVAILABLE = True
except ImportError:
    print("Warning: rosbags not available. Install with: pip install rosbags")


def parse_pointcloud2_proper(msg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse PointCloud2 message using proper field offsets.

    Ouster OS-128 format:
    - 48 bytes per point
    - Fields: x(0), y(4), z(8), intensity(16), t(20), reflectivity(24), ring(26), ambient(28), range(32)

    Returns:
        Tuple of (points Nx3, rings N)
    """
    point_step = msg.point_step
    n_points = msg.height * msg.width
    data = msg.data

    # Pre-allocate arrays
    points = np.zeros((n_points, 3), dtype=np.float32)
    rings = np.zeros(n_points, dtype=np.uint16)

    # Extract xyz and ring using numpy for speed
    for i in range(n_points):
        base = i * point_step
        points[i, 0] = np.frombuffer(data[base:base+4], dtype=np.float32)[0]      # x
        points[i, 1] = np.frombuffer(data[base+4:base+8], dtype=np.float32)[0]    # y
        points[i, 2] = np.frombuffer(data[base+8:base+12], dtype=np.float32)[0]   # z
        rings[i] = np.frombuffer(data[base+26:base+28], dtype=np.uint16)[0]       # ring

    # Filter out invalid points (0,0,0) and out of range
    valid_mask = ~((points[:, 0] == 0) & (points[:, 1] == 0) & (points[:, 2] == 0))
    valid_mask &= (np.abs(points[:, 0]) < 100)
    valid_mask &= (np.abs(points[:, 1]) < 100)
    valid_mask &= (np.abs(points[:, 2]) < 100)

    return points[valid_mask], rings[valid_mask]


def process_bag_file(bag_path: Path,
                     max_scans: int = 100,
                     verbose: bool = True) -> Dict:
    """
    Process a single bag file to extract LiDAR and IMU data.

    Returns:
        Dict with 'lidar_estimates', 'imu_data', 'timestamps'
    """
    if not ROSBAGS_AVAILABLE:
        raise ImportError("rosbags package required")

    if verbose:
        print(f"\nProcessing: {bag_path.name}")

    lidar_tracker = LiDARFloorTracker(
        floor_height=3.5,
        ground_ring_threshold=40,  # Lower rings see ground on OS-128
        ransac_iterations=200,
        ransac_threshold=0.05,
        min_ground_points=100,
        smoothing_window=5
    )

    lidar_estimates = []
    imu_timestamps = []
    imu_accel_x = []
    imu_accel_y = []
    imu_accel_z = []

    scan_count = 0
    imu_count = 0

    with Reader(bag_path) as reader:
        start_time = reader.start_time / 1e9

        for connection, timestamp, rawdata in reader.messages():
            msg_time = timestamp / 1e9 - start_time

            # Process LiDAR
            if '/ouster/points' in connection.topic and scan_count < max_scans:
                try:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    points, rings = parse_pointcloud2_proper(msg)

                    if len(points) > 100:
                        estimate = lidar_tracker.process_scan(points, msg_time, rings)
                        lidar_estimates.append(estimate)
                        scan_count += 1

                        if verbose and scan_count % 20 == 0:
                            print(f"  Scan {scan_count}: {len(points)} pts, "
                                  f"z={estimate.z_height:.3f}m, "
                                  f"floor={estimate.floor_number}, "
                                  f"conf={estimate.confidence:.2f}, "
                                  f"ground_pts={estimate.num_ground_points}")
                except Exception as e:
                    if verbose:
                        print(f"  Error parsing scan: {e}")

            # Process IMU
            if 'imu_compensated' in connection.topic:
                try:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    imu_timestamps.append(msg_time)
                    imu_accel_x.append(msg.linear_acceleration.x)
                    imu_accel_y.append(msg.linear_acceleration.y)
                    imu_accel_z.append(msg.linear_acceleration.z)
                    imu_count += 1
                except:
                    pass

    if verbose:
        print(f"  Processed {scan_count} LiDAR scans, {imu_count} IMU messages")

    return {
        'lidar_estimates': lidar_estimates,
        'imu_timestamps': np.array(imu_timestamps),
        'imu_accel_x': np.array(imu_accel_x),
        'imu_accel_y': np.array(imu_accel_y),
        'imu_accel_z': np.array(imu_accel_z),
        'lidar_tracker': lidar_tracker
    }


def run_imu_floor_detection(imu_timestamps: np.ndarray,
                            imu_accel_x: np.ndarray,
                            imu_accel_y: np.ndarray,
                            imu_accel_z: np.ndarray,
                            verbose: bool = True) -> Tuple[List, np.ndarray]:
    """
    Run IMU-based floor detection on extracted data.
    """
    if len(imu_timestamps) < 10:
        return [], np.array([])

    detector = IMUFloorDetector(
        z_accel_threshold=0.3,  # Lower threshold for better sensitivity
        min_duration=2.0
    )

    events = detector.detect_elevator_events(
        imu_timestamps, imu_accel_x, imu_accel_y, imu_accel_z
    )

    floor_labels = detector.assign_floor_labels(imu_timestamps, start_floor=0)

    if verbose:
        print(f"  IMU: Detected {len(events)} elevator events")
        for event in events:
            print(f"    {event.direction} at t={event.start_time:.1f}s "
                  f"(delta_v={event.velocity_change:.2f}m/s)")

    return events, floor_labels


def create_visualization(results: Dict, output_path: Path):
    """
    Create visualization comparing LiDAR vs IMU floor detection.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Plot 1: LiDAR z-height over time
    ax1 = axes[0]
    if results['lidar_estimates']:
        times = [e.timestamp for e in results['lidar_estimates']]
        z_heights = [e.z_height for e in results['lidar_estimates']]
        confidences = [e.confidence for e in results['lidar_estimates']]

        scatter = ax1.scatter(times, z_heights, c=confidences, cmap='viridis',
                             s=30, alpha=0.8, edgecolors='black', linewidths=0.5)
        ax1.plot(times, z_heights, 'b-', alpha=0.4, linewidth=1)
        plt.colorbar(scatter, ax=ax1, label='Confidence')

        # Add reference lines
        ref_z = results.get('reference_z', 0)
        if ref_z is not None:
            for floor in range(-1, 5):
                y = ref_z + floor * 3.5
                ax1.axhline(y=y, color='gray', linestyle=':', alpha=0.4)

    ax1.set_ylabel('LiDAR Z-Height (m)')
    ax1.set_title('LiDAR Ground Plane Detection (RANSAC)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Ground points count
    ax2 = axes[1]
    if results['lidar_estimates']:
        times = [e.timestamp for e in results['lidar_estimates']]
        ground_pts = [e.num_ground_points for e in results['lidar_estimates']]
        ax2.bar(times, ground_pts, width=0.08, alpha=0.7, color='green')
    ax2.set_ylabel('Ground Points')
    ax2.set_title('Ground Points per Scan (for RANSAC)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: IMU z-acceleration
    ax3 = axes[2]
    if len(results.get('imu_timestamps', [])) > 0:
        # Downsample for cleaner plot
        step = max(1, len(results['imu_timestamps']) // 2000)
        ax3.plot(results['imu_timestamps'][::step], results['imu_accel_z'][::step],
                'g-', alpha=0.7, linewidth=0.5)
        ax3.axhline(y=-9.81, color='r', linestyle='--', alpha=0.5, label='Gravity (-9.81)')

        # Highlight elevator events
        colors = {'up': 'blue', 'down': 'red'}
        for event in results.get('imu_events', []):
            ax3.axvspan(event.start_time, event.end_time,
                       alpha=0.3, color=colors.get(event.direction, 'yellow'))

    ax3.set_ylabel('IMU Z-Accel (m/s²)')
    ax3.set_title('IMU Z-Acceleration (Elevator Detection)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    # Plot 4: Floor estimates comparison
    ax4 = axes[3]

    # LiDAR floor estimates
    if results['lidar_estimates']:
        times = [e.timestamp for e in results['lidar_estimates']]
        floors = [e.floor_number for e in results['lidar_estimates']]
        ax4.step(times, floors, 'b-', where='post', linewidth=2.5,
                label='LiDAR Floor', alpha=0.8)

    # IMU floor estimates
    if len(results.get('imu_floor_labels', [])) > 0:
        step = max(1, len(results['imu_timestamps']) // 500)
        ax4.step(results['imu_timestamps'][::step],
                results['imu_floor_labels'][::step],
                'r--', where='post', linewidth=2,
                label='IMU Floor', alpha=0.8)

    ax4.set_ylabel('Floor Number')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Floor Detection Comparison: LiDAR (solid) vs IMU (dashed)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    ax4.set_yticks(range(-2, 6))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def test_single_floor(bag_path: Path, output_dir: Path, floor_name: str) -> Dict:
    """Test floor tracker on a single-floor bag."""
    print(f"\n{'='*60}")
    print(f"Testing on SINGLE-FLOOR bag: {bag_path.name}")
    print(f"Expected: Stable z-height (variance < 0.1m)")
    print(f"{'='*60}")

    results = process_bag_file(bag_path, max_scans=150)

    # Run IMU floor detection
    events, floor_labels = run_imu_floor_detection(
        results['imu_timestamps'],
        results['imu_accel_x'],
        results['imu_accel_y'],
        results['imu_accel_z']
    )

    results['imu_events'] = events
    results['imu_floor_labels'] = floor_labels
    results['reference_z'] = results['lidar_tracker'].reference_z

    # Create visualization
    output_path = output_dir / f'lidar_vs_imu_{floor_name}.png'
    create_visualization(results, output_path)

    # Statistics
    if results['lidar_estimates']:
        z_heights = [e.z_height for e in results['lidar_estimates']]
        confidences = [e.confidence for e in results['lidar_estimates']]

        print(f"\n  LiDAR Statistics:")
        print(f"    Z-height mean: {np.mean(z_heights):.3f}m")
        print(f"    Z-height std:  {np.std(z_heights):.3f}m")
        print(f"    Z-height range: {min(z_heights):.3f}m to {max(z_heights):.3f}m")
        print(f"    Mean confidence: {np.mean(confidences):.3f}")
        print(f"    Reference z: {results['reference_z']:.3f}m")

        # Check if stable (single floor)
        if np.std(z_heights) < 0.15:
            print(f"    STATUS: PASS - Z-height is stable (std < 0.15m)")
        else:
            print(f"    STATUS: WARN - Z-height variance higher than expected")

    return results


def test_transit(bag_path: Path, output_dir: Path, transit_name: str) -> Dict:
    """Test floor tracker on a transit (elevator) bag."""
    print(f"\n{'='*60}")
    print(f"Testing on TRANSIT bag: {bag_path.name}")
    print(f"Expected: ~3.5m z-height change during elevator ride")
    print(f"{'='*60}")

    results = process_bag_file(bag_path, max_scans=300)

    # Run IMU floor detection
    events, floor_labels = run_imu_floor_detection(
        results['imu_timestamps'],
        results['imu_accel_x'],
        results['imu_accel_y'],
        results['imu_accel_z']
    )

    results['imu_events'] = events
    results['imu_floor_labels'] = floor_labels
    results['reference_z'] = results['lidar_tracker'].reference_z

    # Detect LiDAR floor transitions
    transitions = results['lidar_tracker'].detect_floor_transitions(min_duration=2.0)
    print(f"\n  LiDAR Floor Transitions: {len(transitions)}")
    for t, from_f, to_f in transitions:
        print(f"    t={t:.1f}s: Floor {from_f} -> Floor {to_f}")

    # Create visualization
    output_path = output_dir / f'lidar_vs_imu_{transit_name}.png'
    create_visualization(results, output_path)

    # Statistics
    if results['lidar_estimates']:
        z_heights = [e.z_height for e in results['lidar_estimates']]

        print(f"\n  LiDAR Statistics:")
        print(f"    Z-height range: {min(z_heights):.3f}m to {max(z_heights):.3f}m")
        print(f"    Total change: {max(z_heights) - min(z_heights):.3f}m")

        # Check if transit detected
        z_change = max(z_heights) - min(z_heights)
        if z_change > 2.0:
            floors_changed = int(round(z_change / 3.5))
            print(f"    STATUS: PASS - Detected ~{floors_changed} floor change ({z_change:.1f}m)")
        else:
            print(f"    STATUS: WARN - Expected larger z-height change for transit")

    return results


def main():
    """Main test function."""
    print("="*60)
    print("LiDAR Floor Tracker Validation on ISEC Dataset")
    print("Multi-Modal Floor Detection: LiDAR + IMU Comparison")
    print("="*60)

    if not ROSBAGS_AVAILABLE:
        print("\nERROR: rosbags package not installed.")
        print("Install with: pip install rosbags")
        print("\nRunning demo with synthetic data instead...")
        from lidar_floor_tracker import demo
        demo()
        return

    # Dataset paths
    dataset_base = Path('/home/wadewilliams/Dev/shared/datasets/ISEC')
    output_dir = Path('/home/wadewilliams/Dev/ros1/slam-benchmark/results/semantic_gating')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Test on 5th floor (single floor, should have stable z-height)
    floor_bag = dataset_base / '5th_floor' / '2023-03-14-11-56-21_0.bag'
    if floor_bag.exists():
        all_results['5th_floor'] = test_single_floor(floor_bag, output_dir, '5th_floor')

    # Test on 1st floor
    floor_bag = dataset_base / '1st_floor' / '2023-03-14-12-03-12_15.bag'
    if floor_bag.exists():
        all_results['1st_floor'] = test_single_floor(floor_bag, output_dir, '1st_floor')

    # Test on transit 1->4 (should show floor change)
    transit_bag = dataset_base / 'transit_1_to_4' / '2023-03-14-12-07-36_21.bag'
    if transit_bag.exists():
        all_results['transit_1_to_4'] = test_transit(transit_bag, output_dir, 'transit_1_to_4')

    # Test on transit 5->1
    transit_bag = dataset_base / 'transit_5_to_1' / '2023-03-14-12-23-03_55.bag'
    if transit_bag.exists():
        all_results['transit_5_to_1'] = test_transit(transit_bag, output_dir, 'transit_5_to_1')

    # Summary comparison
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    print("\nSingle-Floor Tests:")
    for name in ['5th_floor', '1st_floor']:
        if name in all_results and all_results[name]['lidar_estimates']:
            z_heights = [e.z_height for e in all_results[name]['lidar_estimates']]
            print(f"  {name}: z_std={np.std(z_heights):.3f}m, "
                  f"imu_events={len(all_results[name].get('imu_events', []))}")

    print("\nTransit Tests:")
    for name in ['transit_1_to_4', 'transit_5_to_1']:
        if name in all_results and all_results[name]['lidar_estimates']:
            z_heights = [e.z_height for e in all_results[name]['lidar_estimates']]
            z_change = max(z_heights) - min(z_heights)
            imu_events = len(all_results[name].get('imu_events', []))
            print(f"  {name}: z_change={z_change:.2f}m, imu_events={imu_events}")

    print("""
Conclusion:
- LiDAR ground plane detection provides absolute height reference
- IMU acceleration detects elevator motion events
- Multi-modal fusion recommended for robust floor estimation:
  * IMU: Fast response to elevator start/stop
  * LiDAR: Accurate floor height verification
""")


if __name__ == '__main__':
    main()
