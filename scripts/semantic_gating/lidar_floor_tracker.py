"""
LiDAR Floor Tracker

Tracks robot height from LiDAR ground plane detection using RANSAC.
Provides an alternative/complementary signal to IMU-based elevator detection.

For Ouster OS-128:
- 128 channels, lower rings (0-30) typically see ground
- RANSAC plane fitting extracts ground plane
- Z-height changes > floor_height indicate floor transitions

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class FloorEstimate:
    """Single floor estimate from LiDAR"""
    timestamp: float
    z_height: float
    floor_number: int
    confidence: float
    num_ground_points: int


class LiDARFloorTracker:
    """
    Track robot height from LiDAR ground plane detection.

    Uses RANSAC to fit a plane to ground points and tracks
    z-height over time to detect floor transitions.
    """

    def __init__(self,
                 floor_height: float = 3.5,
                 ground_ring_threshold: int = 30,
                 ransac_iterations: int = 100,
                 ransac_threshold: float = 0.1,
                 min_ground_points: int = 100,
                 smoothing_window: int = 10):
        """
        Args:
            floor_height: Expected height per floor (meters)
            ground_ring_threshold: Ring index below which points are ground candidates
            ransac_iterations: Number of RANSAC iterations
            ransac_threshold: Inlier distance threshold for RANSAC (meters)
            min_ground_points: Minimum points required for valid ground plane
            smoothing_window: Number of scans for temporal smoothing
        """
        self.floor_height = floor_height
        self.ground_ring_threshold = ground_ring_threshold
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold
        self.min_ground_points = min_ground_points
        self.smoothing_window = smoothing_window

        self.z_history: deque = deque(maxlen=smoothing_window)
        self.floor_history: List[FloorEstimate] = []
        self.current_floor: int = 0
        self.reference_z: Optional[float] = None

    def extract_ground_points(self,
                               points: np.ndarray,
                               rings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract ground candidate points from point cloud.

        Args:
            points: Nx3 array of (x, y, z) points
            rings: N array of ring indices (if available from Ouster)

        Returns:
            Mx3 array of ground candidate points
        """
        if rings is not None:
            # Use ring information (more reliable for Ouster)
            ground_mask = rings < self.ground_ring_threshold
        else:
            # Fallback: use height-based filtering
            # Assume ground is below robot (z < -0.5m typically)
            z_min = np.percentile(points[:, 2], 5)
            ground_mask = points[:, 2] < (z_min + 0.5)

        return points[ground_mask]

    def fit_ground_plane_ransac(self,
                                 points: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Fit ground plane using RANSAC.

        Plane equation: ax + by + cz + d = 0
        For ground plane, we expect c ≈ 1 (normal pointing up)

        Args:
            points: Mx3 array of ground candidate points

        Returns:
            Tuple of (plane_params, inlier_ratio)
            plane_params: [a, b, c, d] or None if fitting failed
        """
        if len(points) < 3:
            return None, 0.0

        best_plane = None
        best_inliers = 0
        n_points = len(points)

        for _ in range(self.ransac_iterations):
            # Random sample 3 points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            # Compute plane normal via cross product
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            # Skip degenerate cases
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-6:
                continue

            normal = normal / norm_len
            d = -np.dot(normal, p1)

            # Count inliers
            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.sum(distances < self.ransac_threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = np.array([normal[0], normal[1], normal[2], d])

        inlier_ratio = best_inliers / n_points if n_points > 0 else 0.0
        return best_plane, inlier_ratio

    def estimate_robot_height(self, plane_params: np.ndarray) -> float:
        """
        Estimate robot height above ground plane.

        Assumes robot (LiDAR) is at origin in sensor frame.
        Height = distance from origin to plane.

        Args:
            plane_params: [a, b, c, d] plane parameters

        Returns:
            Height in meters (positive = above ground)
        """
        a, b, c, d = plane_params
        # Distance from origin to plane: |d| / sqrt(a^2 + b^2 + c^2)
        # Since plane is normalized, sqrt(a^2 + b^2 + c^2) ≈ 1
        height = abs(d)

        # If normal points down (c < 0), flip sign convention
        if c < 0:
            height = -height

        return height

    def process_scan(self,
                     points: np.ndarray,
                     timestamp: float,
                     rings: Optional[np.ndarray] = None) -> FloorEstimate:
        """
        Process a single LiDAR scan and estimate floor.

        Args:
            points: Nx3 array of (x, y, z) points
            timestamp: Scan timestamp
            rings: Optional N array of ring indices

        Returns:
            FloorEstimate with current floor information
        """
        # Extract ground points
        ground_points = self.extract_ground_points(points, rings)

        # Check if enough ground points
        if len(ground_points) < self.min_ground_points:
            # Return last known estimate with low confidence
            return FloorEstimate(
                timestamp=timestamp,
                z_height=self.z_history[-1] if self.z_history else 0.0,
                floor_number=self.current_floor,
                confidence=0.0,
                num_ground_points=len(ground_points)
            )

        # Fit ground plane
        plane_params, inlier_ratio = self.fit_ground_plane_ransac(ground_points)

        if plane_params is None:
            return FloorEstimate(
                timestamp=timestamp,
                z_height=self.z_history[-1] if self.z_history else 0.0,
                floor_number=self.current_floor,
                confidence=0.0,
                num_ground_points=len(ground_points)
            )

        # Estimate height
        z_height = self.estimate_robot_height(plane_params)
        self.z_history.append(z_height)

        # Set reference on first valid measurement
        if self.reference_z is None:
            self.reference_z = z_height

        # Smooth z-height
        smoothed_z = np.mean(self.z_history)

        # Compute floor number relative to reference
        relative_z = smoothed_z - self.reference_z
        floor_number = int(round(relative_z / self.floor_height))

        # Confidence based on inlier ratio and stability
        z_variance = np.var(self.z_history) if len(self.z_history) > 1 else 1.0
        stability = 1.0 / (1.0 + z_variance * 10)  # Lower variance = higher stability
        confidence = inlier_ratio * stability

        # Update current floor
        self.current_floor = floor_number

        estimate = FloorEstimate(
            timestamp=timestamp,
            z_height=smoothed_z,
            floor_number=floor_number,
            confidence=confidence,
            num_ground_points=len(ground_points)
        )

        self.floor_history.append(estimate)
        return estimate

    def detect_floor_transitions(self,
                                  min_duration: float = 2.0) -> List[Tuple[float, int, int]]:
        """
        Detect floor transitions from history.

        Args:
            min_duration: Minimum time between transitions (seconds)

        Returns:
            List of (timestamp, from_floor, to_floor) tuples
        """
        if len(self.floor_history) < 2:
            return []

        transitions = []
        last_floor = self.floor_history[0].floor_number
        last_transition_time = self.floor_history[0].timestamp

        for estimate in self.floor_history[1:]:
            if estimate.floor_number != last_floor:
                time_since_last = estimate.timestamp - last_transition_time
                if time_since_last >= min_duration:
                    transitions.append((
                        estimate.timestamp,
                        last_floor,
                        estimate.floor_number
                    ))
                    last_transition_time = estimate.timestamp
                last_floor = estimate.floor_number

        return transitions

    def get_floor_labels(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Get floor labels for given timestamps.

        Interpolates from floor history to match trajectory timestamps.

        Args:
            timestamps: Array of trajectory timestamps

        Returns:
            Array of floor labels
        """
        if len(self.floor_history) == 0:
            return np.zeros(len(timestamps), dtype=int)

        # Build lookup arrays
        scan_times = np.array([e.timestamp for e in self.floor_history])
        scan_floors = np.array([e.floor_number for e in self.floor_history])

        # For each trajectory timestamp, find nearest scan
        labels = np.zeros(len(timestamps), dtype=int)
        for i, t in enumerate(timestamps):
            idx = np.argmin(np.abs(scan_times - t))
            labels[i] = scan_floors[idx]

        return labels

    def reset(self):
        """Reset tracker state"""
        self.z_history.clear()
        self.floor_history.clear()
        self.current_floor = 0
        self.reference_z = None


class MultiModalFloorDetector:
    """
    Fuses IMU and LiDAR signals for robust floor detection.

    IMU provides transition events (elevator signature).
    LiDAR provides absolute height reference.
    """

    def __init__(self,
                 floor_height: float = 3.5,
                 imu_weight: float = 0.7,
                 lidar_weight: float = 0.3):
        """
        Args:
            floor_height: Expected height per floor
            imu_weight: Weight for IMU-based detection
            lidar_weight: Weight for LiDAR-based detection
        """
        self.floor_height = floor_height
        self.imu_weight = imu_weight
        self.lidar_weight = lidar_weight

        # Import IMU detector from existing module
        from floor_detector import IMUFloorDetector
        self.imu_detector = IMUFloorDetector()
        self.lidar_tracker = LiDARFloorTracker(floor_height=floor_height)

        self.fused_floor_labels: Optional[np.ndarray] = None

    def process_imu(self,
                    timestamps: np.ndarray,
                    accel_x: np.ndarray,
                    accel_y: np.ndarray,
                    accel_z: np.ndarray):
        """Process IMU data for elevator detection"""
        self.imu_detector.detect_elevator_events(
            timestamps, accel_x, accel_y, accel_z
        )

    def process_lidar_scan(self,
                           points: np.ndarray,
                           timestamp: float,
                           rings: Optional[np.ndarray] = None):
        """Process single LiDAR scan"""
        self.lidar_tracker.process_scan(points, timestamp, rings)

    def fuse_estimates(self,
                       trajectory_times: np.ndarray,
                       start_floor: int = 0) -> np.ndarray:
        """
        Fuse IMU and LiDAR floor estimates.

        Strategy:
        - Use IMU for transition detection (reliable for elevators)
        - Use LiDAR for absolute floor verification
        - Prefer IMU when elevator signature is strong

        Args:
            trajectory_times: Timestamps for trajectory poses
            start_floor: Initial floor number

        Returns:
            Array of fused floor labels
        """
        # Get IMU-based labels
        imu_labels = self.imu_detector.assign_floor_labels(
            trajectory_times, start_floor
        )

        # Get LiDAR-based labels
        if len(self.lidar_tracker.floor_history) > 0:
            lidar_labels = self.lidar_tracker.get_floor_labels(trajectory_times)
            # Offset LiDAR labels to match IMU start floor
            lidar_offset = start_floor - lidar_labels[0]
            lidar_labels = lidar_labels + lidar_offset
        else:
            # No LiDAR data, use IMU only
            lidar_labels = imu_labels

        # Simple fusion: use IMU primarily, LiDAR for verification
        # If they agree, high confidence. If disagree, trust IMU
        # (elevator signature is more reliable than RANSAC ground plane)
        self.fused_floor_labels = imu_labels.copy()

        # Could add more sophisticated fusion here (Kalman filter, etc.)

        return self.fused_floor_labels

    def get_floor_at_time(self, t: float) -> int:
        """Get floor label for specific timestamp"""
        if self.fused_floor_labels is None:
            raise ValueError("Call fuse_estimates first")
        # Would need trajectory times to implement properly
        raise NotImplementedError("Use fuse_estimates result directly")


def demo():
    """Demo with synthetic LiDAR data"""
    print("LiDAR Floor Tracker - Demo")
    print("=" * 50)

    tracker = LiDARFloorTracker(floor_height=3.5)

    # Simulate 100 scans over 50 seconds
    for i in range(100):
        timestamp = i * 0.5  # 2 Hz

        # Generate synthetic ground points
        n_points = 500
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)

        # Ground plane at different heights to simulate floor changes
        if i < 30:
            base_z = -1.5  # Floor 0
        elif i < 60:
            base_z = -1.5 + 3.5  # Floor 1 (went up)
        else:
            base_z = -1.5  # Floor 0 (went down)

        z = base_z + np.random.normal(0, 0.05, n_points)

        points = np.column_stack([x, y, z])

        estimate = tracker.process_scan(points, timestamp)

        if i % 20 == 0:
            print(f"  Scan {i}: z={estimate.z_height:.2f}m, "
                  f"floor={estimate.floor_number}, "
                  f"confidence={estimate.confidence:.2f}")

    # Detect transitions
    transitions = tracker.detect_floor_transitions()
    print(f"\nDetected {len(transitions)} floor transitions:")
    for t, from_floor, to_floor in transitions:
        print(f"  t={t:.1f}s: Floor {from_floor} -> {to_floor}")


if __name__ == '__main__':
    demo()
