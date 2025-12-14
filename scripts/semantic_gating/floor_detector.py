"""
Floor Detection from IMU Data

Detects elevator events and floor transitions using IMU acceleration patterns.
Part of the semantic gating pipeline for multi-floor SLAM.

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ElevatorEvent:
    """Represents a detected elevator ride"""
    start_time: float
    end_time: float
    duration: float
    direction: str  # 'up' or 'down'
    start_idx: int
    end_idx: int
    floor_change: int  # +1 for up, -1 for down


class IMUFloorDetector:
    """
    Detect floor transitions from IMU z-acceleration patterns.
    
    Elevator signature:
    - Sustained vertical acceleration deviation from gravity
    - Low horizontal acceleration variance
    - Duration > 2 seconds (typical elevator ride)
    
    Stair signature (for future):
    - Periodic vertical oscillations
    - Correlated with step frequency (~1-2 Hz)
    """
    
    def __init__(self, 
                 z_accel_threshold: float = 0.5,
                 min_duration: float = 2.0,
                 window_size: int = 50,
                 horizontal_var_threshold: float = 1.0):
        """
        Args:
            z_accel_threshold: m/s^2 deviation from gravity to detect elevator
            min_duration: Minimum seconds for valid elevator event
            window_size: Smoothing window in samples
            horizontal_var_threshold: Max horizontal variance during elevator
        """
        self.z_accel_threshold = z_accel_threshold
        self.min_duration = min_duration
        self.window_size = window_size
        self.horizontal_var_threshold = horizontal_var_threshold
        
        self.events: List[ElevatorEvent] = []
        self.floor_labels: Optional[np.ndarray] = None
    
    def detect_elevator_events(self, 
                                timestamps: np.ndarray,
                                accel_x: np.ndarray,
                                accel_y: np.ndarray,
                                accel_z: np.ndarray) -> List[ElevatorEvent]:
        """
        Detect elevator rides from IMU acceleration data.
        
        Args:
            timestamps: Time values (seconds)
            accel_x, accel_y, accel_z: Accelerometer readings (m/s^2)
            
        Returns:
            List of detected ElevatorEvent objects
        """
        # Remove gravity baseline (assuming z-up convention)
        az_detrended = accel_z - np.median(accel_z)
        
        # Smooth to remove sensor noise
        az_smooth = uniform_filter1d(az_detrended, size=self.window_size)
        horiz_var = uniform_filter1d(accel_x**2 + accel_y**2, size=self.window_size)
        
        # Elevator signature: significant z-accel + low horizontal motion
        elevator_mask = ((np.abs(az_smooth) > self.z_accel_threshold) & 
                        (horiz_var < self.horizontal_var_threshold))
        
        # Find continuous elevator segments
        self.events = []
        in_elevator = False
        start_idx = 0
        
        for i, is_elev in enumerate(elevator_mask):
            if is_elev and not in_elevator:
                start_idx = i
                in_elevator = True
            elif not is_elev and in_elevator:
                duration = timestamps[i] - timestamps[start_idx]
                if duration >= self.min_duration:
                    # Determine direction from integrated acceleration
                    z_integral = np.trapz(az_smooth[start_idx:i], 
                                         timestamps[start_idx:i])
                    direction = 'up' if z_integral > 0 else 'down'
                    floor_change = 1 if direction == 'up' else -1
                    
                    self.events.append(ElevatorEvent(
                        start_time=timestamps[start_idx],
                        end_time=timestamps[i],
                        duration=duration,
                        direction=direction,
                        start_idx=start_idx,
                        end_idx=i,
                        floor_change=floor_change
                    ))
                in_elevator = False
        
        return self.events
    
    def assign_floor_labels(self, 
                           trajectory_times: np.ndarray,
                           start_floor: int = 5) -> np.ndarray:
        """
        Assign floor labels to trajectory poses based on elevator events.
        
        Args:
            trajectory_times: Timestamps for each trajectory pose
            start_floor: Initial floor number
            
        Returns:
            Array of floor labels for each pose
        """
        n = len(trajectory_times)
        self.floor_labels = np.zeros(n, dtype=int)
        
        # Sort events by time
        events = sorted(self.events, key=lambda x: x.start_time)
        
        current_floor = start_floor
        last_end = trajectory_times[0]
        
        for event in events:
            # Label segment before this elevator
            mask = ((trajectory_times >= last_end) & 
                   (trajectory_times < event.start_time))
            self.floor_labels[mask] = current_floor
            
            # Update floor based on direction
            current_floor += event.floor_change
            last_end = event.end_time
        
        # Label remaining segment after last elevator
        mask = trajectory_times >= last_end
        self.floor_labels[mask] = current_floor
        
        return self.floor_labels
    
    def get_floor_at_time(self, t: float) -> int:
        """Get floor label for a specific timestamp"""
        if self.floor_labels is None:
            raise ValueError("Call assign_floor_labels first")
        # This is a simplified version - would need trajectory times
        raise NotImplementedError("Use assign_floor_labels result directly")


def load_imu_from_bag(bag_path: str, 
                      imu_topic: str = '/vectornav/imu') -> Tuple[np.ndarray, ...]:
    """
    Load IMU data from a ROS bag file.
    
    Args:
        bag_path: Path to .bag file
        imu_topic: IMU topic name
        
    Returns:
        Tuple of (timestamps, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
    """
    try:
        import rosbag
    except ImportError:
        raise ImportError("rosbag not available. Run inside ROS environment.")
    
    timestamps = []
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[imu_topic]):
            timestamps.append(t.to_sec())
            accel_x.append(msg.linear_acceleration.x)
            accel_y.append(msg.linear_acceleration.y)
            accel_z.append(msg.linear_acceleration.z)
            gyro_x.append(msg.angular_velocity.x)
            gyro_y.append(msg.angular_velocity.y)
            gyro_z.append(msg.angular_velocity.z)
    
    return (np.array(timestamps),
            np.array(accel_x), np.array(accel_y), np.array(accel_z),
            np.array(gyro_x), np.array(gyro_y), np.array(gyro_z))


if __name__ == '__main__':
    # Demo with synthetic data
    print("IMU Floor Detector - Demo")
    print("=" * 50)
    
    # Simulate 60 seconds of IMU data at 200Hz
    dt = 1/200
    t = np.arange(0, 60, dt)
    n = len(t)
    
    # Baseline: gravity in z, noise in x/y
    ax = np.random.normal(0, 0.1, n)
    ay = np.random.normal(0, 0.1, n)
    az = np.random.normal(9.81, 0.1, n)
    
    # Inject elevator event from t=20 to t=25 (going up)
    elev_mask = (t >= 20) & (t <= 25)
    az[elev_mask] += 0.8  # Upward acceleration
    
    # Inject another elevator from t=40 to t=44 (going down)
    elev_mask2 = (t >= 40) & (t <= 44)
    az[elev_mask2] -= 0.7  # Downward acceleration
    
    detector = IMUFloorDetector()
    events = detector.detect_elevator_events(t, ax, ay, az)
    
    print(f"Detected {len(events)} elevator events:")
    for i, event in enumerate(events):
        print(f"  Event {i+1}: t={event.start_time:.1f}s to {event.end_time:.1f}s, "
              f"direction={event.direction}, duration={event.duration:.1f}s")
    
    # Assign floor labels
    traj_times = np.linspace(0, 60, 1000)
    labels = detector.assign_floor_labels(traj_times, start_floor=5)
    
    print(f"\nFloor labels assigned: {np.unique(labels)}")
