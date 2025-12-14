#!/usr/bin/env python3
"""Extract IMU data from transit bags to demonstrate elevator detection."""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def extract_imu_from_bags(bag_dir, imu_topic='/imu/imu_compensated'):
    """Extract IMU data from all bags in directory."""
    bag_files = sorted(Path(bag_dir).glob('*.bag'))
    
    timestamps = []
    linear_accel = []
    angular_vel = []
    
    for bag_file in bag_files:
        print(f"Processing {bag_file.name}...")
        try:
            bag = rosbag.Bag(str(bag_file))
            for topic, msg, t in bag.read_messages(topics=[imu_topic]):
                timestamps.append(t.to_sec())
                linear_accel.append([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ])
                angular_vel.append([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ])
            bag.close()
        except Exception as e:
            print(f"Error processing {bag_file}: {e}")
    
    return np.array(timestamps), np.array(linear_accel), np.array(angular_vel)

def plot_elevator_detection(timestamps, linear_accel, output_path):
    """Plot IMU z-acceleration showing elevator motion."""
    # Normalize time to start at 0
    t = timestamps - timestamps[0]
    
    # Z-acceleration (vertical)
    az = linear_accel[:, 2]
    
    # Apply simple smoothing
    window = 50
    az_smooth = np.convolve(az, np.ones(window)/window, mode='same')
    
    # Detect elevator motion: look for sustained deviation from gravity (~9.81)
    gravity = 9.81
    az_deviation = az_smooth - gravity
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Raw Z acceleration
    axes[0].plot(t, az, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
    axes[0].plot(t, az_smooth, 'b-', linewidth=1.5, label='Smoothed')
    axes[0].axhline(y=gravity, color='r', linestyle='--', label=f'Gravity ({gravity} m/s²)')
    axes[0].set_ylabel('Z Acceleration (m/s²)')
    axes[0].set_title('Vertical Acceleration During Elevator Transit (5th → 1st Floor)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Deviation from gravity (shows elevator acceleration)
    axes[1].plot(t, az_deviation, 'g-', linewidth=1)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    axes[1].axhline(y=-0.5, color='r', linestyle='--', alpha=0.5)
    axes[1].fill_between(t, az_deviation, 0, where=(az_deviation > 0.3), 
                          color='green', alpha=0.3, label='Ascending')
    axes[1].fill_between(t, az_deviation, 0, where=(az_deviation < -0.3), 
                          color='red', alpha=0.3, label='Descending')
    axes[1].set_ylabel('Deviation from Gravity (m/s²)')
    axes[1].set_title('Elevator Motion Detection')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: All accelerations for context
    axes[2].plot(t, linear_accel[:, 0], 'r-', alpha=0.5, linewidth=0.5, label='X (forward)')
    axes[2].plot(t, linear_accel[:, 1], 'g-', alpha=0.5, linewidth=0.5, label='Y (lateral)')
    axes[2].plot(t, linear_accel[:, 2], 'b-', alpha=0.5, linewidth=0.5, label='Z (vertical)')
    axes[2].set_ylabel('Acceleration (m/s²)')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('All Acceleration Components')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    # Also save data
    data_path = output_path.replace('.png', '_data.npz')
    np.savez(data_path, timestamps=timestamps, linear_accel=linear_accel)
    print(f"Data saved to {data_path}")

if __name__ == '__main__':
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else '/data/ISEC/transit_5_to_1'
    output = sys.argv[2] if len(sys.argv) > 2 else '/results/figures/elevator_imu_detection.png'
    
    print(f"Extracting IMU from {bag_dir}")
    timestamps, linear_accel, angular_vel = extract_imu_from_bags(bag_dir)
    
    print(f"Extracted {len(timestamps)} IMU samples over {timestamps[-1]-timestamps[0]:.1f} seconds")
    plot_elevator_detection(timestamps, linear_accel, output)
