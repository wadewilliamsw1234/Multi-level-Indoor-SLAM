#!/usr/bin/env python3
"""
Diagnostic script to dump IMU samples from ISEC bag files.
This helps verify the IMU frame convention (NED vs FLU).

Expected behavior when robot is stationary:
- FLU (Forward-Left-Up): gravity in +Z → az ≈ +9.81
- NED (North-East-Down): gravity in -Z → az ≈ -9.81
- ENU (East-North-Up): gravity in +Z → az ≈ +9.81

Usage:
  python3 dump_imu_samples.py /path/to/bag_file.bag [--topic /imu/imu_uncompensated]
"""

import sys
import argparse
import numpy as np

try:
    import rosbag
except ImportError:
    print("ERROR: rosbag not installed. Run this inside a ROS container.")
    sys.exit(1)


def analyze_imu_data(bag_file, topic, num_samples=100):
    """Extract and analyze IMU data from bag file."""
    
    print(f"\n{'='*60}")
    print(f"IMU Frame Convention Diagnostic")
    print(f"{'='*60}")
    print(f"Bag file: {bag_file}")
    print(f"Topic: {topic}")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}\n")
    
    accel_data = []
    gyro_data = []
    timestamps = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        # Get topic info
        info = bag.get_type_and_topic_info()
        if topic not in info.topics:
            print(f"ERROR: Topic '{topic}' not found in bag!")
            print(f"Available topics with 'imu':")
            for t in info.topics:
                if 'imu' in t.lower():
                    print(f"  {t}: {info.topics[t].msg_type} ({info.topics[t].message_count} msgs)")
            return None
        
        msg_count = info.topics[topic].message_count
        print(f"Total messages in topic: {msg_count}")
        print(f"Message type: {info.topics[topic].msg_type}\n")
        
        # Extract samples
        for i, (_, msg, t) in enumerate(bag.read_messages(topics=[topic])):
            if i >= num_samples:
                break
                
            timestamps.append(t.to_sec())
            accel_data.append([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            gyro_data.append([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            
            # Print first 10 samples in detail
            if i < 10:
                print(f"Sample {i+1}:")
                print(f"  Time: {t.to_sec():.6f}")
                print(f"  Accel: [{msg.linear_acceleration.x:8.4f}, "
                      f"{msg.linear_acceleration.y:8.4f}, "
                      f"{msg.linear_acceleration.z:8.4f}] m/s²")
                print(f"  Gyro:  [{msg.angular_velocity.x:8.4f}, "
                      f"{msg.angular_velocity.y:8.4f}, "
                      f"{msg.angular_velocity.z:8.4f}] rad/s")
                print()
    
    accel_data = np.array(accel_data)
    gyro_data = np.array(gyro_data)
    
    # Statistical analysis
    print(f"\n{'='*60}")
    print("Statistical Analysis (first {num_samples} samples)")
    print(f"{'='*60}\n")
    
    print("Accelerometer Statistics:")
    print(f"  Mean:   [{accel_data[:,0].mean():8.4f}, "
          f"{accel_data[:,1].mean():8.4f}, "
          f"{accel_data[:,2].mean():8.4f}] m/s²")
    print(f"  Std:    [{accel_data[:,0].std():8.4f}, "
          f"{accel_data[:,1].std():8.4f}, "
          f"{accel_data[:,2].std():8.4f}] m/s²")
    
    accel_magnitude = np.linalg.norm(accel_data, axis=1).mean()
    print(f"  Magnitude (mean): {accel_magnitude:.4f} m/s² (expected ~9.81)")
    
    print("\nGyroscope Statistics:")
    print(f"  Mean:   [{gyro_data[:,0].mean():8.4f}, "
          f"{gyro_data[:,1].mean():8.4f}, "
          f"{gyro_data[:,2].mean():8.4f}] rad/s")
    print(f"  Std:    [{gyro_data[:,0].std():8.4f}, "
          f"{gyro_data[:,1].std():8.4f}, "
          f"{gyro_data[:,2].std():8.4f}] rad/s")
    
    # Frame convention analysis
    print(f"\n{'='*60}")
    print("Frame Convention Analysis")
    print(f"{'='*60}\n")
    
    mean_accel = accel_data.mean(axis=0)
    dominant_axis = np.argmax(np.abs(mean_accel))
    axis_names = ['X', 'Y', 'Z']
    
    print(f"Dominant gravity axis: {axis_names[dominant_axis]}")
    print(f"Gravity sign in {axis_names[dominant_axis]}: "
          f"{'POSITIVE' if mean_accel[dominant_axis] > 0 else 'NEGATIVE'}")
    
    if dominant_axis == 2:  # Z axis
        if mean_accel[2] > 0:
            print("\n✅ CONCLUSION: IMU uses Z-UP convention (FLU/ENU compatible)")
            print("   No frame transformation needed for VINS-Fusion")
        else:
            print("\n⚠️  CONCLUSION: IMU uses Z-DOWN convention (NED-like)")
            print("   Frame transformation required for VINS-Fusion!")
            print("   Apply R_flu_ned = diag(1, -1, -1) to extrinsics")
    else:
        print(f"\n❓ UNUSUAL: Gravity dominant in {axis_names[dominant_axis]} axis")
        print("   Robot may not be level, or unusual mounting orientation")
    
    return {
        'accel_mean': mean_accel.tolist(),
        'gyro_mean': gyro_data.mean(axis=0).tolist(),
        'accel_magnitude': float(accel_magnitude),
        'dominant_axis': axis_names[dominant_axis],
        'z_positive': bool(mean_accel[2] > 0)
    }


def compare_imu_topics(bag_file):
    """Compare all IMU topics in the bag."""
    
    print(f"\n{'='*60}")
    print("Comparing All IMU Topics")
    print(f"{'='*60}\n")
    
    topics = [
        '/imu/imu_compensated',
        '/imu/imu_uncompensated', 
        '/ouster/imu'
    ]
    
    results = {}
    for topic in topics:
        print(f"\n--- {topic} ---")
        try:
            result = analyze_imu_data(bag_file, topic, num_samples=50)
            if result:
                results[topic] = result
        except Exception as e:
            print(f"Error: {e}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}\n")
    
    print(f"{'Topic':<30} {'Mean Accel Z':>12} {'Frame':<15}")
    print("-" * 60)
    for topic, result in results.items():
        frame = "Z-UP (FLU)" if result['z_positive'] else "Z-DOWN (NED)"
        print(f"{topic:<30} {result['accel_mean'][2]:>12.4f} {frame:<15}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze IMU data from ROS bag')
    parser.add_argument('bag_file', help='Path to bag file')
    parser.add_argument('--topic', default='/imu/imu_uncompensated',
                       help='IMU topic to analyze')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples to analyze')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all IMU topics')
    
    args = parser.parse_args()
    
    if args.compare_all:
        compare_imu_topics(args.bag_file)
    else:
        analyze_imu_data(args.bag_file, args.topic, args.samples)








