#!/usr/bin/env python3
"""
Extract ISEC bag data to EuRoC format for Basalt processing.

EuRoC format structure:
  sequence/
    mav0/
      cam0/
        data/
          timestamp.png
        data.csv
      cam1/
        data/
          timestamp.png
        data.csv
      imu0/
        data.csv
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np


def extract_to_euroc(bag_files, output_dir, cam0_topic, cam1_topic, imu_topic, 
                     max_frames=None, skip_frames=1):
    """
    Extract bag data to EuRoC format.
    
    Args:
        bag_files: List of bag file paths
        output_dir: Output directory
        cam0_topic: Left camera topic
        cam1_topic: Right camera topic  
        imu_topic: IMU topic
        max_frames: Maximum frames to extract (None for all)
        skip_frames: Extract every Nth frame (1 = all frames)
    """
    output_dir = Path(output_dir)
    
    # Create directory structure
    cam0_dir = output_dir / "mav0" / "cam0" / "data"
    cam1_dir = output_dir / "mav0" / "cam1" / "data"
    imu_dir = output_dir / "mav0" / "imu0"
    
    cam0_dir.mkdir(parents=True, exist_ok=True)
    cam1_dir.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)
    
    bridge = CvBridge()
    
    # Data storage
    cam0_timestamps = []
    cam1_timestamps = []
    imu_data = []
    
    frame_count = 0
    total_frames = 0
    
    # Process all bags
    for bag_file in bag_files:
        print(f"Processing: {bag_file.name}")
        
        with rosbag.Bag(str(bag_file), 'r') as bag:
            # Get message counts for progress bar
            info = bag.get_type_and_topic_info()
            topics_info = info.topics
            
            for topic, msg, t in tqdm(bag.read_messages(
                topics=[cam0_topic, cam1_topic, imu_topic])):
                
                timestamp_ns = int(t.to_nsec())
                
                if topic == imu_topic:
                    # Extract IMU data
                    # EuRoC format: timestamp[ns],w_x,w_y,w_z,a_x,a_y,a_z
                    imu_data.append([
                        timestamp_ns,
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z
                    ])
                    
                elif topic == cam0_topic:
                    total_frames += 1
                    if total_frames % skip_frames != 0:
                        continue
                        
                    # Extract left camera image
                    try:
                        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                        img_path = cam0_dir / f"{timestamp_ns}.png"
                        cv2.imwrite(str(img_path), cv_img)
                        cam0_timestamps.append(timestamp_ns)
                        frame_count += 1
                    except Exception as e:
                        print(f"Error converting cam0 image: {e}")
                        
                elif topic == cam1_topic:
                    if total_frames % skip_frames != 0:
                        continue
                        
                    # Extract right camera image
                    try:
                        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                        img_path = cam1_dir / f"{timestamp_ns}.png"
                        cv2.imwrite(str(img_path), cv_img)
                        cam1_timestamps.append(timestamp_ns)
                    except Exception as e:
                        print(f"Error converting cam1 image: {e}")
                
                if max_frames and frame_count >= max_frames:
                    break
            
            if max_frames and frame_count >= max_frames:
                break
    
    # Write CSV files
    print("Writing CSV files...")
    
    # Camera timestamps
    with open(output_dir / "mav0" / "cam0" / "data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['#timestamp [ns]', 'filename'])
        for ts in sorted(cam0_timestamps):
            writer.writerow([ts, f"{ts}.png"])
    
    with open(output_dir / "mav0" / "cam1" / "data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['#timestamp [ns]', 'filename'])
        for ts in sorted(cam1_timestamps):
            writer.writerow([ts, f"{ts}.png"])
    
    # IMU data
    with open(imu_dir / "data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['#timestamp [ns]', 'w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 
                        'w_RS_S_z [rad s^-1]', 'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 
                        'a_RS_S_z [m s^-2]'])
        for row in sorted(imu_data, key=lambda x: x[0]):
            writer.writerow(row)
    
    print(f"\nExtraction complete:")
    print(f"  - cam0 images: {len(cam0_timestamps)}")
    print(f"  - cam1 images: {len(cam1_timestamps)}")
    print(f"  - IMU samples: {len(imu_data)}")
    print(f"  - Output: {output_dir}")
    
    return len(cam0_timestamps)


def main():
    parser = argparse.ArgumentParser(
        description='Extract ISEC bags to EuRoC format for Basalt')
    parser.add_argument('--floor', type=str, required=True,
                       help='Floor sequence (e.g., 5th_floor)')
    parser.add_argument('--data-dir', type=str, default='/data/ISEC',
                       help='ISEC dataset directory')
    parser.add_argument('--output-dir', type=str, default='/data/euroc',
                       help='Output directory for EuRoC format data')
    parser.add_argument('--cam0-topic', type=str, 
                       default='/camera_array/cam1/image_raw',
                       help='Left camera topic')
    parser.add_argument('--cam1-topic', type=str,
                       default='/camera_array/cam3/image_raw', 
                       help='Right camera topic')
    parser.add_argument('--imu-topic', type=str,
                       default='/vectornav/imu',
                       help='IMU topic')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to extract')
    parser.add_argument('--skip', type=int, default=1,
                       help='Extract every Nth frame')
    
    args = parser.parse_args()
    
    # Find bag files
    floor_dir = Path(args.data_dir) / args.floor
    if not floor_dir.exists():
        print(f"Error: Floor directory not found: {floor_dir}")
        sys.exit(1)
    
    bag_files = sorted(floor_dir.glob("*.bag"))
    if not bag_files:
        print(f"Error: No bag files found in {floor_dir}")
        sys.exit(1)
    
    print(f"Found {len(bag_files)} bag files for {args.floor}")
    
    # Output directory
    output_dir = Path(args.output_dir) / args.floor
    
    # Extract
    extract_to_euroc(
        bag_files,
        output_dir,
        args.cam0_topic,
        args.cam1_topic,
        args.imu_topic,
        args.max_frames,
        args.skip
    )


if __name__ == '__main__':
    main()
