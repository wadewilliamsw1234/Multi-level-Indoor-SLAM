#!/usr/bin/env python3
"""
Run VINS-Fusion on ISEC dataset and extract trajectory in TUM format.
"""

import os
import sys
import subprocess
import time
import signal
import argparse
from pathlib import Path
from threading import Thread, Event
import queue

# ROS imports (available in container)
import rospy
import rosbag
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf.transformations as tf_trans


class TrajectoryRecorder:
    """Records odometry messages and saves to TUM format."""
    
    def __init__(self, output_file):
        self.output_file = output_file
        self.poses = []
        self.recording = True
        
    def callback(self, msg):
        if not self.recording:
            return
            
        # Extract timestamp
        t = msg.header.stamp.to_sec()
        
        # Extract position
        p = msg.pose.pose.position
        tx, ty, tz = p.x, p.y, p.z
        
        # Extract orientation (quaternion)
        q = msg.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        
        self.poses.append((t, tx, ty, tz, qx, qy, qz, qw))
        
    def save(self):
        """Save trajectory to TUM format file."""
        self.recording = False
        
        with open(self.output_file, 'w') as f:
            for pose in self.poses:
                f.write(f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} "
                       f"{pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f} {pose[7]:.6f}\n")
        
        print(f"Saved {len(self.poses)} poses to {self.output_file}")
        return len(self.poses)


def find_bag_files(data_dir, floor):
    """Find all bag files for a floor sequence."""
    floor_dir = Path(data_dir) / floor
    if not floor_dir.exists():
        raise FileNotFoundError(f"Floor directory not found: {floor_dir}")
    
    bags = sorted(floor_dir.glob("*.bag"))
    if not bags:
        raise FileNotFoundError(f"No bag files found in {floor_dir}")
    
    return bags


def run_vins_fusion(config_file, bag_files, output_file, playback_rate=0.5):
    """
    Run VINS-Fusion on bag files and record trajectory.
    
    Args:
        config_file: Path to VINS-Fusion config YAML
        bag_files: List of bag file paths
        output_file: Output trajectory file (TUM format)
        playback_rate: Bag playback rate (default 0.5x for processing headroom)
    """
    processes = []
    
    try:
        # Initialize ROS node
        rospy.init_node('vins_runner', anonymous=True)
        
        # Start VINS-Fusion estimator
        print("Starting VINS-Fusion estimator...")
        vins_proc = subprocess.Popen(
            ['rosrun', 'vins', 'vins_node', config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(vins_proc)
        time.sleep(3)  # Wait for initialization
        
        # Setup trajectory recorder
        recorder = TrajectoryRecorder(output_file)
        rospy.Subscriber('/vins_estimator/odometry', Odometry, recorder.callback)
        
        # Also try camera pose topic
        rospy.Subscriber('/vins_estimator/camera_pose', Odometry, recorder.callback)
        
        # Play bag files
        total_bags = len(bag_files)
        for i, bag_file in enumerate(bag_files):
            print(f"Playing bag {i+1}/{total_bags}: {bag_file.name}")
            
            play_proc = subprocess.Popen(
                ['rosbag', 'play', str(bag_file), '--clock', '-r', str(playback_rate)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            play_proc.wait()
            
            # Brief pause between bags
            time.sleep(1)
        
        # Wait for final processing
        print("Waiting for final processing...")
        time.sleep(5)
        
        # Save trajectory
        num_poses = recorder.save()
        
        if num_poses == 0:
            print("WARNING: No poses recorded! Check topic names and VINS-Fusion initialization.")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
        
    finally:
        # Cleanup processes
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()


def main():
    parser = argparse.ArgumentParser(description='Run VINS-Fusion on ISEC dataset')
    parser.add_argument('--floor', type=str, required=True,
                       help='Floor sequence (e.g., 5th_floor, 1st_floor)')
    parser.add_argument('--config', type=str, 
                       default='/config/vins_fusion/isec_stereo_imu.yaml',
                       help='VINS-Fusion config file')
    parser.add_argument('--data-dir', type=str, default='/data/ISEC',
                       help='ISEC dataset directory')
    parser.add_argument('--output-dir', type=str, 
                       default='/results/trajectories/vins_fusion',
                       help='Output directory for trajectories')
    parser.add_argument('--rate', type=float, default=0.5,
                       help='Bag playback rate')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.floor}.txt"
    
    # Find bag files
    try:
        bag_files = find_bag_files(args.data_dir, args.floor)
        print(f"Found {len(bag_files)} bag files for {args.floor}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Run VINS-Fusion
    print(f"\n{'='*50}")
    print(f"VINS-Fusion: Processing {args.floor}")
    print(f"{'='*50}\n")
    
    success = run_vins_fusion(
        args.config, 
        bag_files, 
        str(output_file),
        args.rate
    )
    
    if success:
        print(f"\n{'='*50}")
        print(f"VINS-Fusion: {args.floor} complete")
        print(f"Output: {output_file}")
        print(f"{'='*50}\n")
    else:
        print(f"\nVINS-Fusion failed on {args.floor}")
        sys.exit(1)


if __name__ == '__main__':
    main()
