#!/usr/bin/env python3
"""
Extract LeGO-LOAM trajectory from recorded ROS bag.
LeGO-LOAM publishes odometry on /aft_mapped_to_init topic.
Saves poses in TUM format.
"""

import argparse
import sys
from pathlib import Path

try:
    import rosbag
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("Warning: ROS packages not available")


def extract_trajectory(bag_path: Path, output_path: Path, topic: str = '/aft_mapped_to_init'):
    """Extract trajectory from LeGO-LOAM output bag."""
    if not HAS_ROS:
        print("ERROR: ROS packages required")
        return False
    
    bag_path = Path(bag_path)
    output_path = Path(output_path)
    
    if not bag_path.exists():
        print(f"ERROR: Bag file not found: {bag_path}")
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    poses = []
    
    print(f"Reading trajectory from: {bag_path}")
    print(f"Topic: {topic}")
    
    with rosbag.Bag(str(bag_path), 'r') as bag:
        topics = bag.get_type_and_topic_info().topics
        print(f"Available topics: {list(topics.keys())}")
        
        possible_topics = [
            topic,
            '/aft_mapped_to_init',
            '/integrated_to_init', 
            '/laser_odom_to_init',
            '/odom',
        ]
        
        actual_topic = None
        for t in possible_topics:
            if t in topics:
                actual_topic = t
                break
        
        if actual_topic is None:
            print(f"ERROR: No odometry topic found. Available: {list(topics.keys())}")
            return False
        
        print(f"Using topic: {actual_topic}")
        
        for _, msg, bag_time in bag.read_messages(topics=[actual_topic]):
            # FIX: Use message header timestamp, not bag recording time
            # This ensures we get the original sensor timestamp even if
            # the bag was recorded without /use_sim_time
            if hasattr(msg, 'header') and msg.header.stamp.to_sec() > 0:
                timestamp = msg.header.stamp.to_sec()
            else:
                # Fallback to bag time if no valid header
                timestamp = bag_time.to_sec()
            
            # Handle both Odometry and PoseStamped messages
            if hasattr(msg, 'pose'):
                if hasattr(msg.pose, 'pose'):
                    pose = msg.pose.pose  # Odometry message
                else:
                    pose = msg.pose  # PoseStamped message
            else:
                continue
            
            tx = pose.position.x
            ty = pose.position.y
            tz = pose.position.z
            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            qw = pose.orientation.w
            
            poses.append((timestamp, tx, ty, tz, qx, qy, qz, qw))
    
    if not poses:
        print("ERROR: No poses extracted")
        return False
    
    poses.sort(key=lambda x: x[0])
    
    print(f"Writing {len(poses)} poses to: {output_path}")
    print(f"Timestamp range: {poses[0][0]:.3f} to {poses[-1][0]:.3f}")
    
    with open(output_path, 'w') as f:
        for pose in poses:
            f.write(f"{pose[0]:.9f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} "
                   f"{pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f} {pose[7]:.6f}\n")
    
    print(f"Successfully extracted {len(poses)} poses")
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract LeGO-LOAM trajectory')
    parser.add_argument('bag_path', type=Path, help='Path to recorded bag file')
    parser.add_argument('output_path', type=Path, help='Path to save TUM trajectory')
    parser.add_argument('--topic', default='/aft_mapped_to_init', help='Odometry topic name')
    
    args = parser.parse_args()
    success = extract_trajectory(args.bag_path, args.output_path, args.topic)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
