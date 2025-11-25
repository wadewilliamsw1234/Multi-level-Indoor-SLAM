#!/usr/bin/env python
"""
Extract LeGO-LOAM trajectory from recorded ROS bag.
Python 2.7 compatible version for ROS Melodic.
"""

import argparse
import sys

try:
    import rosbag
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("Warning: ROS packages not available")


def extract_trajectory(bag_path, output_path, topic='/aft_mapped_to_init'):
    """Extract trajectory from LeGO-LOAM output bag."""
    if not HAS_ROS:
        print("ERROR: ROS packages required")
        return False
    
    try:
        bag = rosbag.Bag(bag_path, 'r')
    except Exception as e:
        print("ERROR: Could not open bag: {}".format(e))
        return False
    
    poses = []
    
    print("Reading trajectory from: {}".format(bag_path))
    print("Topic: {}".format(topic))
    
    topics = bag.get_type_and_topic_info().topics
    print("Available topics: {}".format(list(topics.keys())))
    
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
        print("ERROR: No odometry topic found. Available: {}".format(list(topics.keys())))
        bag.close()
        return False
    
    print("Using topic: {}".format(actual_topic))
    
    for _, msg, bag_time in bag.read_messages(topics=[actual_topic]):
        # FIX: Use message header timestamp, not bag recording time
        if hasattr(msg, 'header') and msg.header.stamp.to_sec() > 0:
            timestamp = msg.header.stamp.to_sec()
        else:
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
    
    bag.close()
    
    if not poses:
        print("ERROR: No poses extracted")
        return False
    
    poses.sort(key=lambda x: x[0])
    
    print("Writing {} poses to: {}".format(len(poses), output_path))
    print("Timestamp range: {:.3f} to {:.3f}".format(poses[0][0], poses[-1][0]))
    
    with open(output_path, 'w') as f:
        for pose in poses:
            f.write("{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*pose))
    
    print("Successfully extracted {} poses".format(len(poses)))
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract LeGO-LOAM trajectory')
    parser.add_argument('bag_path', help='Path to recorded bag file')
    parser.add_argument('output_path', help='Path to save TUM trajectory')
    parser.add_argument('--topic', default='/aft_mapped_to_init', help='Odometry topic name')
    
    args = parser.parse_args()
    success = extract_trajectory(args.bag_path, args.output_path, args.topic)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
