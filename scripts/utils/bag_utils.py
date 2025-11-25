#!/usr/bin/env python3
"""
ROS Bag Utilities
=================

Utilities for handling ROS bag files from the ISEC dataset.
Supports reading, merging, and extracting data from bags.
"""

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import yaml

# ROS imports - handle gracefully if not available
try:
    import rosbag
    from cv_bridge import CvBridge
    import cv2
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("Warning: ROS packages not available. Some functions will be disabled.")


@dataclass
class BagInfo:
    """Information about a ROS bag file."""
    path: Path
    duration: float
    start_time: float
    end_time: float
    message_count: int
    topics: Dict[str, dict]
    size_bytes: int
    
    def __str__(self) -> str:
        size_gb = self.size_bytes / (1024**3)
        return (
            f"Bag: {self.path.name}\n"
            f"  Duration: {self.duration:.2f}s\n"
            f"  Size: {size_gb:.2f} GB\n"
            f"  Messages: {self.message_count}\n"
            f"  Topics: {len(self.topics)}"
        )


# Topic names from ISEC dataset
ISEC_TOPICS = {
    'cam0': '/camera_array/cam0/image_raw',
    'cam1': '/camera_array/cam1/image_raw',
    'cam2': '/camera_array/cam2/image_raw',
    'cam3': '/camera_array/cam3/image_raw',
    'cam4': '/camera_array/cam4/image_raw',
    'cam5': '/camera_array/cam5/image_raw',
    'cam6': '/camera_array/cam6/image_raw',
    'imu': '/vectornav/imu',
    'lidar': '/ouster/points',
}

# Stereo pair for evaluation (as per paper Section V-A)
STEREO_PAIR = ('cam1', 'cam3')


def get_bag_info(bag_path: Path) -> BagInfo:
    """
    Extract information about a ROS bag file.
    
    Args:
        bag_path: Path to the bag file
        
    Returns:
        BagInfo object with bag metadata
    """
    if not HAS_ROS:
        raise RuntimeError("ROS packages not available")
    
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    
    with rosbag.Bag(str(bag_path), 'r') as bag:
        info = bag.get_type_and_topic_info()
        
        topics = {}
        for topic, topic_info in info.topics.items():
            topics[topic] = {
                'msg_type': topic_info.msg_type,
                'message_count': topic_info.message_count,
                'frequency': topic_info.frequency if topic_info.frequency else 0,
            }
        
        return BagInfo(
            path=bag_path,
            duration=bag.get_end_time() - bag.get_start_time(),
            start_time=bag.get_start_time(),
            end_time=bag.get_end_time(),
            message_count=bag.get_message_count(),
            topics=topics,
            size_bytes=bag_path.stat().st_size,
        )


def find_sequence_bags(sequence_dir: Path) -> List[Path]:
    """
    Find all bag files for a sequence and return them in order.
    
    Args:
        sequence_dir: Directory containing bag files for a sequence
        
    Returns:
        List of bag file paths sorted by name
    """
    sequence_dir = Path(sequence_dir)
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")
    
    bags = sorted(sequence_dir.glob("*.bag"))
    if not bags:
        raise FileNotFoundError(f"No bag files found in: {sequence_dir}")
    
    return bags


def get_sequence_info(sequence_dir: Path) -> Dict:
    """
    Get combined information for all bags in a sequence.
    
    Args:
        sequence_dir: Directory containing bag files
        
    Returns:
        Dictionary with combined sequence information
    """
    bags = find_sequence_bags(sequence_dir)
    
    total_duration = 0
    total_size = 0
    total_messages = 0
    all_topics = set()
    
    for bag_path in bags:
        info = get_bag_info(bag_path)
        total_duration += info.duration
        total_size += info.size_bytes
        total_messages += info.message_count
        all_topics.update(info.topics.keys())
    
    return {
        'sequence_name': sequence_dir.name,
        'num_bags': len(bags),
        'total_duration': total_duration,
        'total_size_gb': total_size / (1024**3),
        'total_messages': total_messages,
        'topics': sorted(all_topics),
        'bag_files': [b.name for b in bags],
    }


def iter_bag_messages(
    bag_path: Path,
    topics: Optional[List[str]] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Generator[Tuple[str, any, float], None, None]:
    """
    Iterate over messages in a bag file.
    
    Args:
        bag_path: Path to the bag file
        topics: List of topics to read (None for all)
        start_time: Start time filter (ROS time)
        end_time: End time filter (ROS time)
        
    Yields:
        Tuple of (topic, message, timestamp)
    """
    if not HAS_ROS:
        raise RuntimeError("ROS packages not available")
    
    import rospy
    
    with rosbag.Bag(str(bag_path), 'r') as bag:
        start = rospy.Time.from_sec(start_time) if start_time else None
        end = rospy.Time.from_sec(end_time) if end_time else None
        
        for topic, msg, t in bag.read_messages(
            topics=topics,
            start_time=start,
            end_time=end
        ):
            yield topic, msg, t.to_sec()


def iter_sequence_messages(
    sequence_dir: Path,
    topics: Optional[List[str]] = None,
) -> Generator[Tuple[str, any, float], None, None]:
    """
    Iterate over messages across all bags in a sequence.
    
    Args:
        sequence_dir: Directory containing bag files
        topics: List of topics to read (None for all)
        
    Yields:
        Tuple of (topic, message, timestamp)
    """
    bags = find_sequence_bags(sequence_dir)
    
    for bag_path in bags:
        yield from iter_bag_messages(bag_path, topics=topics)
        gc.collect()  # Free memory between bags


def extract_images(
    bag_path: Path,
    topic: str,
    output_dir: Path,
    image_format: str = 'png',
    max_images: Optional[int] = None,
) -> List[Tuple[float, Path]]:
    """
    Extract images from a bag file to disk.
    
    Args:
        bag_path: Path to the bag file
        topic: Image topic to extract
        output_dir: Directory to save images
        image_format: Output image format ('png' or 'jpg')
        max_images: Maximum number of images to extract (None for all)
        
    Returns:
        List of (timestamp, image_path) tuples
    """
    if not HAS_ROS:
        raise RuntimeError("ROS packages not available")
    
    bridge = CvBridge()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = []
    count = 0
    
    for _, msg, timestamp in iter_bag_messages(bag_path, topics=[topic]):
        if max_images and count >= max_images:
            break
        
        try:
            # Convert ROS image to OpenCV
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Save image
            filename = f"{timestamp:.6f}.{image_format}"
            image_path = output_dir / filename
            cv2.imwrite(str(image_path), cv_image)
            
            extracted.append((timestamp, image_path))
            count += 1
            
        except Exception as e:
            print(f"Warning: Failed to extract image at {timestamp}: {e}")
    
    return extracted


def extract_stereo_images(
    bag_path: Path,
    output_dir: Path,
    left_topic: str = None,
    right_topic: str = None,
    image_format: str = 'png',
    max_pairs: Optional[int] = None,
    time_tolerance: float = 0.01,
) -> List[Tuple[float, Path, Path]]:
    """
    Extract synchronized stereo image pairs from a bag file.
    
    Uses cam1 and cam3 as the stereo pair by default (as specified in paper).
    
    Args:
        bag_path: Path to the bag file
        output_dir: Directory to save images
        left_topic: Left camera topic (default: cam1)
        right_topic: Right camera topic (default: cam3)
        image_format: Output image format
        max_pairs: Maximum number of pairs to extract
        time_tolerance: Maximum time difference for sync (seconds)
        
    Returns:
        List of (timestamp, left_path, right_path) tuples
    """
    if not HAS_ROS:
        raise RuntimeError("ROS packages not available")
    
    left_topic = left_topic or ISEC_TOPICS['cam1']
    right_topic = right_topic or ISEC_TOPICS['cam3']
    
    bridge = CvBridge()
    output_dir = Path(output_dir)
    left_dir = output_dir / 'left'
    right_dir = output_dir / 'right'
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    
    # Buffer for synchronization
    left_buffer = {}
    right_buffer = {}
    extracted = []
    
    for topic, msg, timestamp in iter_bag_messages(
        bag_path, 
        topics=[left_topic, right_topic]
    ):
        if max_pairs and len(extracted) >= max_pairs:
            break
        
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if topic == left_topic:
                left_buffer[timestamp] = cv_image
                # Try to find matching right image
                for rt in list(right_buffer.keys()):
                    if abs(timestamp - rt) <= time_tolerance:
                        # Found match
                        left_path = left_dir / f"{timestamp:.6f}.{image_format}"
                        right_path = right_dir / f"{rt:.6f}.{image_format}"
                        
                        cv2.imwrite(str(left_path), cv_image)
                        cv2.imwrite(str(right_path), right_buffer[rt])
                        
                        extracted.append((timestamp, left_path, right_path))
                        del right_buffer[rt]
                        del left_buffer[timestamp]
                        break
            else:
                right_buffer[timestamp] = cv_image
                # Try to find matching left image
                for lt in list(left_buffer.keys()):
                    if abs(timestamp - lt) <= time_tolerance:
                        # Found match
                        left_path = left_dir / f"{lt:.6f}.{image_format}"
                        right_path = right_dir / f"{timestamp:.6f}.{image_format}"
                        
                        cv2.imwrite(str(left_path), left_buffer[lt])
                        cv2.imwrite(str(right_path), cv_image)
                        
                        extracted.append((lt, left_path, right_path))
                        del left_buffer[lt]
                        del right_buffer[timestamp]
                        break
            
            # Clean old buffer entries
            current_time = timestamp
            left_buffer = {t: img for t, img in left_buffer.items() 
                          if current_time - t < 1.0}
            right_buffer = {t: img for t, img in right_buffer.items() 
                           if current_time - t < 1.0}
            
        except Exception as e:
            print(f"Warning: Failed to process image at {timestamp}: {e}")
    
    return extracted


def extract_imu_data(
    bag_path: Path,
    output_path: Path,
    topic: str = None,
) -> int:
    """
    Extract IMU data from a bag file to CSV.
    
    Args:
        bag_path: Path to the bag file
        output_path: Path for output CSV file
        topic: IMU topic (default: /vectornav/imu)
        
    Returns:
        Number of IMU messages extracted
    """
    if not HAS_ROS:
        raise RuntimeError("ROS packages not available")
    
    topic = topic or ISEC_TOPICS['imu']
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for _, msg, timestamp in iter_bag_messages(bag_path, topics=[topic]):
        data.append({
            'timestamp': timestamp,
            'wx': msg.angular_velocity.x,
            'wy': msg.angular_velocity.y,
            'wz': msg.angular_velocity.z,
            'ax': msg.linear_acceleration.x,
            'ay': msg.linear_acceleration.y,
            'az': msg.linear_acceleration.z,
        })
    
    # Write to CSV
    with open(output_path, 'w') as f:
        f.write('timestamp,wx,wy,wz,ax,ay,az\n')
        for d in data:
            f.write(f"{d['timestamp']:.9f},{d['wx']},{d['wy']},{d['wz']},"
                   f"{d['ax']},{d['ay']},{d['az']}\n")
    
    return len(data)


def create_timestamps_file(
    bag_path: Path,
    topic: str,
    output_path: Path,
) -> int:
    """
    Create a timestamps file for a topic (useful for offline processing).
    
    Args:
        bag_path: Path to the bag file
        topic: Topic to extract timestamps from
        output_path: Path for output file
        
    Returns:
        Number of timestamps written
    """
    if not HAS_ROS:
        raise RuntimeError("ROS packages not available")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamps = []
    for _, _, timestamp in iter_bag_messages(bag_path, topics=[topic]):
        timestamps.append(timestamp)
    
    with open(output_path, 'w') as f:
        for ts in timestamps:
            f.write(f"{ts:.9f}\n")
    
    return len(timestamps)


def test_bag_access(data_dir: Path) -> bool:
    """
    Test that bag files can be accessed and read.
    
    Args:
        data_dir: Path to ISEC data directory
        
    Returns:
        True if bags are accessible
    """
    data_dir = Path(data_dir)
    
    print(f"Testing bag access in: {data_dir}")
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return False
    
    # Look for any sequence directory
    sequences = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']
    
    found_bags = False
    for seq in sequences:
        seq_dir = data_dir / seq
        if seq_dir.exists():
            try:
                bags = find_sequence_bags(seq_dir)
                print(f"Found {len(bags)} bags in {seq}")
                
                # Try to read first bag
                if bags and HAS_ROS:
                    info = get_bag_info(bags[0])
                    print(f"  First bag: {info}")
                    found_bags = True
                    break
            except Exception as e:
                print(f"  Error reading {seq}: {e}")
    
    if not found_bags:
        print("WARNING: No bag files found or readable")
        return False
    
    print("Bag access test PASSED")
    return True


def main():
    """Command-line interface for bag utilities."""
    parser = argparse.ArgumentParser(description='ROS Bag Utilities')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get bag file info')
    info_parser.add_argument('bag_path', type=Path, help='Path to bag file')
    
    # Sequence info command
    seq_parser = subparsers.add_parser('sequence-info', help='Get sequence info')
    seq_parser.add_argument('sequence_dir', type=Path, help='Path to sequence directory')
    
    # Extract images command
    extract_parser = subparsers.add_parser('extract-images', help='Extract images')
    extract_parser.add_argument('bag_path', type=Path, help='Path to bag file')
    extract_parser.add_argument('--topic', default=ISEC_TOPICS['cam1'], help='Image topic')
    extract_parser.add_argument('--output', '-o', type=Path, required=True, help='Output directory')
    extract_parser.add_argument('--max', type=int, help='Maximum images to extract')
    
    # Extract stereo command
    stereo_parser = subparsers.add_parser('extract-stereo', help='Extract stereo pairs')
    stereo_parser.add_argument('bag_path', type=Path, help='Path to bag file')
    stereo_parser.add_argument('--output', '-o', type=Path, required=True, help='Output directory')
    stereo_parser.add_argument('--max', type=int, help='Maximum pairs to extract')
    
    # Extract IMU command
    imu_parser = subparsers.add_parser('extract-imu', help='Extract IMU data')
    imu_parser.add_argument('bag_path', type=Path, help='Path to bag file')
    imu_parser.add_argument('--output', '-o', type=Path, required=True, help='Output CSV file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test bag access')
    test_parser.add_argument('--data-dir', type=Path, default=Path('/data/ISEC'),
                            help='Path to ISEC data directory')
    
    args = parser.parse_args()
    
    if args.command == 'info':
        info = get_bag_info(args.bag_path)
        print(info)
        print("\nTopics:")
        for topic, topic_info in info.topics.items():
            print(f"  {topic}: {topic_info['message_count']} msgs @ {topic_info['frequency']:.1f} Hz")
    
    elif args.command == 'sequence-info':
        info = get_sequence_info(args.sequence_dir)
        print(f"Sequence: {info['sequence_name']}")
        print(f"  Bags: {info['num_bags']}")
        print(f"  Duration: {info['total_duration']:.1f}s")
        print(f"  Size: {info['total_size_gb']:.2f} GB")
        print(f"  Messages: {info['total_messages']}")
        print(f"  Topics: {len(info['topics'])}")
    
    elif args.command == 'extract-images':
        print(f"Extracting images from {args.bag_path}...")
        extracted = extract_images(
            args.bag_path, 
            args.topic, 
            args.output,
            max_images=args.max
        )
        print(f"Extracted {len(extracted)} images to {args.output}")
    
    elif args.command == 'extract-stereo':
        print(f"Extracting stereo pairs from {args.bag_path}...")
        extracted = extract_stereo_images(
            args.bag_path,
            args.output,
            max_pairs=args.max
        )
        print(f"Extracted {len(extracted)} stereo pairs to {args.output}")
    
    elif args.command == 'extract-imu':
        print(f"Extracting IMU data from {args.bag_path}...")
        count = extract_imu_data(args.bag_path, args.output)
        print(f"Extracted {count} IMU messages to {args.output}")
    
    elif args.command == 'test':
        success = test_bag_access(args.data_dir)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
