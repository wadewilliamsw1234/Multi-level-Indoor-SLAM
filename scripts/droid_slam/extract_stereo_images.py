#!/usr/bin/env python3
"""
Extract stereo images from ISEC ROS bags for DROID-SLAM processing.
DROID-SLAM expects image files, not ROS bag format.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def extract_images_from_bag(bag_path, output_dir, left_topic, right_topic):
    """Extract stereo images using rosbag and cv_bridge."""
    
    # This script will be run inside a ROS container
    extract_script = f'''
import rosbag
import cv2
from cv_bridge import CvBridge
import os

bag = rosbag.Bag("{bag_path}")
bridge = CvBridge()

left_dir = "{output_dir}/left"
right_dir = "{output_dir}/right"
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)

timestamps = []
for topic, msg, t in bag.read_messages(topics=["{left_topic}", "{right_topic}"]):
    timestamp = msg.header.stamp.to_sec()
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    if topic == "{left_topic}":
        cv2.imwrite(f"{{left_dir}}/{{timestamp:.6f}}.png", img)
    else:
        cv2.imwrite(f"{{right_dir}}/{{timestamp:.6f}}.png", img)
    timestamps.append(timestamp)

bag.close()
print(f"Extracted {{len(timestamps)//2}} stereo pairs")
'''
    return extract_script

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract stereo images from ROS bags")
    parser.add_argument("bag_dir", help="Directory containing bag files")
    parser.add_argument("output_dir", help="Output directory for images")
    parser.add_argument("--left-topic", default="/camera_array/cam1/image_raw")
    parser.add_argument("--right-topic", default="/camera_array/cam3/image_raw")
    args = parser.parse_args()
    
    print(f"Will extract from {args.bag_dir} to {args.output_dir}")
    print(f"Left: {args.left_topic}, Right: {args.right_topic}")
