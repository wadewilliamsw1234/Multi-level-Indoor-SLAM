#!/usr/bin/env python
"""Extract trajectory from LeGO-LOAM output bag to TUM format"""
import rosbag
import sys

bag_path = sys.argv[1] if len(sys.argv) > 1 else '/results/lego_loam_odom.bag'
output_path = sys.argv[2] if len(sys.argv) > 2 else '/results/trajectories/lego_loam/5th_floor.txt'
topic = '/aft_mapped_to_init'

print("Reading: " + bag_path)
print("Topic: " + topic)
print("Output: " + output_path)

bag = rosbag.Bag(bag_path)
poses = []

for _, msg, t in bag.read_messages(topics=[topic]):
    ts = msg.header.stamp.to_sec()
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    poses.append((ts, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

bag.close()

# Sort by timestamp
poses.sort(key=lambda x: x[0])

# Write TUM format
with open(output_path, 'w') as f:
    f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
    for pose in poses:
        f.write("{:.9f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*pose))

print("Extracted {} poses".format(len(poses)))
print("Done!")
