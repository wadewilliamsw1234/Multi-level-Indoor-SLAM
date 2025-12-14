#!/usr/bin/env python3
"""
ROS Node: NED to FLU IMU Transform

Transforms VectorNav IMU data from NED (North-East-Down) convention 
to FLU (Forward-Left-Up) convention for VINS-Fusion.

NED Convention (VectorNav output):
- X: North/Forward
- Y: East/Right
- Z: Down
- Gravity: [0, 0, +9.81] when stationary (accelerometer reads +g in z)
- Actually VectorNav reports: [0, 0, -9.81] (it reports acceleration, not specific force)

FLU Convention (VINS-Fusion expected):
- X: Forward
- Y: Left
- Z: Up
- Gravity: [0, 0, +9.81] when stationary (accelerometer reads +g in z pointing up)

Transform:
- x_flu = x_ned
- y_flu = -y_ned
- z_flu = -z_ned

This is equivalent to a 180° rotation around the x-axis.

Usage:
    rosrun slam_benchmark imu_ned_to_flu.py

Subscribes:
    /imu/imu_compensated (sensor_msgs/Imu) - NED convention

Publishes:
    /imu/imu_flu (sensor_msgs/Imu) - FLU convention

Author: SLAM Benchmarking Project
"""

import rospy
from sensor_msgs.msg import Imu
import numpy as np


class ImuNedToFlu:
    def __init__(self):
        rospy.init_node('imu_ned_to_flu', anonymous=True)
        
        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/imu/imu_compensated')
        self.output_topic = rospy.get_param('~output_topic', '/imu/imu_flu')
        
        # Publisher
        self.pub = rospy.Publisher(self.output_topic, Imu, queue_size=100)
        
        # Subscriber
        self.sub = rospy.Subscriber(self.input_topic, Imu, self.imu_callback)
        
        # Statistics for debugging
        self.msg_count = 0
        self.last_print_time = rospy.Time.now()
        
        rospy.loginfo(f"IMU NED→FLU Transform Node Started")
        rospy.loginfo(f"  Input:  {self.input_topic}")
        rospy.loginfo(f"  Output: {self.output_topic}")
        rospy.loginfo(f"  Transform: x_flu=x_ned, y_flu=-y_ned, z_flu=-z_ned")
    
    def imu_callback(self, msg_ned: Imu):
        """Transform IMU message from NED to FLU."""
        msg_flu = Imu()
        
        # Copy header
        msg_flu.header = msg_ned.header
        
        # Transform linear acceleration: [x, -y, -z]
        msg_flu.linear_acceleration.x = msg_ned.linear_acceleration.x
        msg_flu.linear_acceleration.y = -msg_ned.linear_acceleration.y
        msg_flu.linear_acceleration.z = -msg_ned.linear_acceleration.z
        
        # Transform angular velocity: [x, -y, -z]
        msg_flu.angular_velocity.x = msg_ned.angular_velocity.x
        msg_flu.angular_velocity.y = -msg_ned.angular_velocity.y
        msg_flu.angular_velocity.z = -msg_ned.angular_velocity.z
        
        # Transform orientation quaternion
        # For a 180° rotation around x-axis: q_new = q_rot * q_old
        # where q_rot = [1, 0, 0, 0] (180° around x)
        # Actually simpler: q_flu = [w, x, -y, -z]
        msg_flu.orientation.w = msg_ned.orientation.w
        msg_flu.orientation.x = msg_ned.orientation.x
        msg_flu.orientation.y = -msg_ned.orientation.y
        msg_flu.orientation.z = -msg_ned.orientation.z
        
        # Copy covariances (assuming they're diagonal and axis-aligned, 
        # the transformation doesn't change variances)
        msg_flu.linear_acceleration_covariance = msg_ned.linear_acceleration_covariance
        msg_flu.angular_velocity_covariance = msg_ned.angular_velocity_covariance
        msg_flu.orientation_covariance = msg_ned.orientation_covariance
        
        # Publish
        self.pub.publish(msg_flu)
        
        # Debug output (every 5 seconds)
        self.msg_count += 1
        now = rospy.Time.now()
        if (now - self.last_print_time).to_sec() > 5.0:
            rospy.loginfo(f"Processed {self.msg_count} IMU messages")
            rospy.loginfo(f"  NED accel: [{msg_ned.linear_acceleration.x:.3f}, "
                         f"{msg_ned.linear_acceleration.y:.3f}, "
                         f"{msg_ned.linear_acceleration.z:.3f}]")
            rospy.loginfo(f"  FLU accel: [{msg_flu.linear_acceleration.x:.3f}, "
                         f"{msg_flu.linear_acceleration.y:.3f}, "
                         f"{msg_flu.linear_acceleration.z:.3f}]")
            self.last_print_time = now


def main():
    try:
        node = ImuNedToFlu()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()







