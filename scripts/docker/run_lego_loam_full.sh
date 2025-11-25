#!/bin/bash
source /opt/ros/melodic/setup.bash
source /catkin_ws/devel/setup.bash

SEQUENCE=$1
if [ -z "$SEQUENCE" ]; then
    SEQUENCE="5th_floor"
fi

echo "=== Running LeGO-LOAM on $SEQUENCE ==="

mkdir -p /results/trajectories/lego_loam

roscore &
sleep 3

rosparam set /use_sim_time true

# Start nodes with unique names
rosrun tf static_transform_publisher 0 0 0 0 0 0 map camera_init 100 __name:=tf_map_camera &
rosrun tf static_transform_publisher 0 0 0 -1.5707963 0 -1.5707963 camera base_link 100 __name:=tf_camera_base &
sleep 1

rosrun lego_loam imageProjection __name:=imageProjection &
sleep 2
rosrun lego_loam featureAssociation __name:=featureAssociation &
sleep 2
rosrun lego_loam mapOptmization __name:=mapOptmization &
sleep 2
rosrun lego_loam transformFusion __name:=transformFusion &
sleep 3

echo "=== Nodes started ==="
rosnode list

# Record odometry
rosbag record -O /results/lego_loam_${SEQUENCE}.bag /aft_mapped_to_init /laser_odom_to_init __name:=bag_recorder &
sleep 2

# Play ALL bags in sequence
echo "=== Playing all bags in /data/ISEC/${SEQUENCE}/ ==="
BAG_FILES=$(ls /data/ISEC/${SEQUENCE}/*.bag 2>/dev/null | sort)
echo "Found bags:"
echo "$BAG_FILES"
echo ""

rosbag play --clock $BAG_FILES

echo "=== Playback complete ==="
sleep 3

rosnode kill /bag_recorder 2>/dev/null
sleep 2

echo ""
echo "=== Output bag info ==="
rosbag info /results/lego_loam_${SEQUENCE}.bag

# Extract trajectory
echo ""
echo "=== Extracting trajectory ==="
python /scripts/extract_trajectory.py /results/lego_loam_${SEQUENCE}.bag /results/trajectories/lego_loam/${SEQUENCE}.txt

killall -9 roslaunch rosrun roscore rosbag 2>/dev/null
echo ""
echo "=== Done! ==="
