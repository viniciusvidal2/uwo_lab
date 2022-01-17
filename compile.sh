#!/bin/bash

# cd workspace directory and source ROS dependencies
cd $HOME/uwo_ws
source /opt/ros/melodic/setup.bash
source devel/setup.bash

# Go through all the packages, sleep in between
catkin_make --pkg livox_ros_driver
sleep 1s

catkin_make --pkg zed_ros
sleep 1s

catkin_make --pkg zed_wrapper zed_interfaces zed_nodelets
sleep 1s

catkin_make --pkg realsense2_camera orb_slam2_ros
sleep 1s

catkin_make --pkg fast_lio 
sleep 1s

catkin_make --pkg uwo_pack 
sleep 1s

catkin_make --pkg aloam_velodyne
