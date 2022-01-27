#!/bin/bash

# cd workspace directory and source ROS dependencies
cd $HOME/uwo_ws
source /opt/ros/melodic/setup.bash
source devel/setup.bash

# Go through all the packages, sleep in between
catkin_make -DCMAKE_BUILD_TYPE=Release --pkg livox_ros_driver
sleep 1s

catkin_make -DCMAKE_BUILD_TYPE=Release --pkg zed_ros
sleep 1s

catkin_make -DCMAKE_BUILD_TYPE=Release --pkg zed_wrapper zed_interfaces zed_nodelets
sleep 1s

catkin_make -DCMAKE_BUILD_TYPE=Release --pkg fast_lio
sleep 1s

catkin_make -DCMAKE_BUILD_TYPE=Release --pkg fuse_color_3d
sleep 1s

