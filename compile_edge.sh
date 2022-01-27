#!/bin/bash

# cd workspace directory and source ROS dependencies
cd $HOME/uwo_ws
mv src/uwo_lab/mesh_open3d .
mv src/uwo_lab/SC-PGO .
source /opt/ros/melodic/setup.bash
source devel/setup.bash


# Go through all the packages, sleep in between
catkin_make -DCMAKE_BUILD_TYPE=Release -j2 --pkg livox_ros_driver
sleep 1s

catkin_make -DCMAKE_BUILD_TYPE=Release -j2 --pkg zed_ros
sleep 1s

catkin_make -DCMAKE_BUILD_TYPE=Release -j2 --pkg zed_wrapper zed_interfaces zed_nodelets
sleep 1s

catkin_make -DCMAKE_BUILD_TYPE=Release -j1 --pkg fast_lio
sleep 1s

catkin_make -DCMAKE_BUILD_TYPE=Release -j2 --pkg fuse_color_3d
sleep 1s

mv mesh_open3d src/uwo_lab/
mv SC-PGO src/uwo_lab/
