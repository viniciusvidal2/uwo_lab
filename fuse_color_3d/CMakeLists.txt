cmake_minimum_required(VERSION 3.0.2)
project(fuse_color_3d)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    image_transport
    roscpp
    sensor_msgs
    std_msgs
    pcl_ros
    tf2_ros
    pcl_conversions
    nav_msgs
    message_filters
    rosgraph_msgs
)

find_package(OpenCV 4.2.0 QUIET)
if(NOT OpenCV_FOUND)
  message(STATUS "Compiling in edge device, earlier OpenCV version ...")
  find_package(OpenCV 4.1.1 QUIET) # This is for the edge with ros melodic
endif()
if(NOT OpenCV_FOUND)
  message(STATUS "Did not find Opencv4, lets go with the ROS builtin version ...")
  find_package(OpenCV REQUIRED) # This is for the edge with ros melodic
endif()
# find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})  # or if you use C: ${OpenMP_C_FLAGS}
  set(OpenMP_LIBS gomp)
endif()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES uwo_pack
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenMP_INCLUDE_DIRS}
)

# set(CUDA_NVCC_FLAGS "-arch=compute_75" CACHE STRING "nvcc flags" FORCE)
# set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
# set(LIB_TYPE STATIC)
#cuda_add_library(CUDAprojectLib ${LIB_TYPE} src/project.cu)

add_executable(project_cloud_cpu_node src/project_image_pointcloud_cpu.cpp)
#add_dependencies(project_cloud_cpu_node CUDAprojectLib)
TARGET_LINK_LIBRARIES(project_cloud_cpu_node
${catkin_LIBRARIES}
#CUDAprojectLib
${OpenCV_LIBS}
${PCL_LIBRARIES}
${OpenMP_LIBS}
)

add_executable(study_octree_efficiency_node src/study_octree_efficiency_node.cpp)
TARGET_LINK_LIBRARIES(study_octree_efficiency_node
${catkin_LIBRARIES}
${OpenCV_LIBS}
${PCL_LIBRARIES}
${OpenMP_LIBS}
)
