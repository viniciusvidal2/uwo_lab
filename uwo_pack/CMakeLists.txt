cmake_minimum_required(VERSION 3.0.2)
project(uwo_pack)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++11)

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
)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Open3D REQUIRED HINTS "~/open3d_install")
if(OPENMP_FOUND)
  message(STATUS "OPENMP FOUND")
  set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})  # or if you use C: ${OpenMP_C_FLAGS}
  set(OpenMP_LIBS gomp)
endif()
if(NOT Open3D_FOUND)
    message(FATAL_ERROR "Open3D not found, please use -DCMAKE_INSTALL_PREFIX=open3d_install_dir")
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
#  ${Open3D_INCLUDE_DIRS}
)

set(CUDA_NVCC_FLAGS "-arch=compute_75" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC)
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

add_executable(final_mesh_node src/final_mesh.cpp)
TARGET_LINK_LIBRARIES(final_mesh_node
${catkin_LIBRARIES}
${PCL_LIBRARIES}
#${Open3D_LIBRARIES}
${OpenMP_LIBS}
)
