cmake_minimum_required(VERSION 3.0.2)
project(mesh_open3d)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#set(CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++14)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    std_msgs
    pcl_ros
    tf2_ros
    pcl_conversions
    nav_msgs
    message_generation
)

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
if(Open3D_FOUND)
    message(STATUS "Open3D FOUND !!!!!!!!!!!!!!!!!!!!!!!!!!")
endif()

# Add service files
add_service_files(
  FILES
  cloud.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES
#  CATKIN_DEPENDS roscpp sensor_msgs std_msgs pcl_ros tf2_ros pcl_conversions nav_msgs
#  DEPENDS OpenMP Open3D
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenMP_INCLUDE_DIRS}
  ${Open3D_INCLUDE_DIRS}
)

#link_directories("~/open3d_install/lib/")
#add_executable(final_mesh_node src/final_mesh.cpp)
#target_link_libraries(final_mesh_node
#${catkin_LIBRARIES}
#${PCL_LIBRARIES}
#${Open3D_LIBRARIES}
#${OpenMP_LIBS}
#)
