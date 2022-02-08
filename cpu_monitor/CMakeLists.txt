cmake_minimum_required(VERSION 3.0.2)
project(cpu_monitor)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
    rospy
    roscpp
    std_msgs
)
catkin_package(
#  INCLUDE_DIRS
#  LIBRARIES cpu_monitor
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(log_ram_cpu_node src/log_ram_cpu_node.cpp)
target_link_libraries(log_ram_cpu_node
${catkin_LIBRARIES}
)

add_executable(log_cpu_total src/log_cpu_total.cpp)
target_link_libraries(log_cpu_total
${catkin_LIBRARIES}
)

#install(PROGRAMS
#   monitor.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)
#install(DIRECTORY launch
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#  PATTERN *~ EXCLUDE
#)


