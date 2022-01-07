#include <ros/ros.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <math.h>
#include <mutex>
#include <ctime>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace pcl;
using namespace std;
using namespace Eigen;

// Type defs for point types PCL
typedef PointXYZRGB PointT;
#define CLOUD_FRAME "camera_init"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "final_mesh_node");
  ros::NodeHandle nh;
  ROS_INFO("Initialyzing node ...");

  // Get the point cloud
  PointCloud<PointT>::Ptr cloud (new PointCloud<PointT>);
  pcl::io::loadPLYFile("name.ply", *cloud);

  // Convert from PCL to Open3D

  // Separate the point cloud in regions

  // Calculate the normals (todo: involve the path too in the future)

  // Separate planes and clusters

  // Calculate mesh for planes

  // Calculate mesh for clusters

  // Put all of them toguether

  // Display final cloud

  ros::spinOnce();
  ros::shutdown();
  return 0;
}
