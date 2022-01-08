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

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/VoxelGrid.h"
#include "open3d/geometry/Geometry3D.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/Open3D.h"
#include "open3d/geometry/MeshBase.h"
#include "open3d/geometry/TriangleMesh.h"

using namespace pcl;
using namespace std;
using namespace Eigen;

// Type defs for point types PCL
typedef PointXYZRGB PointT;
#define CLOUD_FRAME "camera_init"

/// Convert PCL cloud to Open3D
///
open3d::geometry::PointCloud convertPCL2Open3D(PointCloud<PointT>::Ptr c){
  open3d::geometry::PointCloud ptc;
  ptc.points_.resize(c->points.size());
  ptc.colors_.resize(c->points.size());
  ptc.normals_.resize(c->points.size());
#pragma omp parallel for
  for (size_t i=0; i<c->points.size(); i++) {
    ptc.points_[i][0] = double(c->points[i].x);
    ptc.points_[i][1] = double(c->points[i].y);
    ptc.points_[i][2] = double(c->points[i].z);
    ptc.colors_[i][0] = double(c->points[i].b)/255.0;
    ptc.colors_[i][1] = double(c->points[i].g)/255.0;
    ptc.colors_[i][2] = double(c->points[i].r)/255.0;
  }

  return ptc;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "final_mesh_node");
  ros::NodeHandle nh;
  ROS_INFO("Initialyzing node ...");

  // Get the point cloud
  PointCloud<PointT>::Ptr cloud (new PointCloud<PointT>);
  pcl::io::loadPLYFile("/home/vinicius/Downloads/point_cloud.ply", *cloud);

  // Convert from PCL to Open3D
  open3d::geometry::PointCloud o3d_cloud = convertPCL2Open3D(cloud);
  o3d_cloud.EstimateNormals(open3d::geometry::KDTreeSearchParamKNN(10), true);
  o3d_cloud.OrientNormalsToAlignWithDirection();

  // Separate the point cloud in regions

  // Calculate the normals (todo: involve the path too in the future)

  // Separate planes and clusters

  // Calculate mesh for planes

  // Calculate mesh for clusters

  // Put all of them toguether

  // Display final cloud
  open3d::visualization::Visualizer vis3;
  vis3.CreateVisualizerWindow("teste3", 1400, 700);
  vis3.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(o3d_cloud));
  vis3.Run();
  vis3.DestroyVisualizerWindow();
  vis3.ClearGeometries();

  ros::spinOnce();
  ros::shutdown();
  return 0;
}
