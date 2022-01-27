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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree.h>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/VoxelGrid.h"
#include "open3d/geometry/Geometry3D.h"
#include "open3d/geometry/KDTreeFlann.h"
#include "open3d/geometry/KDTreeSearchParam.h"
#include "open3d/core/nns/NearestNeighborSearch.h"
#include "open3d/Open3D.h"
#include "open3d/geometry/MeshBase.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/pipelines/registration/ColoredICP.h"

#include "uwo_pack/cloud.h"

using namespace pcl;
using namespace std;
using namespace Eigen;

// Definitions for PCL
typedef PointXYZRGB PointT;
#define CLOUD_FRAME "camera_init"

// Definitions for Open3D
namespace o3d = open3d;
namespace o3g = open3d::geometry;

// Directory to save the final mesh
string save_directory;

/// Compute normals in parallel
///
void compute_normals_efficient(PointCloud<PointT>::Ptr in, PointCloud<PointXYZRGBNormal>::Ptr out){
  // Initi normal estimator
  search::KdTree<PointT>::Ptr tree (new search::KdTree<PointT>());
  NormalEstimationOMP<PointT, Normal> ne;
  ne.setInputCloud(in);
  ne.setSearchMethod(tree);
  ne.setKSearch(30);
  ne.setNumberOfThreads(100);
  // Normals cloud
  PointCloud<Normal>::Ptr normals (new PointCloud<Normal>());
  ne.compute(*normals);
  // Add cloud components
  concatenateFields(*in, *normals, *out);
  // Filter NaN normals
  std::vector<int> indicesnan;
  removeNaNNormalsFromPointCloud(*out, *out, indicesnan);
}

/// Check normals directions
///
void check_normals_orientations(PointCloud<PointXYZRGBNormal>::Ptr in, vector<double> xs, vector<double> ys, vector<double> zs){
  // Create point cloud from odometry data
  PointCloud<PointXYZ>::Ptr odom (new PointCloud<PointXYZ>);
  odom->resize(xs.size());
#pragma omp parallel for
  for (size_t i=0; i<xs.size(); i++) {
    odom->points[i].x = xs[i];
    odom->points[i].y = ys[i];
    odom->points[i].z = zs[i];
  }

  // Find closest point with kdtree search
  KdTreeFLANN<PointXYZ>::Ptr tree (new KdTreeFLANN<PointXYZ>);
  tree->setInputCloud(odom);

  // For each point in original point cloud
#pragma omp parallel for
  for (size_t i=0; i<in->size(); i++) {
    vector<int> indices;
    vector<float> dists;
    PointXYZ p;
    p.x = in->points[i].x; p.y = in->points[i].y; p.z = in->points[i].z;
    tree->nearestKSearch(p, 1, indices, dists);
    // Calculate angle between direction and current normal
    Eigen::Vector3f normal, cp, C;
    C << odom->points[indices[0]].x, odom->points[indices[0]].y, odom->points[indices[0]].z;
    normal << in->points[i].normal_x, in->points[i].normal_y, in->points[i].normal_z;
    cp << C(0) - p.x, C(1) - p.y, C(2) - p.z;
    float cos_theta = (normal.dot(cp))/(normal.norm()*cp.norm());

    // If pointing to the opposite side, twist it
    if(cos_theta <= 0){
      in->points[i].normal_x = -in->points[i].normal_x;
      in->points[i].normal_y = -in->points[i].normal_y;
      in->points[i].normal_z = -in->points[i].normal_z;
    }
  }
}

/// Convert PCL cloud to Open3D
///
o3g::PointCloud convertPCL2Open3D(PointCloud<PointXYZRGBNormal>::Ptr c){
  o3g::PointCloud ptc;
  ptc.points_.resize(c->points.size());
  ptc.colors_.resize(c->points.size());
  ptc.normals_.resize(c->points.size());
#pragma omp parallel for
  for (size_t i=0; i<c->points.size(); i++) {
    ptc.points_[i][0] = double(c->points[i].x);
    ptc.points_[i][1] = double(c->points[i].y);
    ptc.points_[i][2] = double(c->points[i].z);
    ptc.colors_[i][0] = double(c->points[i].r)/255.0;
    ptc.colors_[i][1] = double(c->points[i].g)/255.0;
    ptc.colors_[i][2] = double(c->points[i].b)/255.0;
    ptc.normals_[i][0] = double(c->points[i].normal_x);
    ptc.normals_[i][1] = double(c->points[i].normal_y);
    ptc.normals_[i][2] = double(c->points[i].normal_z);
  }
  ptc.NormalizeNormals();
  return ptc;
}
void convertOpen3D2PCL(PointCloud<PointXYZRGBNormal>::Ptr c, o3g::PointCloud o){
  c->clear();
  c->resize(o.points_.size());
#pragma omp parallel for
  for (size_t i=0; i<c->points.size(); i++) {
    c->points[i].x = o.points_[i][0];
    c->points[i].y = o.points_[i][1];
    c->points[i].z = o.points_[i][2];
    c->points[i].r = int(o.colors_[i][0]*255);
    c->points[i].g = int(o.colors_[i][1]*255);
    c->points[i].b = int(o.colors_[i][2]*255);
  }
}

vector<bool> find_low_densities(vector<double> densities, double thresh){
  vector<bool> indices_to_remove(densities.size());
  for (size_t i=0; i<indices_to_remove.size(); i++) {
    indices_to_remove[i] = (densities[i] < thresh) ? true : false;
  }
  return indices_to_remove;
}

/// Service Callback
///
bool work_cloud(uwo_pack::cloud::Request &req, uwo_pack::cloud::Response &res){
  ROS_INFO("Service called to calculate final mesh!");
  ros::Time t;
  // Convert the message into PCL
  PointCloud<PointT>::Ptr cloud (new PointCloud<PointT>);
  fromROSMsg(req.cloud, *cloud);

  // Compute Normals in parallel with PCL - more efficient
  t = ros::Time::now();
  ROS_INFO("Calculating normals ...");
  PointCloud<PointXYZRGBNormal>::Ptr cloud_normals (new PointCloud<PointXYZRGBNormal>);
  compute_normals_efficient(cloud, cloud_normals);
  ROS_WARN("Normals calculated in %.6f seconds ...", (ros::Time::now() - t).toSec());

  t = ros::Time::now();
  ROS_INFO("Checking normals orientation ...");
  check_normals_orientations(cloud_normals, req.xs, req.ys, req.zs);
  ROS_WARN("Normals oriented in %.6f seconds ...", (ros::Time::now() - t).toSec());

  // Convert from PCL to Open3D
  t = ros::Time::now();
  ROS_INFO("Converting PCL to Open3d");
  o3g::PointCloud o3d_cloud = convertPCL2Open3D(cloud_normals);
  ROS_WARN("Converted in %.6f seconds ...", (ros::Time::now() - t).toSec());
  cloud_normals->clear();

  ///// Calculate with no processing, only for comparison
  ROS_INFO("Calculating the mesh for whole point cloud, for comparison ...");
  t = ros::Time::now();
  auto mesh_tuple = o3g::TriangleMesh::CreateFromPointCloudPoisson(o3d_cloud, 10);
  o3d_cloud.Clear();
  vector<bool> indices_to_remove = find_low_densities(get<1>(mesh_tuple), 0.09);
  get<0>(mesh_tuple)->RemoveVerticesByMask(indices_to_remove);
  ROS_WARN("Mesh calculated in %.6f seconds ...", (ros::Time::now() - t).toSec());

  o3d::io::WriteTriangleMesh(save_directory+"/output_mesh.ply", *get<0>(mesh_tuple));

  res.answer = get<0>(mesh_tuple)->IsEmpty() ? false : true;
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "final_mesh_node");
  ros::NodeHandle nh;
  ROS_INFO("Initialyzing node ...");

  nh.getParam("/final_mesh_server_node/save_directory", save_directory);
  cout << save_directory << endl;

  ros::ServiceServer server = nh.advertiseService("calculate_mesh", work_cloud);
  ROS_INFO("Server for mesh calculation is already running.");
//  ros::Time t;

//  // Get the point cloud
//  PointCloud<PointT>::Ptr cloud (new PointCloud<PointT>);
//  io::loadPLYFile("/home/vinicius/Downloads/point_cloud.ply", *cloud);
//  ROS_INFO("Read point cloud with %zu points ...", cloud->points.size());

//  // Compute Normals in parallel with PCL - more efficient
//  t = ros::Time::now();
//  ROS_INFO("Going to calculate normals ...");
//  PointCloud<PointXYZRGBNormal>::Ptr cloud_normals (new PointCloud<PointXYZRGBNormal>);
//  compute_normals_efficient(cloud, cloud_normals);
//  ROS_WARN("Normals calculated in %.6f seconds ...", (ros::Time::now() - t).toSec());

//  // Convert from PCL to Open3D
//  t = ros::Time::now();
//  ROS_INFO("Converting PCL to Open3d");
//  o3g::PointCloud o3d_cloud = convertPCL2Open3D(cloud_normals);
//  ROS_WARN("Converted in %.6f seconds ...", (ros::Time::now() - t).toSec());

//  ///// Calculate with no processing, only for comparison
//  ROS_INFO("Calculating the mesh for whole point cloud, for comparison ...");
//  t = ros::Time::now();
//  auto mesh_tuple = o3g::TriangleMesh::CreateFromPointCloudPoisson(o3d_cloud, 12);
//  vector<bool> indices_to_remove = find_low_densities(get<1>(mesh_tuple), 0.01);
//  get<0>(mesh_tuple)->RemoveVerticesByMask(indices_to_remove);
//  ROS_WARN("Mesh calculated in %.6f seconds ...", (ros::Time::now() - t).toSec());
//  ///
//  ///

//  // Separate the point cloud in regions - calculate centers
//  t = ros::Time::now();
//  ROS_INFO("Calculating the cloud sections centers based on cloud dimensions ...");
//  double cube_side = 20;
//  Vector3d extent = o3d_cloud.GetMaxBound() - o3d_cloud.GetMinBound();
//  int x_side = int(extent[0]/cube_side) + 1, y_side = int(extent[0]/cube_side) + 1, n_centers = x_side*y_side;
//  vector<Vector3d> centers(n_centers);
//#pragma omp parallel for num_threads=n_centers
//  for(int i=0; i<x_side; i++){
//    for(int j=0; j<y_side; j++){
//      centers[i*y_side + j] = {o3d_cloud.GetMinBound()[0] + cube_side/2 + i*cube_side,
//                               o3d_cloud.GetMinBound()[1] + cube_side/2 + j*cube_side,
//                               o3d_cloud.GetMinBound()[2] + extent[2]/2};
//    }
//  }
//  ROS_INFO("Applying bounding boxes to segment data ...");
//  // Separate in cloud vector to be processed in parallel
//  vector<o3g::PointCloud> cloud_vector(centers.size());
//#pragma omp parallel for num_threads=centers.size()
//  for (size_t i=0; i<centers.size(); i++ ) {
//    Vector3d ext{cube_side, cube_side, extent[2]};
//    o3g::OrientedBoundingBox box(centers[i], Matrix3d::Identity(), ext);
//    auto indices = box.GetPointIndicesWithinBoundingBox(o3d_cloud.points_);
//    cloud_vector[i] = *o3d_cloud.SelectByIndex(indices);
//  }
//  o3d_cloud.Clear();
//  ROS_WARN("Segmented cloud sections in %.6f seconds, calculating meshes ...", (ros::Time::now() - t).toSec());

//  // Process the vector o clouds in parallel to get the meshes //
//  vector<o3g::TriangleMesh> mesh_vector(cloud_vector.size());
//#pragma omp parallel for
//  for (size_t i=0; i<cloud_vector.size(); i++) {
//    if (cloud_vector[i].points_.size() > 100) { // There must be a rellevant number of points to be processed
////      ROS_INFO("Processing cloud section %zu !!", i);
//      // Separate planes
////      ROS_INFO("Segmenting plane ...");
////      t = ros::Time::now();
//      auto plane_tuple = cloud_vector[i].SegmentPlane(0.05, 30, 100);
//      auto plane = cloud_vector[i].SelectByIndex(get<1>(plane_tuple));
//      auto objects = cloud_vector[i].SelectByIndex(get<1>(plane_tuple), true);
////      ROS_WARN("Segmented plane in %.6f seconds ...", (ros::Time::now() - t).toSec());
//      // Obtain best parameters for DBSCAN

////      // Separate clusters with DBSCAN
////      ROS_INFO("Segmenting DBSCAN clusters ...");
////      t = ros::Time::now();
////      vector<int> labels = objects->ClusterDBSCAN(0.1, 5, true);
////      cout << labels[0] << "   " << labels[10] << endl;
////      ROS_INFO("aqui %zu ...", objects->points_.size());
////      int num_clusters = *max_element(labels.begin(), labels.end());
////      ROS_INFO("aqui %d ...", num_clusters);
////      vector<vector<size_t>> clusters_labels(num_clusters);
////      ROS_INFO("aqui ...");
////      for (size_t k=0; k<labels.size(); k++) {
////        clusters_labels[labels[k]].emplace_back(k);
////      }
////      vector<o3g::PointCloud> clusters(num_clusters);
////      for (size_t k=0; k<labels.size(); k++) {
////        clusters[k] = *objects->SelectByIndex(clusters_labels[k]);
////      }
////      objects->Clear();
////      ROS_WARN("Segmented clusters in %.6f seconds ...", (ros::Time::now() - t).toSec());

//      // Calculate mesh for planes
////      ROS_INFO("Calculating the mesh for plane ...");
////      t = ros::Time::now();
//      auto plane_mesh_tuple = o3g::TriangleMesh::CreateFromPointCloudPoisson(*plane, 8);
//      vector<bool> indices_to_remove = find_low_densities(get<1>(plane_mesh_tuple), 0.01);
//      get<0>(plane_mesh_tuple)->RemoveVerticesByMask(indices_to_remove);
//      plane->Clear();
////      ROS_WARN("Mesh calculated in %.6f seconds ...", (ros::Time::now() - t).toSec());

//      // Calculate mesh for the objects
////      ROS_INFO("Calculating the mesh for objects ...");
////      t = ros::Time::now();
//      auto objects_mesh_tuple = o3g::TriangleMesh::CreateFromPointCloudPoisson(*objects, 12);
//      indices_to_remove = find_low_densities(get<1>(objects_mesh_tuple), 0.01);
//      get<0>(objects_mesh_tuple)->RemoveVerticesByMask(indices_to_remove);
//      objects->Clear();
////      ROS_WARN("Mesh calculated in %.6f seconds ...", (ros::Time::now() - t).toSec());

//      // Put all of them toguether for this cloud section
//      mesh_vector[i] = *get<0>(plane_mesh_tuple) + *get<0>(objects_mesh_tuple);

////      // Initialize and calculate the mesh vector for this cloud section, one for each cluster
////      ROS_INFO("Calculating the mesh for the clusters ...");
////      t = ros::Time::now();
////      vector<o3g::TriangleMesh> cloud_section_cluster_meshes(clusters.size());
////      //#pragma omp parallel for
////      for (size_t k=0; k<clusters.size(); k++) {
////        auto cluster_mesh_tuple = o3g::TriangleMesh::CreateFromPointCloudPoisson(clusters[k], 12);
////        vector<bool> indices_to_remove = find_low_densities(get<1>(cluster_mesh_tuple), 0.01);
////        get<0>(cluster_mesh_tuple)->RemoveVerticesByMask(indices_to_remove);
////        cloud_section_cluster_meshes[k] = *get<0>(cluster_mesh_tuple);
////      }
////      clusters.clear();
////      ROS_WARN("Mesh calculated in %.6f seconds ...", (ros::Time::now() - t).toSec());

////      // Put all of them toguether for this cloud section
////      mesh_vector[i] = *get<0>(plane_mesh_tuple);
////      for (auto m:cloud_section_cluster_meshes)
////        mesh_vector[i] += m;
////      cloud_section_cluster_meshes.clear();
//    }
//  }

////  ROS_INFO("Putting toguether all the meshes from all cloud sections ...");
//  // Put toguether all the meshes
//  o3g::TriangleMesh mesh;
//  for (auto m:mesh_vector){
//    if (!m.IsEmpty())
//      mesh += m;
//  }
//  mesh_vector.clear();
//  ROS_WARN("All done in %.5f seconds !!!", (ros::Time::now() - t).toSec());

//  // Display final cloud
//  open3d::visualization::Visualizer vis3;
//  vis3.CreateVisualizerWindow("final mesh", 1400, 700);
//  vis3.AddGeometry(std::make_shared<open3d::geometry::TriangleMesh>(mesh));
//  vis3.Run();
//  vis3.DestroyVisualizerWindow();
//  vis3.ClearGeometries();

  ros::spin();
  ros::shutdown();
  return 0;
}
