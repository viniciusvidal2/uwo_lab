#include <ros/ros.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <math.h>
#include <ctime>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CompressedImage.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/videoio.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

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
using namespace cv;
using namespace std;
using namespace Eigen;

// Type defs for point types PCL
typedef PointXYZ PointIn;
typedef PointXYZRGB PointT;

// Intrinsic and extrinsic matrices, distortion coeficients
static Matrix3f K;
static MatrixXf P(3, 4);
Mat dist_coefs, K_;

// Rotation and translation from body frame to global frame
Eigen::Quaternionf q_bg;
Vector3f t_bg;
Matrix4f T_lc; // From laser to camera frame (X forward to Z forward)

// Global image pointer and undistorted image
cv_bridge::CvImagePtr cam_img;
Mat image_undistorted;

// Global point cloud
PointCloud<PointT>::Ptr cloud;

// Cloud publisher
ros::Publisher cloud_publisher;

// Void to project the cloud points in the image and get the color from it
void colorCloudCPU(PointCloud<PointT>::Ptr cloud_in, Mat image){
#pragma omp parallel for
  for(size_t i = 0; i < cloud_in->size(); i++){
    // Homogeneous coordiantes
    MatrixXf X_(4, 1);
    X_ << cloud_in->points[i].x,
        cloud_in->points[i].y,
        cloud_in->points[i].z,
        1;
    X_ = T_lc*X_;
    MatrixXf X(3, 1);
    X = P*X_;
    if(X(2, 0) > 0){
      X = X/X(2, 0);
      // If projected inside the image, pick pixel color
      if(floor(X(0,0)) > 0 && floor(X(0,0)) < image.cols && floor(X(1,0)) > 0 && floor(X(1,0)) < image.rows){
        cv::Vec3b cor = image.at<Vec3b>(Point(X(0,0), X(1,0)));
        PointT point = cloud_in->points[i];
        point.b = cor.val[0]; point.g = cor.val[1]; point.r = cor.val[2];
        cloud_in->points[i] = point;
      }
    }
  }
//  // Points that were not projected should be removed
//  ExtractIndices<PointXYZRGB> extract;
//  PointIndices::Ptr inliers (new PointIndices);
//  for (size_t i = 0; i < cloud_in->size(); i++){
//    if (cloud_in->points[i].r == 0 && cloud_in->points[i].g == 0 && cloud_in->points[i].b == 0)
//      inliers->indices.emplace_back(i);
//  }
//  extract.setInputCloud(cloud_in);
//  extract.setIndices(inliers);
//  extract.setNegative(true);
//  extract.filter(*cloud_in);
}

/// Camera callback
///
void camCallback(const sensor_msgs::CompressedImageConstPtr& msg){
  // Update the image pointer
  cam_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  // Undistort the image
  undistort(cam_img->image, image_undistorted, K_, dist_coefs);
}

/// Point cloud callback
///
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg){
  // Convert the message - force PointT cloud
  PointCloud<PointIn>::Ptr cloud_in (new PointCloud<PointIn>);
  fromROSMsg(*msg, *cloud_in);

  // Copy the point cloud to the global one with RGB
  copyPointCloud(*cloud_in, *cloud);
}

/// Odometry callback
///
void odomCallback(const nav_msgs::OdometryConstPtr& msg){
  // Get the current odometry to transform the cloud back to the body frame
  q_bg.w() = msg->pose.pose.orientation.w;
  q_bg.x() = msg->pose.pose.orientation.x;
  q_bg.y() = msg->pose.pose.orientation.y;
  q_bg.z() = msg->pose.pose.orientation.z;
  t_bg << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "project_image_pointcloud_node");
  ros::NodeHandle nh;
  ROS_INFO("Initialyzing node ...");

  // Initialize point cloud and image pointers
  cloud = (PointCloud<PointT>::Ptr) new PointCloud<PointT>();
  cloud->header.frame_id = "camera_init";
  cam_img = (cv_bridge::CvImagePtr) new cv_bridge::CvImage;

  // Initialize matrices - r3live datasets
  K << 863.4241, 0.0, 640.6808,
      0.0,  863.4171, 518.3392,
      0.0, 0.0, 1.0 ;
  P << 1, 0, 0, 0.050166,
      0, 1, 0, 0.0474116,
      0, 0, 1, -0.0312415;
  P.block<3,3>(0, 0) = K;
  float dist_coefs_[5] = {-0.1080, 0.1050, -1.2872e-04, 5.7923e-05, -0.0222};
  dist_coefs = Mat(5, 1, CV_64F, dist_coefs_);
  eigen2cv(K, K_);

  // Rotation from laser to camera frame
  Matrix3f R;
  R = AngleAxisf(M_PI/2, Vector3f::UnitZ()) * AngleAxisf(-M_PI/2, Vector3f::UnitY());
  T_lc = Matrix4f::Identity();
  T_lc.block<3, 3>(0, 0) = R;

  // Initialize cloud publisher
  cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/cloud_colored", 10);

  // Initialize sync subscribers
  ros::Subscriber sub_cam   = nh.subscribe("/camera/image_color/compressed", 3, camCallback);
  ros::Subscriber sub_cloud = nh.subscribe("/cloud_registered", 3, cloudCallback);
  ros::Subscriber sub_odom  = nh.subscribe("/Odometry", 3, odomCallback);

  // Do the work in here so the subscribers are free
  while(ros::ok()){
    ros::spinOnce();
    // Transform the cloud from global frame to body frame
//    transformPointCloud<PointT>(*cloud, *cloud, -t_bg, q_bg.inverse());
    // Color cloud
    colorCloudCPU(cloud, image_undistorted);
    // Transform the cloud from body frame to global frame
//    transformPointCloud<PointT>(*cloud, *cloud, t_bg, q_bg);

    // Publish in the new topic
    sensor_msgs::PointCloud2 out_msg;
    toROSMsg(*cloud, out_msg);
    out_msg.header.frame_id = "camera_init";
    cloud_publisher.publish(out_msg);
  }

  return 0;
}
