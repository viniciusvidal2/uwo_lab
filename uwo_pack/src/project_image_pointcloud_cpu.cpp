#include <ros/ros.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <math.h>
#include <ctime>

#include <nav_msgs/Odometry.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/videoio.hpp"
#include <opencv2/calib3d.hpp>

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

// Intrinsic and extrinsic matrices
static Matrix3f K;
static MatrixXf P(3, 4);

// Rotation and translation from body frame to global frame
Eigen::Quaternionf q_bg;
Vector3f t_bg;

// Global image pointer
cv_bridge::CvImagePtr cam_img;
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
        1          ;
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
  // Points that were not projected should be removed
  ExtractIndices<PointXYZRGB> extract;
  PointIndices::Ptr inliers (new PointIndices);
  for (size_t i = 0; i < cloud_in->size(); i++){
    if (cloud_in->points[i].r == 0 && cloud_in->points[i].g == 0 && cloud_in->points[i].b == 0)
      inliers->indices.emplace_back(i);
  }
  extract.setInputCloud(cloud_in);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*cloud_in);
}

/// Camera callback
///
void camCallback(const sensor_msgs::ImageConstPtr& msg){
  // Update the image pointer
  cam_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
}

/// Odometry callback
///
void odomCallback(const nav_msgs::OdometryConstPtr& msg){
  // Get the current odometry to transform the cloud back to the body frame
  q_bg = {msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
          msg->pose.pose.orientation.y, msg->pose.pose.orientation.z};
  t_bg = {msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z};
}

/// Point cloud callback
///
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg){
  // Convert the message - force PointT cloud
  PointCloud<PointIn>::Ptr cloud_in (new PointCloud<PointIn>);
  fromROSMsg(*msg, *cloud_in);

  // Transform the cloud from global frame to body frame
  transformPointCloud<PointXYZ>(*cloud_in, *cloud_in, -t_bg, q_bg.inverse());
  // Project the cloud in the image
  copyPointCloud(*cloud_in, *cloud);
  colorCloudCPU(cloud, cam_img->image);
  // Transform the cloud from body frame to global frame
  transformPointCloud<PointT>(*cloud, *cloud, t_bg, q_bg);

  // Publish in the new topic
  sensor_msgs::PointCloud2 out_msg;
  toROSMsg(*cloud, out_msg);
  out_msg.header.frame_id = "camera_init";
  cloud_publisher.publish(out_msg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "project_image_pointcloud_node");
  ros::NodeHandle nh;

  // Initialize point cloud and image pointers
  cloud = (PointCloud<PointT>::Ptr) new PointCloud<PointT>();
  cloud->header.frame_id = "camera_init";
  cam_img = (cv_bridge::CvImagePtr) new cv_bridge::CvImage;

  // Initialize matrices
  K << 264.7706604003906, 0.0, 340.0319519042969,
      0.0, 264.7706604003906, 191.3612823486328,
      0.0, 0.0, 1.0;
  P << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0;
  P.block<3, 3>(0, 0) = K;

  // Initialize cloud publisher
  cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/cloud_colored", 10);

  // Initialize sync subscribers
  ros::Subscriber sub_cam   = nh.subscribe("zed2/zed_node/rgb/image_rect_color", 100, camCallback);
  ros::Subscriber sub_cloud = nh.subscribe("/cloud_registered", 100, cloudCallback);

  while(ros::ok())
    ros::spin();

  return 0;
}
