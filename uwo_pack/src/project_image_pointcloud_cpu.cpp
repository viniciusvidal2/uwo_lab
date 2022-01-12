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
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

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
#define CLOUD_FRAME "body"
PointCloud<PointIn>::Ptr cloud_in;
PointCloud<PointT>::Ptr cloud_rgb;
PassThrough<PointIn> pass;

// Intrinsic and extrinsic matrices, distortion coeficients
static Matrix4f P;
Mat dist_coefs, K_;

// Mutex for subscriber
mutex mtx;

// Undistorted image
Mat image_undistorted;

// Publishers
ros::Publisher cloud_publisher;
ros::Publisher odom_publisher;

// Output message for odometry
nav_msgs::Odometry odom_msg_out;

// Time control for new messages, to check if we have finished our goal
ros::Time t_control_input;

// Void to project the cloud points in the image and get the color from it
void colorCloudCPU(PointCloud<PointT>::Ptr cloud_in, Mat image){
#pragma omp parallel for
  for(size_t i = 0; i < cloud_in->size(); i++){
    // Calculate the third element that represent the pixel at scale, if it is positive the 3D point lies in
    // front of the camera, so we are good to go
    PointT point = cloud_in->points[i];
    float w = P(2, 0)*point.x + P(2, 1)*point.y + P(2, 2)*point.z + P(2, 3);
    if (w > 0){
      // Calculate the pixel coordinates
      float u = (P(0, 0)*point.x + P(0, 1)*point.y + P(0, 2)*point.z + P(0, 3))/w;
      float v = (P(1, 0)*point.x + P(1, 1)*point.y + P(1, 2)*point.z + P(1, 3))/w;
      // Get the pixel color value
      if(floor(u) > 0 && floor(u) < image.cols && floor(v) > 0 && floor(v) < image.rows){
        cv::Vec3b cor = image.at<Vec3b>(Point(u, v));
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

/// Sync callback
///
void syncCallback(const sensor_msgs::CompressedImageConstPtr &im_msg,
                  const sensor_msgs::PointCloud2ConstPtr &cl_msg,
                  const nav_msgs::OdometryConstPtr &o_msg){ 
  t_control_input = ros::Time::now();

  // Update the image pointer
  cv_bridge::CvImagePtr cam_img = cv_bridge::toCvCopy(im_msg, sensor_msgs::image_encodings::BGR8);
  // Undistort the image
  undistort(cam_img->image, image_undistorted, K_, dist_coefs);

  // Convert the message
  fromROSMsg(*cl_msg, *cloud_in);

  // Get the Odometry msg
  odom_msg_out = *o_msg;
}


/// Process callback
/// process data and publish it
///
void processCallback(const ros::TimerEvent&){
  // Lock mutex
  mtx.lock();

  pass.setInputCloud(cloud_in);
  pass.filter(*cloud_in);

  // Copy the point cloud to the one with RGB
  copyPointCloud(*cloud_in, *cloud_rgb);

  // Color cloud
  colorCloudCPU(cloud_rgb, image_undistorted);

  // Copy into ROS message to be published
  sensor_msgs::PointCloud2 cl_msg_out;
  toROSMsg(*cloud_rgb, cl_msg_out);
  cl_msg_out.header.frame_id = CLOUD_FRAME;
  cl_msg_out.header.stamp = odom_msg_out.header.stamp;

  // Publish both odometry and cloud synchronized for the Scan Context
  if ((ros::Time::now() - t_control_input).toSec() < 1){ // So we dont publish forever when data stops coming
    if (cloud_rgb->points.size() > 0){      
      ROS_INFO("Sending cloud and odometry for optimization ...");
      cloud_publisher.publish(cl_msg_out);
      odom_publisher.publish(odom_msg_out);
    }
  } else {
    ROS_INFO("No new messages to send ...");
  }

  // Free mutex
  mtx.unlock();
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "project_image_pointcloud_node");
  ros::NodeHandle nh;
  ROS_INFO("Initialyzing node ...");

  // Initialize global point cloud pointer
  cloud_in  = (PointCloud<PointIn>::Ptr) new PointCloud<PointIn>;
  cloud_rgb = (PointCloud<PointT>::Ptr) new PointCloud<PointT>;
  cloud_rgb->header.frame_id = CLOUD_FRAME;

  // Initialize matrices - r3live datasets
  Matrix3f K;
  K << 863.4241, 0.0, 640.6808,
      0.0,  863.4171, 518.3392,
      0.0, 0.0, 1.0 ;
  P << 1, 0, 0, 0.050166,
      0, 1, 0, 0.0474116,
      0, 0, 1, -0.0312415,
      0, 0, 0, 1;
  P.block<3,3>(0, 0) = K;
  float dist_coefs_[5] = {-0.1080, 0.1050, -1.2872e-04, 5.7923e-05, -0.0222};
  dist_coefs = Mat(5, 1, CV_64F, dist_coefs_);
  eigen2cv(K, K_);

  // Rotation from laser to camera frame
  Matrix3f R;
  R = AngleAxisf(M_PI/2, Vector3f::UnitZ()) * AngleAxisf(-M_PI/2, Vector3f::UnitY());
  Matrix4f T_lc = Matrix4f::Identity(); // From laser to camera frame (X forward to Z forward)
  T_lc.block<3, 3>(0, 0) = R;

  // Incorporate frame transform in the extrinsic matrix for once
  P = P*T_lc;

  // Initialize publishers
  cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/edge/cloud_colored", 10000);
  odom_publisher  = nh.advertise<nav_msgs::Odometry>("/edge/odometry", 10000);

  // Initialize sync subscribers
  message_filters::Subscriber<sensor_msgs::CompressedImage> im_sub(nh, "/camera/image_color/compressed", 10);
  message_filters::Subscriber<sensor_msgs::PointCloud2>  cloud_sub(nh, "/cloud_registered_body", 10);
  message_filters::Subscriber<nav_msgs::Odometry>         odom_sub(nh, "/Odometry", 10);
  t_control_input = ros::Time::now();

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage,
      sensor_msgs::PointCloud2, nav_msgs::Odometry> sync_pol;
  message_filters::Synchronizer<sync_pol> sync(sync_pol(1000), im_sub, cloud_sub, odom_sub);
  sync.registerCallback(boost::bind(&syncCallback, _1, _2, _3));
  ros::SubscribeOptions ops;
  ops.allow_concurrent_callbacks = true;

  ROS_INFO("Listening to sensors data ...");

  // Filter for far points
  pass.setFilterFieldName("x");
  pass.setFilterLimits(0.0, 30.0);

  ros::Timer timer = nh.createTimer(ros::Duration(0.1), processCallback);

  ros::MultiThreadedSpinner spinner(6);
  spinner.spin();

  cloud_rgb->clear();
  cloud_in->clear();
  ros::shutdown();
  return 0;
}
