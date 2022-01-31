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
#include <rosgraph_msgs/Clock.h>

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
PointCloud<PointIn>::Ptr cloud_in, cloud_save;
PointCloud<PointT>::Ptr cloud_rgb;
PassThrough<PointIn> pass;

// Intrinsic and extrinsic matrices, distortion coeficients
static Matrix4f P;
Mat dist_coefs, K_;
bool save_calibration = false;
int save_cloud_counter = 0;

// Mutex for subscriber
mutex mtx;
mutex mtx_save;

// Undistorted image
Mat image_undistorted;

// Subscriber for image
ros::Subscriber im_sub;

// Publishers
ros::Publisher cloud_publisher;
ros::Publisher odom_publisher;

// Output message for odometry
nav_msgs::Odometry odom_msg_out;

// Time control for new messages, to check if we have finished our goal
ros::Time t_control_input;
bool new_data = false;

// Current clock time
ros::Time clk;

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
  // Points that were either not projected or come from noise should be removed
  ExtractIndices<PointXYZRGB> extract;
  PointIndices::Ptr outliers (new PointIndices);
  for (size_t i = 0; i < cloud_in->size(); i++){
    if ((cloud_in->points[i].r == 0 && cloud_in->points[i].g == 0 && cloud_in->points[i].b == 0) ||
        (cloud_in->points[i].x == 0 && cloud_in->points[i].y == 0 && cloud_in->points[i].z == 0))
      outliers->indices.emplace_back(i);
  }
  extract.setInputCloud(cloud_in);
  extract.setIndices(outliers);
  extract.setNegative(true);
  extract.filter(*cloud_in);
}

/// Sync callback
///
void syncCallback(const sensor_msgs::PointCloud2ConstPtr &cl_msg,
                  const nav_msgs::OdometryConstPtr &o_msg){ 
  t_control_input = ros::Time::now();

  // Convert the message
  fromROSMsg(*cl_msg, *cloud_in);

  // Get the Odometry msg
  odom_msg_out = *o_msg;

  // Save the cloud for calibration, if we want so
  if (save_calibration){
    *cloud_save += *cloud_in;    
    save_cloud_counter++;
    if (save_cloud_counter > 60){
      mtx_save.lock();
      // Rotation from laser to camera frame
      Matrix3f R;
      R = AngleAxisf(M_PI/2, Vector3f::UnitZ()) * AngleAxisf(-M_PI/2, Vector3f::UnitY());
      Matrix4f T_lc = Matrix4f::Identity(); // From laser to camera frame (X forward to Z forward)
      T_lc.block<3, 3>(0, 0) = R;
      transformPointCloud<PointIn>(*cloud_save, *cloud_save, T_lc);
      io::savePLYFileBinary<PointIn>(string(getenv("HOME"))+"/Desktop/cloud_for_callibation.ply", *cloud_save);
      cv::imwrite(string(getenv("HOME"))+"/Desktop/image_for_callibation.png", image_undistorted);
      sleep(5);
      ROS_WARN("Everything saved for callibration !");
      mtx_save.unlock();
      ros::shutdown();
    }
  }

  new_data = true;
}

/// Process callback
/// process data and publish it
///
void processCallback(const ros::TimerEvent&){
  if(new_data){
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
    if (im_sub.getNumPublishers() > 0){
      if ((ros::Time::now() - t_control_input).toSec() < 1){ // So we dont publish forever when data stops coming
        if (cloud_rgb->points.size() > 0){
          cloud_publisher.publish(cl_msg_out);
          odom_publisher.publish(odom_msg_out);
        }
      }
    } else {
      ROS_WARN("No new messages to send, closing node ...");
      ros::shutdown();
    }

    new_data = false;

    // Free mutex
    mtx.unlock();
  }
}

/// Image callback
///
void imageCallback(const sensor_msgs::CompressedImageConstPtr &msg){
  // Check latency through timestamp
  double t = (clk - msg->header.stamp).toSec();
  ROS_WARN("LATENCY here is %.5f", t);
  int size = msg->data.size()*sizeof(uint8_t) + msg->format.size()*sizeof(uint8_t) + sizeof(msg->header.stamp);
  ROS_WARN("MESSAGE SIZE here is %d", size);
  // Convert to cv_bridge
  cv_bridge::CvImagePtr cam_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  // Undistort the image
  undistort(cam_img->image, image_undistorted, K_, dist_coefs);
}

/// Current time callback
///
void clockCallback(const rosgraph_msgs::ClockConstPtr &msg){
  clk = msg->clock;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "project_image_pointcloud_node");
  ros::NodeHandle nh;
  ros::NodeHandle n_("~");
  ROS_INFO("Initialyzing node ...");

  // Initialize global point cloud pointer
  cloud_in   = (PointCloud<PointIn>::Ptr) new PointCloud<PointIn>;
  cloud_save = (PointCloud<PointIn>::Ptr) new PointCloud<PointIn>; // In case we are saving for calibration
  cloud_rgb  = (PointCloud<PointT>::Ptr) new PointCloud<PointT>;
  cloud_rgb->header.frame_id = CLOUD_FRAME;

  // Initialize matrices according to camera parameters
  Matrix3f K;
  vector<float> K_vec(9, 0.0), P_vec(16, 0.0), dist_coefs_(5, 0.0);
  n_.param<vector<float>>("camera_calibration_parameters/intrinsic_K", K_vec, vector<float>());
  n_.param<vector<float>>("camera_calibration_parameters/extrinsic_P", P_vec, vector<float>());
  n_.param<vector<float>>("camera_calibration_parameters/dist_coeffs", dist_coefs_, vector<float>());
  n_.param<bool>("camera_calibration_parameters/save_cloud", save_calibration, false);
  K << K_vec[0], K_vec[1], K_vec[2],
      K_vec[3], K_vec[4], K_vec[5],
      K_vec[6], K_vec[7], K_vec[8];
  P << P_vec[0], P_vec[1], P_vec[2], P_vec[3],
      P_vec[4], P_vec[5], P_vec[6], P_vec[7],
      P_vec[8], P_vec[9], P_vec[10], P_vec[11],
      P_vec[12], P_vec[13], P_vec[14], P_vec[15];
  P.block<3,3>(0, 0) = K;
  dist_coefs = Mat(5, 1, CV_64F, dist_coefs_.data());
  eigen2cv(K, K_);

  // Rotation from laser to camera frame
  Matrix3f R;
  R = AngleAxisf(M_PI/2, Vector3f::UnitZ()) * AngleAxisf(-M_PI/2, Vector3f::UnitY());
  Matrix4f T_lc = Matrix4f::Identity(); // From laser to camera frame (X forward to Z forward)
  T_lc.block<3, 3>(0, 0) = R;

  // Incorporate frame transform in the extrinsic matrix for once
  P = P*T_lc;

  // Robot name
  string robot_name;
  n_.param<string>("robot_name", robot_name, "robot");

  // Initialize publishers
  cloud_publisher = nh.advertise<sensor_msgs::PointCloud2>("/"+robot_name+"/cloud_colored", 10000);
  odom_publisher  = nh.advertise<nav_msgs::Odometry>("/"+robot_name+"/odometry", 10000);

  // Initialize sync subscribers
  string image_topic, cloud_topic, odometry_topic;
  n_.param<string>("input_topics/image_topic", image_topic, "/camera/image_color/compressed");
  n_.param<string>("input_topics/cloud_topic", cloud_topic, "/cloud_registered_body");
  n_.param<string>("input_topics/odometry_topic", odometry_topic, "/Odometry");

  ros::Subscriber clk_sub = nh.subscribe("/"+robot_name+"/clock", 100, &clockCallback);

  im_sub = nh.subscribe("/"+robot_name+image_topic, 1, &imageCallback);
  ros::Rate r(5);
  while (im_sub.getNumPublishers() < 1){
    r.sleep();
    ROS_WARN("Waiting for the image topic to show up ...");
  }
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, "/"+robot_name+cloud_topic, 10000);
  message_filters::Subscriber<nav_msgs::Odometry>       odom_sub(nh, "/"+robot_name+odometry_topic, 10000);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> sync_pol;
  message_filters::Synchronizer<sync_pol> sync(sync_pol(10000), cloud_sub, odom_sub);
  sync.registerCallback(boost::bind(&syncCallback, _1, _2));
  ros::SubscribeOptions ops;
  ops.allow_concurrent_callbacks = true;
  t_control_input = ros::Time::now();

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
