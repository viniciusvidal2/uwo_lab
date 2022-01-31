#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "open3d/geometry/PointCloud.h"
//#include "open3d/geometry/BoundingVolume.h"
//#include "open3d/geometry/KDTreeFlann.h"
//#include "open3d/geometry/VoxelGrid.h"
#include "open3d/geometry/Geometry3D.h"
//#include "open3d/geometry/KDTreeFlann.h"
//#include "open3d/geometry/KDTreeSearchParam.h"
//#include "open3d/core/nns/NearestNeighborSearch.h"
//#include "open3d/Open3D.h"
//#include "open3d/geometry/MeshBase.h"
//#include "open3d/geometry/TriangleMesh.h"
#include "open3d/pipelines/registration/ColoredICP.h"

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "scancontext/Scancontext.h"
#include "mesh_open3d/cloud.h"

using namespace gtsam;

using std::cout;
using std::endl;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false; 

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 

std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;

std::mutex mBuf;
std::mutex mKF;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds; 
std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserCloudsFull;
std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
int recentIdxUpdated = 0;

gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;

pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mtxICP;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
bool laserCloudMapPGORedraw = true;

bool useGPS = true;
// bool useGPS = false;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

ros::Publisher pubMapAftPGO, pubLastKf, pubOdomAftPGO, pubPathAftPGO;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubOdomRepubVerifier;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory;
std::string odomKITTIformat;
std::fstream pgTimeSaveStream;

// Vinicius
// Control input and service client
ros::Time t_control_input;
ros::ServiceClient mesh_client;
ros::Subscriber subLaserOdometry;
// Check index pairs being added to the graph
int last_previous, last_curr, count_same = 0;
// Definitions for Open3D
namespace o3d = open3d;
namespace o3g = open3d::geometry;
/// Convert PCL cloud to Open3D
///
o3g::PointCloud convertPCL2Open3D(pcl::PointCloud<pcl::PointXYZRGB>::Ptr c){
  o3g::PointCloud ptc;
  ptc.points_.resize(c->points.size());
  ptc.colors_.resize(c->points.size());
#pragma omp parallel for
  for (size_t i=0; i<c->points.size(); i++) {
    ptc.points_[i][0] = double(c->points[i].x);
    ptc.points_[i][1] = double(c->points[i].y);
    ptc.points_[i][2] = double(c->points[i].z);
    ptc.colors_[i][0] = double(c->points[i].r)/255.0;
    ptc.colors_[i][1] = double(c->points[i].g)/255.0;
    ptc.colors_[i][2] = double(c->points[i].b)/255.0;
  }
  return ptc;
}
void convertOpen3D2PCL(pcl::PointCloud<pcl::PointXYZRGB>::Ptr c, o3g::PointCloud o){
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

std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
  return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

void saveOdometryVerticesKITTIformat(std::string _filename)
{
  // ref from gtsam's original code "dataset.cpp"
  std::fstream stream(_filename.c_str(), std::fstream::out);
  for(const auto& _pose6d: keyframePoses) {
    gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
    Point3 t = pose.translation();
    Rot3 R = pose.rotation();
    auto col1 = R.column(1); // Point3
    auto col2 = R.column(2); // Point3
    auto col3 = R.column(3); // Point3

    stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
           << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
           << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
  }
}

void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
  using namespace gtsam;

  // ref from gtsam's original code "dataset.cpp"
  std::fstream stream(_filename.c_str(), std::fstream::out);

  for(const auto& key_value: _estimates) {
    auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
    if (!p) continue;

    const Pose3& pose = p->value();

    Point3 t = pose.translation();
    Rot3 R = pose.rotation();
    auto col1 = R.column(1); // Point3
    auto col2 = R.column(2); // Point3
    auto col3 = R.column(3); // Point3

    stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
           << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
           << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
  }
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
  t_control_input = ros::Time::now(); // Every time there is a new message, save current time
  mBuf.lock();
  odometryBuf.push(_laserOdometry);
  mBuf.unlock();
} // laserOdometryHandler

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
  mBuf.lock();
  fullResBuf.push(_laserCloudFullRes);
  mBuf.unlock();
} // laserCloudFullResHandler

void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
{
  if(useGPS) {
    mBuf.lock();
    gpsBuf.push(_gps);
    mBuf.unlock();
  }
} // gpsHandler

void initNoises( void )
{
  gtsam::Vector priorNoiseVector6(6);
  priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
  priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

  gtsam::Vector odomNoiseVector6(6);
  // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
  odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
  odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

  double loopNoiseScore = 0.5; // constant is ok...
  gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
  robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
  robustLoopNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
        gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

  double bigNoiseTolerentToXY = 1000000000.0; // 1e9
  double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
  gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
  robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
  robustGPSNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
        gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
  auto tx = _odom->pose.pose.position.x;
  auto ty = _odom->pose.pose.position.y;
  auto tz = _odom->pose.pose.position.z;

  double roll, pitch, yaw;
  geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
  tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

  return Pose6D{tx, ty, tz, roll, pitch, yaw};
} // getOdom

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
  Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
  Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
  Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
  Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
  float dx, dy, dz, droll, dpitch, dyaw;
  pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
  // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

  return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);

  int numberOfCores = 16;
#pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < cloudSize; ++i)
  {
    const auto &pointFrom = cloudIn->points[i];
    cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
    cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
    cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
//    cloudOut->points[i].intensity = pointFrom.intensity;
    cloudOut->points[i].r = pointFrom.r;
    cloudOut->points[i].g = pointFrom.g;
    cloudOut->points[i].b = pointFrom.b;
  }

  return cloudOut;
}

void pubPath( void )
{
  // pub odom and path
  nav_msgs::Odometry odomAftPGO;
  nav_msgs::Path pathAftPGO;
  pathAftPGO.header.frame_id = "camera_init";
  mKF.lock();
  // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
  for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
  {
    const Pose6D& pose_est = keyframePosesUpdated.at(node_idx); // upodated poses
    // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

    nav_msgs::Odometry odomAftPGOthis;
    odomAftPGOthis.header.frame_id = "camera_init";
    odomAftPGOthis.child_frame_id = "/aft_pgo";
    odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
    odomAftPGOthis.pose.pose.position.x = pose_est.x;
    odomAftPGOthis.pose.pose.position.y = pose_est.y;
    odomAftPGOthis.pose.pose.position.z = pose_est.z;
    odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
    odomAftPGO = odomAftPGOthis;

    geometry_msgs::PoseStamped poseStampAftPGO;
    poseStampAftPGO.header = odomAftPGOthis.header;
    poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

    pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
    pathAftPGO.header.frame_id = "camera_init";
    pathAftPGO.poses.push_back(poseStampAftPGO);
  }
  pubOdomAftPGO.publish(odomAftPGO); // last pose
  pubPathAftPGO.publish(pathAftPGO); // poses
  sensor_msgs::PointCloud2 lastKf_msg;
  pcl::toROSMsg(*local2global(keyframeLaserClouds.back(), keyframePosesUpdated.back()), lastKf_msg);
  lastKf_msg.header.stamp = pathAftPGO.header.stamp;
  lastKf_msg.header.frame_id = "camera_init";
  pubLastKf.publish(lastKf_msg);
  mKF.unlock();

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
  q.setW(odomAftPGO.pose.pose.orientation.w);
  q.setX(odomAftPGO.pose.pose.orientation.x);
  q.setY(odomAftPGO.pose.pose.orientation.y);
  q.setZ(odomAftPGO.pose.pose.orientation.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "/aft_pgo"));
} // pubPath

void updatePoses(void)
{
  mKF.lock();
  for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
  {
    Pose6D& p =keyframePosesUpdated[node_idx];
    p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
    p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
    p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
    p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
    p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
    p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
  }
  mKF.unlock();

  mtxRecentPose.lock();
  const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
  recentOptimizedX = lastOptimizedPose.translation().x();
  recentOptimizedY = lastOptimizedPose.translation().y();

  recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

  mtxRecentPose.unlock();
} // updatePoses

void runISAM2opt(void)
{
  // called when a variable added
  isam->update(gtSAMgraph, initialEstimate);
  isam->update();

  gtSAMgraph.resize(0);
  initialEstimate.clear();

  isamCurrentEstimate = isam->calculateEstimate();
  updatePoses();
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
  pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

  PointType *pointFrom;

  int cloudSize = cloudIn->size();
  cloudOut->resize(cloudSize);

  Eigen::Affine3f transCur = pcl::getTransformation(
        transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(),
        transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );

  int numberOfCores = 8; // TODO move to yaml
#pragma omp parallel for num_threads(numberOfCores)
  for (int i = 0; i < cloudSize; ++i)
  {
    pointFrom = &cloudIn->points[i];
    cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
    cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
    cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
//    cloudOut->points[i].intensity = pointFrom->intensity;
    cloudOut->points[i].r = pointFrom->r;
    cloudOut->points[i].g = pointFrom->g;
    cloudOut->points[i].b = pointFrom->b;
  }
  return cloudOut;
} // transformPointCloud

void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size, const int& root_idx)
{
  // extract and stacking near keyframes (in global coord)
  nearKeyframes->clear();
  for (int i = -submap_size; i <= submap_size; ++i) {
    int keyNear = key + i;
    if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) )
      continue;

    mKF.lock();
    *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_idx]);
    mKF.unlock();
  }

  if (nearKeyframes->empty())
    return;

  // downsample near keyframes
  pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
  downSizeFilterICP.setInputCloud(nearKeyframes);
  downSizeFilterICP.filter(*cloud_temp);
  *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud


std::optional<gtsam::Pose3> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx )
{
  // parse pointclouds
  int historyKeyframeSearchNum = 25; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
  pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
  loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0, _loop_kf_idx); // use same root of loop kf idx
  loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx);

  // loop verification
  sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
  pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
  cureKeyframeCloudMsg.header.frame_id = "camera_init";
  pubLoopScanLocal.publish(cureKeyframeCloudMsg);

  sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
  pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
  targetKeyframeCloudMsg.header.frame_id = "camera_init";
  pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);

  // Convert point clouds
  o3g::PointCloud src_o, tgt_o;
  src_o = convertPCL2Open3D(cureKeyframeCloud);
  tgt_o = convertPCL2Open3D(targetKeyframeCloud);
  src_o.EstimateNormals();
  tgt_o.EstimateNormals();

  // Perform ICP
  auto icp_result = o3d::pipelines::registration::RegistrationColoredICP(src_o, tgt_o,
                    10*downSizeFilterICP.getLeafSize().x(), Eigen::Matrix4d::Identity(),
                    o3d::pipelines::registration::TransformationEstimationForColoredICP(),
                    o3d::pipelines::registration::ICPConvergenceCriteria(9.9999e-09, 9.9999e-09, 300));

  // Check if it has passed
  float loopFitnessScoreThreshold = 0.90; // user parameter but fixed low value is safe.
  if (icp_result.fitness_ < loopFitnessScoreThreshold) {
    std::cout << "[SC loop] ICP fitness test failed (" << icp_result.fitness_ << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
    return std::nullopt;
  } else {
    std::cout << "[SC loop] ICP fitness test passed (" << icp_result.fitness_ << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
  }

  // Get pose transformation
  float x, y, z, roll, pitch, yaw;
  Eigen::Affine3f correctionLidarFrame;
  correctionLidarFrame = icp_result.transformation_.cast<float>();//icp.getFinalTransformation();
  pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
  gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
  gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

//  // ICP Settings
//  pcl::IterativeClosestPoint<PointType, PointType> icp;
//  icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
//  icp.setMaximumIterations(200);
//  icp.setTransformationEpsilon(1e-6);
//  icp.setEuclideanFitnessEpsilon(1e-6);
//  icp.setRANSACIterations(10);

//  // Align pointclouds
//  icp.setInputSource(cureKeyframeCloud);
//  icp.setInputTarget(targetKeyframeCloud);
//  pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
//  icp.align(*unused_result);

//  float loopFitnessScoreThreshold = 0.5; // user parameter but fixed low value is safe.
//  if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
//    std::cout << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
//    return std::nullopt;
//  } else {
//    std::cout << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
//  }

//  // Get pose transformation
//  float x, y, z, roll, pitch, yaw;
//  Eigen::Affine3f correctionLidarFrame;
//  correctionLidarFrame = icp.getFinalTransformation();
//  pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
//  gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
//  gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

  return poseFrom.between(poseTo);
} // doICPVirtualRelative

void process_pg()
{
  while(1)
  {
    while ( !odometryBuf.empty() && !fullResBuf.empty() )
    {
      //
      // pop and check keyframe is or not
      //
      mBuf.lock();
      while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
        odometryBuf.pop();
      if (odometryBuf.empty())
      {
        mBuf.unlock();
        break;
      }

      // Time equal check
      timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
      timeLaser = fullResBuf.front()->header.stamp.toSec();
      // TODO

      laserCloudFullRes->clear();
      pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
      pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
      fullResBuf.pop();

      Pose6D pose_curr = getOdom(odometryBuf.front());
      odometryBuf.pop();

      // find nearest gps
      double eps = 0.1; // find a gps topioc arrived within eps second
      while (!gpsBuf.empty()) {
        auto thisGPS = gpsBuf.front();
        auto thisGPSTime = thisGPS->header.stamp.toSec();
        if( abs(thisGPSTime - timeLaserOdometry) < eps ) {
          currGPS = thisGPS;
          hasGPSforThisKF = true;
          break;
        } else {
          hasGPSforThisKF = false;
        }
        gpsBuf.pop();
      }
      mBuf.unlock();

      //
      // Early reject by counting local delta movement (for equi-spereated kf drop)
      //
      odom_pose_prev = odom_pose_curr;
      odom_pose_curr = pose_curr;
      Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

      double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value.
      translationAccumulated += delta_translation;
      rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.

      if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap ) {
        isNowKeyFrame = true;
        translationAccumulated = 0.0; // reset
        rotaionAccumulated = 0.0; // reset
      } else {
        isNowKeyFrame = false;
      }

      if( ! isNowKeyFrame )
        continue;

      if( !gpsOffsetInitialized ) {
        if(hasGPSforThisKF) { // if the very first frame
          gpsAltitudeInitOffset = currGPS->altitude;
          gpsOffsetInitialized = true;
        }
      }

      //
      // Save data and Add consecutive node
      //
      pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
      downSizeFilterScancontext.setInputCloud(thisKeyFrame);
      downSizeFilterScancontext.filter(*thisKeyFrameDS);

      mKF.lock();
      keyframeLaserClouds.push_back(thisKeyFrameDS);
      keyframeLaserCloudsFull.push_back(thisKeyFrame);
      keyframePoses.push_back(pose_curr);
      keyframePosesUpdated.push_back(pose_curr); // init
      keyframeTimes.push_back(timeLaserOdometry);

      scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);

      laserCloudMapPGORedraw = true;
      mKF.unlock();

      const int prev_node_idx = keyframePoses.size() - 2;
      const int curr_node_idx = keyframePoses.size() - 1; // because cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
      if( ! gtSAMgraphMade /* prior node */) {
        const int init_node_idx = 0;
        gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));
        // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

        mtxPosegraph.lock();
        {
          // prior factor
          gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
          initialEstimate.insert(init_node_idx, poseOrigin);
          runISAM2opt();
        }
        mtxPosegraph.unlock();

        gtSAMgraphMade = true;

        cout << "posegraph prior node " << init_node_idx << " added" << endl;
      } else /* consecutive node (and odom factor) after the prior added */ { // == keyframePoses.size() > 1
        gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
        gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

        mtxPosegraph.lock();
        {
          // odom factor
          gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, poseFrom.between(poseTo), odomNoise));

          // gps factor
          if(hasGPSforThisKF) {
            double curr_altitude_offseted = currGPS->altitude - gpsAltitudeInitOffset;
            mtxRecentPose.lock();
            gtsam::Point3 gpsConstraint(recentOptimizedX, recentOptimizedY, curr_altitude_offseted); // in this example, only adjusting altitude (for x and y, very big noises are set)
            mtxRecentPose.unlock();
            gtSAMgraph.add(gtsam::GPSFactor(curr_node_idx, gpsConstraint, robustGPSNoise));
            cout << "GPS factor added at node " << curr_node_idx << endl;
          }
          initialEstimate.insert(curr_node_idx, poseTo);
           runISAM2opt();
        }
        mtxPosegraph.unlock();

        if(curr_node_idx % 100 == 0)
          cout << "posegraph odom node " << curr_node_idx << " added." << endl;
      }
      // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");

      // save utility
//      std::string curr_node_idx_str = padZeros(curr_node_idx);
//      pcl::io::savePCDFileBinary(pgScansDirectory + curr_node_idx_str + ".pcd", *thisKeyFrame); // scan
//      pgTimeSaveStream << timeLaser << std::endl; // path
    }

    // ps.
    // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
    // pub path and point cloud in another thread

    // wait (must required for running the while loop)
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
} // process_pg

void performSCLoopClosure(void)
{
  if( int(keyframePoses.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early
    return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) {
      const int prev_node_idx = SCclosestHistoryFrameID;
      const int curr_node_idx = keyframePoses.size() - 1; // because cpp starts 0 and ends n-1
      cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

      // Control the detection to see if we have reached the end
      count_same = (last_previous == prev_node_idx && last_curr == curr_node_idx) ? count_same + 1 : 0;
      last_previous = prev_node_idx;
      last_curr = curr_node_idx;
      // Vinicius - Mission is done, so call service to build the mesh
      if(count_same >= 10){ // reached the end, repeating on it
        mBuf.lock();
        laserCloudMapPGO->clear();
        std::vector<double> xs(keyframePosesUpdated.size()), ys(keyframePosesUpdated.size()), zs(keyframePosesUpdated.size());
        for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
          *laserCloudMapPGO += *local2global(keyframeLaserCloudsFull[node_idx], keyframePosesUpdated[node_idx]);
          xs[node_idx] = keyframePosesUpdated[node_idx].x;
          ys[node_idx] = keyframePosesUpdated[node_idx].y;
          zs[node_idx] = keyframePosesUpdated[node_idx].z;
        }
        //  downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
        //  downSizeFilterMapPGO.filter(*laserCloudMapPGO);

        mesh_open3d::cloud srv_msg;
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*laserCloudMapPGO, cloud_msg);
        laserCloudMapPGO->clear();
        srv_msg.request.cloud = cloud_msg;
        srv_msg.request.xs = xs;
        srv_msg.request.ys = ys;
        srv_msg.request.zs = zs;
        if(mesh_client.call(srv_msg))
          ROS_INFO("Building up the final mesh");
        else
          ROS_ERROR("Could not build the final mesh, check if server is working properly.");
        mBuf.unlock();

        // Finish this node
        ros::shutdown();
      }

      mBuf.lock();
      scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
      // addding actual 6D constraints in the other thread, icp_calculation.
      mBuf.unlock();
    }
} // performSCLoopClosure

void process_lcd(void)
{
  float loopClosureFrequency = 5.0; // can change
  ros::Rate rate(loopClosureFrequency);
  while (ros::ok())
  {
    rate.sleep();
    performSCLoopClosure();
    // performRSLoopClosure(); // TODO
  }
} // process_lcd

void process_icp(void)
{
  while(1)
//  while(subLaserOdometry.getNumPublishers() > 0 || std::abs((ros::Time::now() - t_control_input).toSec()) < 30) // Vinicius - while there are new messages
  {
    while ( !scLoopICPBuf.empty() )
    {
      if( scLoopICPBuf.size() > 30 ) {
        ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
      }

      mBuf.lock();
      std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
      scLoopICPBuf.pop();
      mBuf.unlock();

      const int prev_node_idx = loop_idx_pair.first;
      const int curr_node_idx = loop_idx_pair.second;
      auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
      if(relative_pose_optional) {
        gtsam::Pose3 relative_pose = relative_pose_optional.value();
        mtxPosegraph.lock();
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
        runISAM2opt();
        mtxPosegraph.unlock();
      }
    }

    // wait (must required for running the while loop)
    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }

} // process_icp

void process_viz_path(void)
{
  float hz = 10.0;
  ros::Rate rate(hz);
  while (ros::ok()) {
    rate.sleep();
    if(recentIdxUpdated > 1) {
      pubPath();
    }
  }
}

void process_isam(void)
{
  float hz = 2;
  ros::Rate rate(hz);
  while (ros::ok()) {
    rate.sleep();
    if( gtSAMgraphMade ) {
      mtxPosegraph.lock();
      runISAM2opt();
      cout << "running isam2 optimization ..." << endl;
      mtxPosegraph.unlock();

//      saveOptimizedVerticesKITTIformat(isamCurrentEstimate, pgKITTIformat); // pose
//      saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
    }
  }
}

void pubMap(void)
{
  int SKIP_FRAMES = 2; // sparse map visulalization to save computations
  int counter = 0;

  laserCloudMapPGO->clear();

  mKF.lock();
  // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()); node_idx++) {
  for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
    if(counter % SKIP_FRAMES == 0) {
      *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
    }
    counter++;
  }
  mKF.unlock();

  downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
  downSizeFilterMapPGO.filter(*laserCloudMapPGO);

  sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
  pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
  laserCloudMapPGOMsg.header.frame_id = "camera_init";
  pubMapAftPGO.publish(laserCloudMapPGOMsg);
}

void process_viz_map(void)
{
  float vizmapFrequency = 0.1; // 0.1 means run onces every 10s
  ros::Rate rate(vizmapFrequency);
  while (ros::ok()) {
    rate.sleep();
    if(recentIdxUpdated > 1) {
      pubMap();
    }
  }
} // pointcloud_viz


int main(int argc, char **argv)
{
  ros::init(argc, argv, "laserPGO");
  ros::NodeHandle nh;
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Error);

//  nh.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move
//  pgKITTIformat = save_directory + "optimized_poses.txt";
//  odomKITTIformat = save_directory + "odom_poses.txt";
//  pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out);
//  pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
//  pgScansDirectory = save_directory + "Scans/";
//  auto unused = system((std::string("exec rm -r ") + pgScansDirectory).c_str());
//  unused = system((std::string("mkdir -p ") + pgScansDirectory).c_str());

  string robot_name;
  nh.param<string>("robot_name", robot_name, "robot");

  nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 1.0); // pose assignment every k m move
  nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot
  keyframeRadGap = deg2rad(keyframeDegGap);

  nh.param<double>("sc_dist_thres", scDistThres, 0.2);
  nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor

  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  isam = new ISAM2(parameters);
  initNoises();

  scManager.setSCdistThres(scDistThres);
  scManager.setMaximumRadius(scMaximumRadius);

  float filter_size = 0.4;
  downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
  downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

  double mapVizFilterSize;
  nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.1); // pose assignment every k frames
  downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

  t_control_input = ros::Time::now(); // So we know when no more messages are coming
  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 100, laserCloudFullResHandler);
  subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, laserOdometryHandler);
  ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

  pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/"+robot_name+"/aft_pgo_odom", 100);
  pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/"+robot_name+"/repub_odom", 100);
  pubPathAftPGO = nh.advertise<nav_msgs::Path>("/"+robot_name+"/aft_pgo_path", 100);
  pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/"+robot_name+"/aft_pgo_map", 100);
  pubLastKf = nh.advertise<sensor_msgs::PointCloud2>("/"+robot_name+"/pgo_last_kf", 100);

  pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/"+robot_name+"/loop_scan_local", 100);
  pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/"+robot_name+"/loop_submap_local", 100);

  // Vinicius
  mesh_client = nh.serviceClient<mesh_open3d::cloud>("/"+robot_name+"/calculate_mesh");
  ros::Rate r(2);
  while (subLaserOdometry.getNumPublishers() == 0){
    ROS_INFO("SC node still waiting for cloud and odometry to start ...");
    r.sleep();
  }

  std::thread posegraph_slam {process_pg}; // pose graph construction
  std::thread lc_detection {process_lcd}; // loop closure detection
  std::thread icp_calculation {process_icp}; // loop constraint calculation via icp
//  std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added.

  std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
  std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

  ros::spin();

  return 0;
}
