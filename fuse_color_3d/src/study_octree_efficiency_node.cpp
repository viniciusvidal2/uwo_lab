#include <ros/ros.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <math.h>
#include <mutex>
#include <ctime>
#include <boost/filesystem.hpp>
#include <signal.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <rosgraph_msgs/Clock.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>

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
typedef PointXYZRGBNormal PointTN;
#define CLOUD_FRAME "body"
PointCloud<PointIn>::Ptr cloud_in, cloud_save;
PointCloud<PointT>::Ptr cloud_rgb, cloud_acc;
PassThrough<PointIn> pass;

// Robot name to organize logs
string robot_name;

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
ros::Publisher pub_raw_time;
ros::Publisher pub_it_time;
ros::Publisher pub_cent_time;
ros::Publisher pub_it_time_only;
ros::Publisher pub_cent_time_only;
ros::Publisher pub_it_iterations;
ros::Publisher pub_cent_iterations;

// Output message for odometry
nav_msgs::Odometry odom_msg_out;

// Time control for new messages, to check if we have finished our goal
ros::Time t_control_input;
bool new_data = false;
bool save_acc = false;

// Current clock time
ros::Time clk;

// Vectors to save the latency, process time and data sizes along the execution
vector<double> im_latencies, cloud_latencies, process_times;
vector<size_t> im_sizes, cloud_sizes_in, cloud_sizes_out;

Matrix3f euler2matrix(float r, float p, float y){
  // Ja recebe os angulos aqui em radianos
  return ( AngleAxisf(y, Vector3f::UnitY()) * AngleAxisf(r, Vector3f::UnitZ()) * AngleAxisf(p, Vector3f::UnitX()) ).matrix();
}

void filterCloudDepthCovariance(PointCloud<PointT>::Ptr cloud, int kn, float thresh, float depth){
  // Definir area a filtrar pela distancia dos pontos
  PointCloud<PointT>::Ptr close_area (new PointCloud<PointT>);
  PointCloud<PointT>::Ptr far_area   (new PointCloud<PointT>);
  for(size_t i=0; i<cloud->size(); i++){
    if(sqrt(pow(cloud->points[i].x, 2) + pow(cloud->points[i].y, 2) + pow(cloud->points[i].z, 2)) > depth)
      far_area->points.push_back(cloud->points[i]);
    else
      close_area->points.push_back(cloud->points[i]);
  }
  // Se houver area proxima relevante, calcular
  if(close_area->points.size() > kn){
    // Nuvens para filtrar
    PointCloud<PointT>::Ptr temp_out        (new PointCloud<PointT>);
    PointCloud<PointT>::Ptr marcar_outliers (new PointCloud<PointT>);
    marcar_outliers->resize(close_area->points.size());
    // Objetos para procurar na nuvem o centroide e a covariancia
    KdTreeFLANN<PointT>::Ptr tree (new KdTreeFLANN<PointT>);
    tree->setInputCloud(close_area);
    // Varrer todos os pontos atras da covariancia
#pragma omp parallel for
    for(size_t i=0; i<close_area->points.size(); i++){
      // Cria variaveis aqui dentro pelo processo ser paralelizado
      Eigen::Vector4f centroide;
      Eigen::Matrix4f rotacao_radial;
      Eigen::Matrix3f covariancia;
      rotacao_radial = Matrix4f::Identity();
      // Calcula angulo para rotacionar a nuvem e cria matriz de rotacao (yaw em torno de Y, pitch em torno de X)
      float yaw_y   = -atan2(close_area->points[i].x, close_area->points[i].z);
      float pitch_x = -atan2(close_area->points[i].y, close_area->points[i].z);
      Eigen::Matrix3f rot = euler2matrix(0, pitch_x, yaw_y);
      rotacao_radial.block<3,3>(0, 0) << rot;
      // Calcula vizinhos mais proximos aqui por raio ou K neighbors
      vector<int> indices_vizinhos;
      vector<float> distancias_vizinhos;
      tree->nearestKSearch(int(i), kn, indices_vizinhos, distancias_vizinhos);
      // Separa nuvem com esses vizinhos
      PointCloud<PointT>::Ptr temp (new PointCloud<PointT>);
      temp->points.resize(indices_vizinhos.size());
#pragma omp parallel for
      for(size_t j=0; j<indices_vizinhos.size(); j++)
        temp->points[j] = close_area->points[ indices_vizinhos[j] ];
      // Rotaciona a nuvem separada segundo o raio que sai do centro do laser (origem)
      transformPointCloud(*temp, *temp, rotacao_radial);
      // Calcula centroide e covariancia da nuvem
      compute3DCentroid(*temp, centroide);
      computeCovarianceMatrix(*temp, centroide, covariancia);
      // Se for muito maior em z que em x e y, considera ruim e marca na nuvem
      if(covariancia(2, 2) > thresh*covariancia(0, 0) && covariancia(2, 2) > thresh*covariancia(1, 1))
        marcar_outliers->points[i].x = 1;
      else
        marcar_outliers->points[i].x = 0;
    }
    // Passa rapidamente para nuvem de saida
    for(size_t i=0; i<marcar_outliers->size(); i++){
      if(marcar_outliers->points[i].x == 0)
        temp_out->push_back(close_area->points[i]);
    }
    marcar_outliers->clear();
    *close_area = *temp_out;
    temp_out->clear();
    *cloud = *close_area + *far_area;
    close_area->clear(); far_area->clear();
  }
}

void divideInOctreeLevels(PointCloud<PointT>::Ptr cloud, vector<PointCloud<PointT>> &leafs, float level){
  /// Obter os limites de dimensao da nuvem de entrada
  PointT min_limits, max_limits;
  getMinMax3D(*cloud, min_limits, max_limits);
  // Se a nuvem variar consideravelmente em todas as dimensoes, aumentar o level automaticamente
  float dl = 5; // [m]
  if(abs(max_limits.x - min_limits.x) > dl && abs(max_limits.x - min_limits.x) > dl && abs(max_limits.x - min_limits.x) > dl)
    level *= 2;
  /// De acordo com a quantidade de niveis da octree, calcular os centroides das folhas da arvore
  // Dimensoes da caixa que corresponde a folha
  float stepx, stepy, stepz;
  stepx = abs(max_limits.x - min_limits.x)/level;
  stepy = abs(max_limits.y - min_limits.y)/level;
  stepz = abs(max_limits.z - min_limits.z)/level;
  // Centros em que vamos caminhar naquela dimensao para cada folha
  vector<float> centros_x, centros_y, centros_z;
  // Se temos bastante variacao, dividir, senao mantem somente uma divisao ali na dimensao
  float tol = 0.1; // [m]
  if(stepx > tol)
    centros_x.resize(size_t(level));
  else
    centros_x.resize(size_t(level/2));
  if(stepy > tol)
    centros_y.resize(size_t(level));
  else
    centros_y.resize(size_t(level/2));
  if(stepz > tol)
    centros_z.resize(size_t(level));
  else
    centros_z.resize(size_t(level/2));
  for(int i=0; i<centros_x.size(); i++)
    centros_x[i] = min_limits.x + stepx/2 + float(i)*stepx;
  for(int i=0; i<centros_y.size(); i++)
    centros_y[i] = min_limits.y + stepy/2 + float(i)*stepy;
  for(int i=0; i<centros_z.size(); i++)
    centros_z[i] = min_limits.z + stepz/2 + float(i)*stepz;
  // Montar a nuvem de pontos com os centroides combinados
  PointCloud<PointT>::Ptr centroides (new PointCloud<PointT>);
  for(int i=0; i<centros_x.size(); i++){
    for(int j=0; j<centros_y.size(); j++){
      for(int k=0; k<centros_z.size(); k++){
        PointT c;
        c.x = centros_x[i]; c.y = centros_y[j]; c.z = centros_z[k];
        c.r = 250; c.b = 0; c.g = 0;
        centroides->points.push_back(c);
      }
    }
  }

  // if(centroides->points.size() < 10)
  //   return;
  // Colocar o numero de folhas como o tamanho do vetor de saida
  leafs.resize(centroides->points.size());
  /// Iterar sobre a nuvem para colocar cada ponto na sua folha
  /// Descobrir qual centroide esta mais perto por KdTree
  KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(centroides);
  vector<int> indices;
  vector<float> distances;
  for(size_t i=0; i<cloud->points.size(); i++){
    kdtree.nearestKSearch(cloud->points[i], 1, indices, distances);
    if(indices.size() == 1 && indices[0] >= 0 && indices[0] < centroides->points.size())
      leafs[indices[0]].push_back(cloud->points[i]);
  }
}

void divideInCentroidOctreeLevels(PointCloud<PointT>::Ptr cloud, vector<PointCloud<PointT>> &clusters, size_t thresh){
  // Create temp point cloud to be split into clusters
  vector<PointCloud<PointT>> temp(1);
  temp[0] = *cloud;
  // Temporary cluster vector with 8 regions
  vector<PointCloud<PointT>> curr_clusters(8);

  // while not all the clusters are small enough - there are points in temm point cloud to be split
  int n_iterations = 0;
  while(temp.size() > 0)
  {
    // Calculate the point cloud centroid
    Vector4f centroid;
    compute3DCentroid(temp[0], centroid);
    // Split into 8 clusters from coordinates logic
    for(auto p:temp[0].points){
      if(p.x <= centroid[0] && p.y <= centroid[1] && p.z <= centroid[2]){
        curr_clusters[0].points.emplace_back(p);
      } else if(p.x <= centroid[0] && p.y <= centroid[1] && p.z > centroid[2]) {
        curr_clusters[1].points.emplace_back(p);
      } else if(p.x <= centroid[0] && p.y > centroid[1] && p.z <= centroid[2]) {
        curr_clusters[2].points.emplace_back(p);
      } else if(p.x <= centroid[0] && p.y > centroid[1] && p.z > centroid[2]) {
        curr_clusters[3].points.emplace_back(p);
      } else if(p.x > centroid[0] && p.y <= centroid[1] && p.z <= centroid[2]) {
        curr_clusters[4].points.emplace_back(p);
      } else if(p.x > centroid[0] && p.y <= centroid[1] && p.z > centroid[2]) {
        curr_clusters[5].points.emplace_back(p);
      } else if(p.x > centroid[0] && p.y > centroid[1] && p.z <= centroid[2]) {
        curr_clusters[6].points.emplace_back(p);
      } else if(p.x > centroid[0] && p.y > centroid[1] && p.z > centroid[2]) {
        curr_clusters[7].points.emplace_back(p);
      }
    }

    // Push back to cluster vector
    PointCloud<PointT> cluster;
    for (size_t i=0; i < curr_clusters.size(); i++){
      if(curr_clusters[i].points.size() <= 20 )
        continue;
      if(curr_clusters[i].points.size() > 20 && curr_clusters[i].points.size() <= thresh)
      {
        cluster = curr_clusters[i];
        clusters.emplace_back(curr_clusters[i]);
      }
      // If cluster is too big, add it to keep being split next iteration
      if(curr_clusters[i].points.size() > thresh)
      {
        cluster = curr_clusters[i];
        temp.emplace_back(cluster);
      }
      curr_clusters[i].clear();
    }
    // Clear the current inspected temporary point cloud section
    temp.erase(temp.begin());
    n_iterations++;
  }
  std_msgs::Int32 iterations_msg;
  iterations_msg.data = n_iterations;
  pub_cent_iterations.publish(iterations_msg);
}

void divideInIterativeOctreeLevels2(PointCloud<PointT>::Ptr cloud, vector<PointCloud<PointT>> &clusters, size_t thresh){
  // Create temp point cloud to be split into clusters
  vector<PointCloud<PointT>> temp(1);
  temp[0] = *cloud;
  // Temporary cluster vector with 8 regions
  vector<PointCloud<PointT>> curr_clusters(8);

  // while not all the clusters are small enough - there are points in temm point cloud to be split
  int n_iterations = 0;
  while(temp.size() > 0)
  {
    // Calculate the point cloud centroid
    PointT min_limits, max_limits;
    getMinMax3D(temp[0], min_limits, max_limits);
    Vector3f centroid {(min_limits.x + max_limits.x)/2, (min_limits.y + max_limits.y)/2, (min_limits.z + max_limits.z)/2};
    // Split into 8 clusters from coordinates logic
    for(auto p:temp[0].points){
      if(p.x <= centroid[0] && p.y <= centroid[1] && p.z <= centroid[2]){
        curr_clusters[0].points.emplace_back(p);
      } else if(p.x <= centroid[0] && p.y <= centroid[1] && p.z > centroid[2]) {
        curr_clusters[1].points.emplace_back(p);
      } else if(p.x <= centroid[0] && p.y > centroid[1] && p.z <= centroid[2]) {
        curr_clusters[2].points.emplace_back(p);
      } else if(p.x <= centroid[0] && p.y > centroid[1] && p.z > centroid[2]) {
        curr_clusters[3].points.emplace_back(p);
      } else if(p.x > centroid[0] && p.y <= centroid[1] && p.z <= centroid[2]) {
        curr_clusters[4].points.emplace_back(p);
      } else if(p.x > centroid[0] && p.y <= centroid[1] && p.z > centroid[2]) {
        curr_clusters[5].points.emplace_back(p);
      } else if(p.x > centroid[0] && p.y > centroid[1] && p.z <= centroid[2]) {
        curr_clusters[6].points.emplace_back(p);
      } else if(p.x > centroid[0] && p.y > centroid[1] && p.z > centroid[2]) {
        curr_clusters[7].points.emplace_back(p);
      }
    }

    // Push back to cluster vector
    PointCloud<PointT> cluster;
    for (size_t i=0; i < curr_clusters.size(); i++){
      if(curr_clusters[i].points.size() <= 20 )
        continue;
      if(curr_clusters[i].points.size() > 20 && curr_clusters[i].points.size() <= thresh)
      {
        cluster = curr_clusters[i];
        clusters.emplace_back(curr_clusters[i]);
      }
      // If cluster is too big, add it to keep being split next iteration
      if(curr_clusters[i].points.size() > thresh)
      {
        cluster = curr_clusters[i];
        temp.emplace_back(cluster);
      }
      curr_clusters[i].clear();
    }
    // Clear the current inspected temporary point cloud section
    temp.erase(temp.begin());
    n_iterations++;
  }
  std_msgs::Int32 iterations_msg;
  iterations_msg.data = n_iterations;
  pub_it_iterations.publish(iterations_msg);
}

void calculateNormals(PointCloud<PointT>::Ptr in, PointCloud<PointTN>::Ptr acc_normal, int threads){
  if (in->points.size() < 10)
    return;
  // Vetor de zeros simbolizando a origem
  Eigen::Vector3f C = Eigen::Vector3f::Zero();
  // Remove pontos nan aqui para nao correr risco de derrubar o calculo das normais
  std::vector<int> indicesnan;
  removeNaNFromPointCloud<PointTN>(*acc_normal, *acc_normal, indicesnan);
  // Inicia estimador de normais
  search::KdTree<PointT>::Ptr tree (new search::KdTree<PointT>());
  NormalEstimationOMP<PointT, Normal> ne;
  ne.setInputCloud(in);
  ne.setSearchMethod(tree);
  ne.setKSearch(100);
  ne.setNumberOfThreads(threads);
  // Nuvem de normais calculada
  PointCloud<Normal>::Ptr cloud_normals (new PointCloud<Normal>());
  ne.compute(*cloud_normals);
  // Adiciona saida na nuvem concatenada PointTN
  concatenateFields(*in, *cloud_normals, *acc_normal);
  // Filtra por normais problematicas
  removeNaNNormalsFromPointCloud(*acc_normal, *acc_normal, indicesnan);

  // Forcar virar as normais na marra para a origem
#pragma omp parallel for
  for(size_t i=0; i < acc_normal->points.size(); i++){
    Eigen::Vector3f normal, cp;
    normal << acc_normal->points[i].normal_x, acc_normal->points[i].normal_y, acc_normal->points[i].normal_z;
    cp     << C(0)-acc_normal->points[i].x  , C(1)-acc_normal->points[i].y  , C(2)-acc_normal->points[i].z  ;
    float cos_theta = (normal.dot(cp))/(normal.norm()*cp.norm());
    if(cos_theta <= 0){ // Esta apontando errado, deve inverter
      acc_normal->points[i].normal_x = -acc_normal->points[i].normal_x;
      acc_normal->points[i].normal_y = -acc_normal->points[i].normal_y;
      acc_normal->points[i].normal_z = -acc_normal->points[i].normal_z;
    }
  }
}

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
  // Convert the message
  fromROSMsg(*cl_msg, *cloud_in);

  new_data = true;
}

/// Image callback
///
void imageCallback(const sensor_msgs::CompressedImageConstPtr &msg){
  cv_bridge::CvImagePtr cam_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  // Undistort the image
  undistort(cam_img->image, image_undistorted, K_, dist_coefs);
}

void process_raw_data(){
    pcl::PointCloud<PointT>::Ptr prd (new PointCloud<PointT>);
    pcl::PointCloud<PointTN>::Ptr out (new PointCloud<PointTN>);
    *prd = *cloud_rgb;
    // Covariance
    float depth_cov_filter = 3;
    filterCloudDepthCovariance(prd, 30, 1.5, depth_cov_filter);
    // SOR
    StatisticalOutlierRemoval<PointT> sor;
    sor.setMeanK(30);
    sor.setStddevMulThresh(2);
    sor.setNegative(false);
    sor.setInputCloud(prd);
    sor.filter(*prd);
    // Normals
    calculateNormals(prd, out, 1);    
}

void process_iterative_octree(){
    pcl::PointCloud<PointT>::Ptr prd (new PointCloud<PointT>);
    pcl::PointCloud<PointTN>::Ptr out (new PointCloud<PointTN>);
    *prd = *cloud_rgb;
    ros::Time t1 = ros::Time::now();
    // Separando a nuvem de entrada em clusters
    vector<PointCloud<PointT>> cin_clusters;
    divideInIterativeOctreeLevels2(prd, cin_clusters, 100);
    // // Continuar separando enquanto houver algum maior que 30 mil pontos
    // bool so_pequenos = false;
    // size_t indice_grande;
    // PointCloud<PointT>::Ptr nuvem_grande (new PointCloud<PointT>);
    // vector<PointCloud<PointT>> big_clusters;
    // int n_iterations = 1;
    // while(!so_pequenos){
    //   so_pequenos = true;
    //   // Se achar algum cluster grande ainda, separar indice e nuvem correspondente
    //   for(size_t i=0; i<cin_clusters.size(); i++){
    //     if(cin_clusters[i].size() > 100){
    //       so_pequenos = false;
    //       indice_grande = i;
    //       *nuvem_grande = cin_clusters[i];
    //       n_iterations ++;
    //       break;
    //     }
    //     // Matar cluster pequeno
    //     if(cin_clusters[i].size() < 20) 
    //       cin_clusters[i].clear();
    //   }
    //   // Se foi achado algum cluster grande, processar ele, substituir o original com o primeiro cluster obtido e adicionar
    //   // o restante dos clusters ao final da nuvem de clusters
    //   if(!so_pequenos && nuvem_grande->size() > 0 && cin_clusters.size() > 0){
    //     divideInOctreeLevels(nuvem_grande, big_clusters, 2);
    //     cin_clusters[indice_grande] = big_clusters[0];
    //     cin_clusters.insert(cin_clusters.end(), big_clusters.begin()+1, big_clusters.end());
    //     big_clusters.clear(); nuvem_grande->clear();
    //   }
    // }
    std_msgs::Float64 time_only_octree;
    time_only_octree.data = (ros::Time::now() - t1).toSec()*1000;
    pub_it_time_only.publish(time_only_octree);
    // std_msgs::Int32 iterations_msg;
    // iterations_msg.data = n_iterations;
    // pub_it_iterations.publish(iterations_msg);
    // Para cada cluster, filtrar e retornar no vetor de filtradas
    vector<PointCloud<PointTN>> out_clusters(cin_clusters.size());
#pragma omp parallel for
    for(size_t i=0; i<out_clusters.size(); i++){
      PointCloud<PointT>::Ptr temp (new PointCloud<PointT>);
      if (cin_clusters[i].points.size() > 30){
        PointCloud<PointT>::Ptr temp (new PointCloud<PointT>);
        *temp = cin_clusters[i];
        // Calcular filtro de covariancia na regiao mais proxima
        float depth_cov_filter = 3;
        filterCloudDepthCovariance(temp, 30, 1.5, depth_cov_filter);
        // Filtro de ruidos aleatorios
        StatisticalOutlierRemoval<PointT> sor;
        sor.setMeanK(30);
        sor.setStddevMulThresh(2);
        sor.setNegative(false);
        sor.setInputCloud(temp);
        sor.filter(*temp);
        // Normais apos tudo filtrado
        pcl::PointCloud<PointTN>::Ptr out_temp (new PointCloud<PointTN>);
        calculateNormals(temp, out_temp, 20);
        out_clusters[i] = *out_temp;
      } else {
        pcl::PointCloud<PointTN>::Ptr out_temp (new PointCloud<PointTN>);
        calculateNormals(temp, out_temp, 20);
        out_clusters[i] = *out_temp;
      }
    } 
    // Somar todos os clusters filtrados de volta na nuvem de saida
    for(size_t i=0; i<out_clusters.size(); i++)
      *out += out_clusters[i];    
}

void process_centroid_octree(){
    pcl::PointCloud<PointT>::Ptr prd (new PointCloud<PointT>);
    pcl::PointCloud<PointTN>::Ptr out (new PointCloud<PointTN>);
    *prd = *cloud_rgb;
    // Separando a nuvem de entrada em clusters
    vector<PointCloud<PointT>> cin_clusters;
    ros::Time t1 = ros::Time::now();
    divideInCentroidOctreeLevels(prd, cin_clusters, 100);
    std_msgs::Float64 time_only_octree;
    time_only_octree.data = (ros::Time::now() - t1).toSec()*1000;
    pub_cent_time_only.publish(time_only_octree);
    // Para cada cluster, filtrar e retornar no vetor de filtradas
    vector<PointCloud<PointTN>> out_clusters(cin_clusters.size());
#pragma omp parallel for
    for(size_t i=0; i<out_clusters.size(); i++){
      PointCloud<PointT>::Ptr temp (new PointCloud<PointT>);
      *temp = cin_clusters[i];
      if (cin_clusters[i].points.size() > 30){
        // Calcular filtro de covariancia na regiao mais proxima
        float depth_cov_filter = 3;
        filterCloudDepthCovariance(temp, 30, 1.5, depth_cov_filter);
        // Filtro de ruidos aleatorios
        StatisticalOutlierRemoval<PointT> sor;
        sor.setMeanK(30);
        sor.setStddevMulThresh(2);
        sor.setNegative(false);
        sor.setInputCloud(temp);
        sor.filter(*temp);
        // Normais apos tudo filtrado
        pcl::PointCloud<PointTN>::Ptr out_temp (new PointCloud<PointTN>);
        calculateNormals(temp, out_temp, 20);
        out_clusters[i] = *out_temp;
      } else {
        pcl::PointCloud<PointTN>::Ptr out_temp (new PointCloud<PointTN>);
        calculateNormals(temp, out_temp, 20);
        out_clusters[i] = *out_temp;
      }
    }
    // Somar todos os clusters filtrados de volta na nuvem de saida
    for(size_t i=0; i<out_clusters.size(); i++)
      *out += out_clusters[i];  
}

/// Process callback
///
void processCallback(const ros::TimerEvent&){
  if(new_data){
    ros::Time start = ros::Time::now();

    // Lock mutex
    mtx.lock();

    pass.setInputCloud(cloud_in);
    pass.filter(*cloud_in);

    // Copy the point cloud to the one with RGB
    copyPointCloud(*cloud_in, *cloud_rgb);

    // Color cloud
    colorCloudCPU(cloud_rgb, image_undistorted);

    // Process raw data
    ROS_INFO("POINT CLOUD SIZE: %zu", cloud_rgb->points.size());
    ros::Time t1 = ros::Time::now();
    process_raw_data();
    std_msgs::Float64 time_data;
    time_data.data = 3*(ros::Time::now() - t1).toSec();
    pub_raw_time.publish(time_data);
    ROS_INFO("Time for RAW: %.5f", time_data.data);

    // Process with octree
    t1 = ros::Time::now();
    process_iterative_octree();
    time_data.data = 2*(ros::Time::now() - t1).toSec();
    pub_it_time.publish(time_data);
    ROS_INFO("Time for IT: %.5f", time_data.data);

    // Process with octree modified with centroids calculation
    t1 = ros::Time::now();
    process_centroid_octree();
    time_data.data = (ros::Time::now() - t1).toSec();
    pub_cent_time.publish(time_data);
    ROS_INFO("Time for CENT: %.5f", time_data.data);

    new_data = false;

    // Free mutex
    mtx.unlock();
  }
}

/// MAIN
///
int main(int argc, char **argv)
{
  ros::init(argc, argv, "study_octree_node", ros::init_options::NoSigintHandler);
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

  // Diagnostics publishers
  pub_raw_time = nh.advertise<std_msgs::Float64>("diagnostics/raw_time", 100);
  pub_it_time = nh.advertise<std_msgs::Float64>("diagnostics/iterative_octree_time", 100);
  pub_cent_time = nh.advertise<std_msgs::Float64>("diagnostics/centroid_octree_time", 100);
  pub_it_iterations = nh.advertise<std_msgs::Int32>("diagnostics/iterative_octree_iterations", 100);
  pub_cent_iterations = nh.advertise<std_msgs::Int32>("diagnostics/centroid_octree_iterations", 100);
  pub_it_time_only = nh.advertise<std_msgs::Float64>("diagnostics/iterative_octree_algorithm_time", 100);
  pub_cent_time_only = nh.advertise<std_msgs::Float64>("diagnostics/centroid_octree_algorithm_time", 100);

  // Initialize sync subscribers
  string image_topic, cloud_topic, odometry_topic;
  n_.param<string>("input_topics/image_topic", image_topic, "/zed2/zed_node/left/image_rect_color/compressed");
  n_.param<string>("input_topics/cloud_topic", cloud_topic, "/cloud_registered_body");
  n_.param<string>("input_topics/odometry_topic", odometry_topic, "/Odometry");

  im_sub = nh.subscribe(image_topic, 1, &imageCallback);
  ros::Rate r(5);
  while (im_sub.getNumPublishers() < 1){
    r.sleep();
    ROS_WARN("Waiting for the image topic to show up ...");
  }
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, cloud_topic, 10000);
  message_filters::Subscriber<nav_msgs::Odometry>       odom_sub(nh, odometry_topic, 10000);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> sync_pol;
  message_filters::Synchronizer<sync_pol> sync(sync_pol(10000), cloud_sub, odom_sub);
  sync.registerCallback(boost::bind(&syncCallback, _1, _2));
  ros::SubscribeOptions ops;
  ops.allow_concurrent_callbacks = true;
  t_control_input = ros::Time::now();

  ROS_INFO("Listening to sensors data ...");

  // Filter for far points
  // pass.setFilterFieldName("x");
  // pass.setFilterLimits(0.0, 30.0);

  ros::Timer timer = nh.createTimer(ros::Duration(0.1), processCallback);

//   ros::MultiThreadedSpinner spinner(4);
//   spinner.spin();
  ros::spin();

  return 0;
}