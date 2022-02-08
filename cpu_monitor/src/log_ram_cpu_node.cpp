#include <ros/ros.h>
#include <stdio.h>
#include <math.h>
#include <mutex>
#include <string>
#include <vector>
#include <ctime>
#include <boost/filesystem.hpp>
#include <sys/types.h>
#include <dirent.h>

#include <std_msgs/String.h>
#include <std_msgs/UInt64.h>
#include <std_msgs/Float32.h>

using namespace std;

vector<size_t> rams;
vector<float> cpus;
mutex mtx;
float current_total_cpu;

void ramCallback(const std_msgs::UInt64ConstPtr &msg)
{
  // Save the current RAM for further logs
  rams.emplace_back(msg->data);
}

void cpuCallback(const std_msgs::Float32ConstPtr &msg)
{
  // Save the current CPU for further logs
  cpus.emplace_back(msg->data);
}

void processCallback(const std_msgs::Float32ConstPtr &msg)
{
  current_total_cpu = msg->data;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "log_ram_cpu_node");
  ros::NodeHandle nh;
  ros::NodeHandle n_("~");

  float pct_cpu_idle;
  string robot_name, node_name, network_entity;
  n_.param<string>("robot_name", robot_name, "robot");
  n_.param<string>("node_name", node_name, "node");
  n_.param<string>("network_entity", network_entity, "fog");
  n_.param<float>("pct_cpu_idle", pct_cpu_idle, 15);
  pct_cpu_idle = 15.0; // easier this way

  string ram_topic_name = "/"+robot_name+"/"+network_entity+"_cpu_monitor/"+robot_name+"/"+node_name+"/mem";
  string cpu_topic_name = "/"+robot_name+"/"+network_entity+"_cpu_monitor/"+robot_name+"/"+node_name+"/cpu";
  ros::Subscriber sub1 = nh.subscribe(ram_topic_name, 100, &ramCallback);
  ros::Subscriber sub2 = nh.subscribe(cpu_topic_name, 100, &cpuCallback);

  ros::Rate r1(1), r2(10);
  while(sub1.getNumPublishers() == 0 || sub2.getNumPublishers() == 0){
    ROS_INFO("Waiting for the logging messages in node %s ...", node_name.c_str());
    r1.sleep();
  }

  // This subscribes the whole percentage, so we can monitor if the processes are over
  string mon_topic_name = "/"+robot_name+"/"+network_entity+"_cpu_monitor/total_cpu";
  ros::Subscriber sub3 = nh.subscribe(mon_topic_name, 100, &processCallback);

  while(ros::ok()){
    r2.sleep();
    ros::spinOnce();

    if(current_total_cpu <= pct_cpu_idle && pct_cpu_idle > 0 && cpus.size() > 15){
      // Wait 5 seconds
      ros::Rate rr(2);
      for (int i=0; i<10; i++){
        rr.sleep();
        ros::spinOnce();
      }
      // Save log
      mtx.lock();
      // Check if the folder exists
      string log_dir = string(getenv("HOME")) + "/Desktop/" + robot_name + "_log/";
      if(!opendir(log_dir.c_str()))
        boost::filesystem::create_directory(log_dir.c_str());
      ROS_INFO("Writing logs to %s ...", log_dir.c_str());
      // Open the files in the folder
      FILE *fp;
      // For each data type, write the results to the file
      fp = fopen((log_dir+"/ram_"+node_name+".txt").c_str(),"w");
      for(auto ra:rams)
        fprintf(fp, "%zu\n", ra);
      fclose(fp);
      fp = fopen((log_dir+"/cpu_"+node_name+".txt").c_str(),"w");
      for(auto c:cpus)
        fprintf(fp, "%.2f\n", c);
      fclose(fp);
      mtx.unlock();
      ROS_INFO("Wrote logs for node %s!", node_name.c_str());
      ros::shutdown();
    }
  }

  return 0;
}
