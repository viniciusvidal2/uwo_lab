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

vector<float> cpus;
mutex mtx;
float current_total_cpu;

void cpuCallback(const std_msgs::Float32ConstPtr &msg)
{
  // Save the current CPU for further logs
  current_total_cpu = msg->data;
  cpus.emplace_back(current_total_cpu);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "log_cpu_total");
  ros::NodeHandle nh;
  ros::NodeHandle n_("~");

  string robot_name, entity_name;
  n_.param<string>("robot_name", robot_name, "robot");
  n_.param<string>("entity_name", entity_name, "fog");
  float pct_cpu_idle = 15.0;

  string cpu_topic_name = "/"+robot_name+"/"+entity_name+"_cpu_monitor/total_cpu";
  ros::Subscriber sub2 = nh.subscribe(cpu_topic_name, 100, &cpuCallback);

  ros::Rate r1(1), r2(10);
  while(sub2.getNumPublishers() == 0){
    ROS_INFO("Waiting for the logging messages ...");
    r1.sleep();
  }

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
      fp = fopen((log_dir+"/cpu_total_"+entity_name+".txt").c_str(),"w");
      for(auto c:cpus)
        fprintf(fp, "%.2f\n", c);
      fclose(fp);
      mtx.unlock();
      ROS_INFO("Wrote logs for node log_cpu_total at %s !", entity_name.c_str());
      ros::shutdown();
    }
  }
  return 0;
}
