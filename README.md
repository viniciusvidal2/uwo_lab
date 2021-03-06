# uwo_lab
This package contains all the dependencies and source code for autonomous navigation and sensor fusion developed in the UWO Computer Engineering lab.

The intention is to evaluate State of the Art algorithms and techniques running at the edge, fog and cloud, using 5G conenction and real world experiments. Plenty of modern equipment and sensors are used, such as Livox lidars and ZED2 camera, with embedded processor from NVidia.

Some dependencies were adapted to better cope with the computational constraints, but the respective installation procedures can still be found in their original repository.

Check each algorithm for their respective dependencies, being them:
## [livox_ros_driver](https://github.com/Livox-SDK/livox_ros_driver/tree/880c46a91aaa602dbecf20e204da4751747b3826)
## [FAST_LIO2](https://github.com/viniciusvidal2/FAST_LIO/tree/232eefb6c982319bf56e3d8dd3a668e538719002)
## [zed-ros-wrapper](https://github.com/viniciusvidal2/zed-ros-wrapper/tree/dcd9d972e62c2fd2bd4d3114d70194934fad364c)
## [Scan-Context](https://github.com/irapkaist/scancontext)

The original repositories were forked, so some modifications could be performed in order to adapt to our needs.

## Original packages

These packages were developed to perform tasks not seen in the original algorithms. Some are meant to be performed only in a fog tier.

### Mesh calculation
The mesh is calculated by parallel point cloud processing using the Open3D library, converting the ROS data, that used PCL.

### Point cloud - image fusion
The camera is callibrated with the point cloud, and the collor is attributed to each point via parallel projection using OpenMP.
