#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from mesh_open3d.srv import cloud, cloudResponse

global mesh_save_directory, robot_name


def handle_cloud(req):
    global mesh_save_directory, robot_name
    # Build the open3d point cloud
    ptc = o3d.geometry.PointCloud()
    ptc.points = o3d.utility.Vector3dVector([])
    ptc.colors = o3d.utility.Vector3dVector([])
    ptc.normals = o3d.utility.Vector3dVector([])

    # Estimate mesh
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ptc, depth=11)
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # Return response
    return cloudResponse(True)


def run_server():
    global mesh_save_directory, robot_name
    rospy.init_node('mesh_server_node')

    robot_name = rospy.get_param('robot_name')
    mesh_save_directory = rospy.get_param('mesh_save_directory')

    s = rospy.Service('/'+robot_name+'/calculate_mesh', cloud, handle_cloud)
    rospy.spin()


if __name__ == "__main__":
    run_server()
