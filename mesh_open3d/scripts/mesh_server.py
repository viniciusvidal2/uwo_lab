#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import os
import sys
import rospy
from mesh_open3d.srv import cloud, cloudResponse

global mesh_save_directory, robot_name, mesh_built


def handle_cloud(req):
    rospy.loginfo('Received request to build mesh ...')
    global mesh_save_directory, robot_name, mesh_built
    # Build the open3d point cloud
    points = []
    colors = []
    normals = []
    for i in range(len(req.x)):
        points.append(np.array([req.x[i], req.y[i], req.z[i]]))
        colors.append(np.array([req.r[i]/255, req.g[i]/255, req.b[i]/255]))
        normals.append(np.array([req.nx[i], req.ny[i], req.nz[i]]))

    ptc = o3d.geometry.PointCloud()
    ptc.points = o3d.utility.Vector3dVector(np.asarray(points))
    ptc.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    ptc.normals = o3d.utility.Vector3dVector(np.asarray(normals))

    # Estimate mesh
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ptc, depth=11)
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)

    # Save mesh
    rospy.loginfo('Saving mesh ...')
    o3d.io.write_triangle_mesh(os.path.join(mesh_save_directory, 'output_mesh.ply'), mesh)

    # Return response
    rospy.loginfo('Final mesh is saved, killing the node ...')
    mesh_built = True
    return cloudResponse(True)


def run_server():
    global mesh_save_directory, robot_name, mesh_built
    rospy.init_node('mesh_server_node')

    mesh_save_directory = sys.argv[1]
    robot_name = sys.argv[2]
    mesh_built = False

    s = rospy.Service('/'+robot_name+'/calculate_mesh', cloud, handle_cloud)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
        if mesh_built:
            rospy.signal_shutdown('Destroying mesh server node ...')



if __name__ == "__main__":
    rospy.loginfo('Initializing MESH server ...')
    run_server()
