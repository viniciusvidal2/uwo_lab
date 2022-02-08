import numpy as np
import os
import pandas as pd
from work_data import work_cpu, work_ram, work_latency


def read_log_file(f, name):
    file_name = os.path.join(f, name)
    return np.squeeze(pd.read_fwf(file_name).to_numpy(dtype=float))


def filter_array_mean_std_dev(a, dev):
    mean = np.mean(a, axis=0)
    sd = np.std(a, axis=0)
    return mean, sd, np.asarray([x for x in a if (mean - dev*sd < x < mean + dev*sd)])


class Struct:
    pass


if __name__ == '__main__':
    # Define the test folder to read data from
    robot_test_name = 'robot'+'_log'
    folder = os.path.join(os.getenv('HOME'), 'Desktop', robot_test_name)
    # Architecture, from 0 to 4
    architecture = 0

    ####################################################################
    ###### Read files
    ####################################################################
    ## Latency
    lat = Struct()
    lat.source_fastlio_cloud = read_log_file(folder, 'latencies_source_fastlio_cloud.txt')
    lat.source_fusecolor_im = read_log_file(folder, 'latencies_source_fusecolor_im.txt')
    lat.fastlio_fusecolor_cloud = read_log_file(folder, 'latencies_fastlio_fusecolor_cloud.txt')
    lat.fusecolor_scancontext_cloud = read_log_file(folder, 'latencies_fusecolor_scancontext_cloud.txt')
    ## Message sizes
    msg_size = Struct()
    msg_size.fusecolor_im = read_log_file(folder, 'messagesize_fusecolor_im.txt')
    msg_size.fusecolor_cloudin = read_log_file(folder, 'messagesize_fusecolor_cloudin.txt')
    msg_size.fusecolor_cloudout = read_log_file(folder, 'messagesize_fusecolor_cloudout.txt')
    ## CPU usage
    cpu = Struct()
    if os.path.isfile(os.path.join(folder, 'cpu_total_edge.txt')):
        cpu.total_edge = read_log_file(folder, 'cpu_total_edge.txt')
    cpu.total_fog = read_log_file(folder, 'cpu_total_fog.txt')
    cpu.scan_context_node = read_log_file(folder, 'cpu_scan_context_node.txt')
    cpu.project_image_pointcloud_node = read_log_file(folder, 'cpu_project_image_pointcloud_node.txt')
    cpu.final_mesh_server_node = read_log_file(folder, 'cpu_final_mesh_server_node.txt')
    cpu.fastlio_node = read_log_file(folder, 'cpu_FASTLIO_node.txt')
    ## RAM consumption
    ram = Struct()
    ram.fastlio_node = read_log_file(folder, 'ram_FASTLIO_node.txt')
    ram.final_mesh_server_node = read_log_file(folder, 'ram_final_mesh_server_node.txt')
    ram.project_image_pointcloud_node = read_log_file(folder, 'ram_project_image_pointcloud_node.txt')
    ram.scan_context_node = read_log_file(folder, 'ram_scan_context_node.txt')
    ## Process time
    pt = Struct()
    processtime_scancontext = read_log_file(folder, 'processtime_scancontext.txt')
    processtime_fusecolor = read_log_file(folder, 'processtime_fusecolor.txt')
    processtime_fastlio = read_log_file(folder, 'processtime_fastlio.txt')
    pt.mean_fastlio, pt.std_fastlio, pt.fastlio_filt = filter_array_mean_std_dev(processtime_fastlio, 2)
    pt.mean_fusecolor, pt.std_fusecolor, pt.fusecolor_filt = filter_array_mean_std_dev(processtime_fusecolor, 2)
    pt.mean_scancontext, pt.std_scancontext, pt.scancontext_filt = filter_array_mean_std_dev(processtime_scancontext, 2)

    ####################################################################
    ###### Throughput estimations
    ####################################################################
    tp = Struct()
    tp.im_source_fusecolor = msg_size.fusecolor_im/lat.source_fusecolor_im
    tp.cloud_source_fastlio = msg_size.fusecolor_cloudin/lat.source_fastlio_cloud[0:len(msg_size.fusecolor_cloudin)]
    tp.cloud_fastlio_fusecolor = msg_size.fusecolor_cloudin/lat.fastlio_fusecolor_cloud
    tp.cloud_fusecolor_scancontext = msg_size.fusecolor_cloudout/lat.fusecolor_scancontext_cloud

    ####################################################################################################################

    ####################################################################
    ###### Plot CPU data
    ####################################################################
    work_cpu(cpu=cpu, arch=architecture)
    work_ram(ram=ram, arch=architecture)
    lat_mean, lat_std = work_latency(lat=lat, arch=architecture)
