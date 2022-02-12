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
    tp.im_source_fusecolor[tp.im_source_fusecolor > 1e308] = 0
    tp.cloud_source_fastlio = msg_size.fusecolor_cloudin/lat.source_fastlio_cloud[0:len(msg_size.fusecolor_cloudin)]
    tp.cloud_source_fastlio[tp.cloud_source_fastlio > 1e308] = 0
    tp.cloud_fastlio_fusecolor = msg_size.fusecolor_cloudin/lat.fastlio_fusecolor_cloud
    tp.cloud_fastlio_fusecolor[tp.cloud_fastlio_fusecolor > 1e308] = 0
    tp.cloud_fusecolor_scancontext = msg_size.fusecolor_cloudout/lat.fusecolor_scancontext_cloud
    tp.cloud_fusecolor_scancontext[tp.cloud_fusecolor_scancontext > 1e308] = 0

    ####################################################################################################################

    ####################################################################
    ###### Time for a point cloud to go from source to SC
    ####################################################################
    total_t_process = np.max(pt.fastlio_filt) + np.max(pt.fusecolor_filt) + np.max(pt.scancontext_filt)
    total_t_latency = np.max(lat.source_fastlio_cloud) + np.max(lat.fastlio_fusecolor_cloud) \
                      + np.max(lat.fusecolor_scancontext_cloud)
    time_full_cloud_process = total_t_process + total_t_latency

    ####################################################################
    ###### Max. CPU in each device, plus plots
    ####################################################################
    work_cpu(cpu=cpu, arch=architecture)
    max_cpu_edge = np.max(cpu.total_edge) if 'cpu.total_edge' in locals() else 0
    max_cpu_fog = np.max(cpu.total_fog)

    ####################################################################
    ###### Max. RAM in each device, plus plots
    ####################################################################
    max_ram_edge, max_ram_fog = work_ram(ram=ram, arch=architecture)

    ####################################################################
    ###### Max. Throughtput requested by this architecture
    ####################################################################
    tp_max_requested = np.max(tp.im_source_fusecolor)
    if architecture == 0 or architecture == 1:
        tp_max_requested += np.max(tp.cloud_source_fastlio)
    elif architecture == 2:
        tp_max_requested += np.max(tp.cloud_fastlio_fusecolor)
    elif architecture == 3:
        tp_max_requested += np.max(tp.cloud_fusecolor_scancontext)

    ####################################################################
    ###### Analysing latency, plus plots
    ####################################################################
    lat_mean, lat_std = work_latency(lat=lat, arch=architecture)

    ####################################################################
    ###### Saving interesting data in output table
    ####################################################################
    output_data = {'Max CPU Edge': max_cpu_edge, 'Max CPU Fog': max_cpu_fog,
                   'Max RAM Edge': max_ram_edge, 'Max RAM Fog': max_ram_fog,
                   'Mean Latency': lat_mean,
                   'Max throughput requested': tp_max_requested,
                   'Time for cloud process': time_full_cloud_process}
    data_table = pd.DataFrame.from_dict(output_data, orient='index', columns=['Value'])
    print(data_table)
    data_table.to_csv(os.path.join(folder, 'data_table.csv'), header=False, float_format='%.2f')
