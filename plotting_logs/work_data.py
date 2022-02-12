import numpy as np
import matplotlib.pyplot as plt


def work_cpu(cpu, arch):
    # Define numbers of processors for each entity where the process takes place, depending on the architecture
    n_procs = np.array([12, 12, 12, 12])
    if arch == 0 or arch == 1:
        n_procs = np.array([12, 12, 12, 12])
    elif arch == 2:
        n_procs = np.array([4, 12, 12, 12])
    elif arch == 3:
        n_procs = np.array([4, 4, 12, 12])
    elif arch == 4:
        n_procs = np.array([4, 4, 4, 12])

    # Divide the percentage by the number of processors
    cpu_fastlio = cpu.fastlio_node / n_procs[0]
    cpu_fusecolor = cpu.project_image_pointcloud_node / n_procs[1]
    cpu_scancontext = cpu.scan_context_node / n_procs[2]
    cpu_mesh = cpu.final_mesh_server_node / n_procs[3]

    # Plot the percentages throughout the processing for each node
    plt.plot(np.arange(0, len(cpu_fastlio), 1, dtype=int), cpu_fastlio, 'c', label='Odometry')
    plt.plot(np.arange(0, len(cpu_fusecolor), 1, dtype=int), cpu_fusecolor, 'g', label='Color fusion')
    plt.plot(np.arange(0, len(cpu_scancontext), 1, dtype=int), cpu_scancontext, 'b', label='Loop closure')
    plt.plot(np.arange(0, len(cpu_mesh), 1, dtype=int), cpu_mesh, 'k', label='Mesh calculation')
    plt.title('CPU usage for each node')
    plt.xlabel('Seconds')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.xlim([0, len(cpu_mesh)])
    plt.ylim([0, 100])
    plt.grid()
    plt.show()

    # Plot the total CPU usage
    if 'cpu.total_edge' in locals():
        plt.plot(np.arange(0, len(cpu.total_edge), 1, dtype=int), cpu.total_edge, 'g', label='Edge node')
    plt.plot(np.arange(0, len(cpu.total_fog), 1, dtype=int), cpu.total_fog, 'b', label='Fog node')
    plt.xlim([0, len(cpu.total_fog)])
    plt.ylim([0, 100])
    plt.title('Total CPU usage in each device')
    plt.xlabel('Seconds')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid()
    plt.show()


def work_ram(ram, arch):
    # Define the total RAM for each entity, depending on the architecture
    total_ram = np.array([12, 12, 12, 12]) * 1024 ** 2
    if arch == 0 or arch == 1:
        total_ram = np.array([12, 12, 12, 12])*1024**2
    elif arch == 2:
        total_ram = np.array([2, 12, 12, 12])*1024**2
    elif arch == 3:
        total_ram = np.array([2, 2, 12, 12])*1024**2
    elif arch == 4:
        total_ram = np.array([2, 2, 2, 12])*1024**2

    # Divide the percentage by the total ram of the entity
    ram_fastlio = np.zeros_like(ram.final_mesh_server_node)
    ram_fusecolor = np.zeros_like(ram.final_mesh_server_node)
    ram_scancontext = np.zeros_like(ram.final_mesh_server_node)
    ram_mesh = np.zeros_like(ram.final_mesh_server_node)
    ram_fastlio[0:len(ram.fastlio_node)] = ram.fastlio_node / total_ram[0]
    ram_fusecolor[0:len(ram.project_image_pointcloud_node)] = ram.project_image_pointcloud_node / total_ram[1]
    ram_scancontext[0:len(ram.scan_context_node)] = ram.scan_context_node / total_ram[2]
    ram_mesh[0:len(ram.final_mesh_server_node)] = ram.final_mesh_server_node / total_ram[3]

    # Plot the RAM evolution for each node
    plt.plot(np.arange(0, len(ram_fastlio), 1, dtype=int), ram_fastlio, 'c', label='Odometry')
    plt.plot(np.arange(0, len(ram_fusecolor), 1, dtype=int), ram_fusecolor, 'g', label='Color fusion')
    plt.plot(np.arange(0, len(ram_scancontext), 1, dtype=int), ram_scancontext, 'b', label='Loop closure')
    plt.plot(np.arange(0, len(ram_mesh), 1, dtype=int), ram_mesh, 'k', label='Mesh calculation')
    plt.xlim([0, len(ram_mesh)])
    plt.ylim([0, 100])
    plt.title('RAM usage for each node')
    plt.xlabel('Seconds')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the total RAM evolution in percentage for edge and fog
    ram_edge = np.zeros_like(ram.final_mesh_server_node)
    ram_fog = np.zeros_like(ram.final_mesh_server_node)
    if arch == 0 or arch == 1:
        ram_fog = ram_fastlio + ram_fusecolor + ram_scancontext + ram_mesh
    elif arch == 2:
        ram_edge = ram_fastlio
        ram_fog = ram_fusecolor + ram_scancontext + ram_mesh
    elif arch == 3:
        ram_edge = ram_fastlio + ram_fusecolor
        ram_fog = ram_scancontext + ram_mesh
    elif arch == 4:
        ram_edge = ram_fastlio + ram_fusecolor + ram_scancontext
        ram_fog = ram_mesh
    plt.plot(np.arange(0, len(ram_edge), 1, dtype=int), ram_edge, 'g', label='Edge node')
    plt.plot(np.arange(0, len(ram_fog), 1, dtype=int), ram_fog, 'b', label='Fog node')
    plt.xlim([0, len(ram_fog)])
    plt.ylim([0, 100])
    plt.title('Total RAM usage in each device')
    plt.xlabel('Seconds')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid()
    plt.show()

    return np.max(ram_edge), np.max(ram_fog)


def work_latency(lat, arch):
    # Check which latency really matter depending on the architecture - edge-fog interface
    lat1 = []
    if arch == 0 or arch == 1:
        lat1 = lat.source_fastlio_cloud
    elif arch == 2:
        lat1 = lat.fastlio_fusecolor_cloud
    elif arch == 3:
        lat1 = lat.fusecolor_scancontext_cloud
    elif arch == 4:
        pass
    lat2 = lat.source_fusecolor_im
    # Get the mean and standard deviation for both
    lat1_mean = np.mean(lat1, axis=0)
    lat1_sd = np.std(lat1, axis=0)
    lat2_mean = np.mean(lat2, axis=0)
    lat2_sd = np.std(lat2, axis=0)

    # Return the data for latency
    return lat1_mean + lat2_mean, lat1_sd + lat2_sd
