import numpy as np
import matplotlib.pyplot as plt

fonte_legenda = 10
fonte_axis = 12
################################### total times ################################### 
# Read in the numbers from the file
with open("raw_time_total.txt", "r") as f:
    raw_times = np.array([float(line)*1000 for line in f if line != '\n'])
with open("it_time_total.txt", "r") as f:
    it_times = np.array([float(line)*1000 for line in f if line != '\n'])
with open("cent_time_total.txt", "r") as f:
    cent_times = np.array([float(line)*1000 for line in f if line != '\n'])

# Calculate the moving average with a window size of 5
window_size = 5
raw_times_averages = np.convolve(raw_times, np.ones(window_size)/window_size, mode='valid')
it_times_averages = np.convolve(it_times, np.ones(window_size)/window_size, mode='valid')
cent_times_averages = np.convolve(cent_times, np.ones(window_size)/window_size, mode='valid')

# Plot the original and moving average arrays
plt.plot(raw_times, '--m', label='Serial raw', linewidth=1)
plt.plot(raw_times_averages, 'm', label='Serial media movel', linewidth=2)
plt.plot(it_times, '--b', label='Octree Iterativa raw', linewidth=1)
plt.plot(it_times_averages, 'b', label='Octree Iterativa media movel', linewidth=2)
plt.plot(cent_times, '--c', label='Octree Centroide raw', linewidth=1)
plt.plot(cent_times_averages, 'c', label='Octree Centroide media movel', linewidth=2)
plt.grid()
plt.legend(fontsize=fonte_legenda, loc='upper right')
plt.xlabel('# Scan', fontsize=fonte_axis)
plt.ylabel('Tempo [ms]', fontsize=fonte_axis)
plt.xlim(-20, 300)
plt.ylim(0, 120)
plt.savefig('tempos_totais_octree.eps', format='eps')
# plt.show()

################################### alg times ###################################
# Read in the numbers from the file
with open("it_time_alg.txt", "r") as f:
    it_times_alg = np.array([float(line) for line in f if line != '\n'])
with open("cent_time_alg.txt", "r") as f:
    cent_times_alg = np.array([float(line) for line in f if line != '\n'])
    cent_times_alg[cent_times_alg > 0.5] = 0.5

# Calculate the moving average with a window size of 5
window_size = 5
it_times_alg_averages = np.convolve(it_times_alg, np.ones(window_size)/window_size, mode='valid')
cent_times_alg_averages = np.convolve(cent_times_alg, np.ones(window_size)/window_size, mode='valid')

# Plot the original and moving average arrays
plt.figure()
plt.plot(it_times_alg, '--b', label='Octree Iterativa raw', linewidth=1)
plt.plot(it_times_alg_averages, 'b', label='Octree Iterativa media movel', linewidth=2)
plt.plot(cent_times_alg, '--c', label='Octree Centroide raw', linewidth=1)
plt.plot(cent_times_alg_averages, 'c', label='Octree Centroide media movel', linewidth=2)
plt.grid()
plt.legend(fontsize=fonte_legenda, loc='upper right')
plt.xlabel('# Scan', fontsize=fonte_axis)
plt.ylabel('Tempo [ms]', fontsize=fonte_axis)
plt.xlim(-20, 300)
plt.ylim(0, 0.6)
plt.savefig('tempos_algoritmos_octree.eps', format='eps')
# plt.show()

################################### iterations ###################################
# Read in the numbers from the file
with open("iterations_it.txt", "r") as f:
    it_iterations = np.array([float(line) for line in f if line != '\n'])
with open("iterations_cent.txt", "r") as f:
    cent_iterations = np.array([float(line) for line in f if line != '\n'])

# Calculate the moving average with a window size of 5
window_size = 5
it_iterations_averages = np.convolve(it_iterations, np.ones(window_size)/window_size, mode='valid')
cent_iterations_averages = np.convolve(cent_iterations, np.ones(window_size)/window_size, mode='valid')

# Plot the original and moving average arrays
plt.figure()
plt.plot(it_iterations, '--b', label='Octree Iterativa raw', linewidth=1)
plt.plot(it_iterations_averages, 'b', label='Octree Iterativa media movel', linewidth=2)
plt.plot(cent_iterations, '--c', label='Octree Centroide raw', linewidth=1)
plt.plot(cent_iterations_averages, 'c', label='Octree Centroide media movel', linewidth=2)
plt.grid()
plt.legend(fontsize=fonte_legenda, loc='upper right')
plt.xlabel('# Scan', fontsize=fonte_axis)
plt.ylabel('Iteracoes', fontsize=fonte_axis)
plt.xlim(-20, 300)
plt.ylim(0, 23)
plt.savefig('iteracoes_octree.eps', format='eps')
plt.show()