import numpy as np
import matplotlib.pyplot as plt

fonte_legenda = 10
fonte_axis = 12

def plot_publications(y, tds_, rm_, efa_, fr_, fc_, ylim, name):
    _, ax = plt.subplots()
    ax.plot(y, tds_, '--*', linewidth=3, markersize=9, label='3D Scanning')
    ax.plot(y, rm_, '--*', linewidth=3, markersize=9, label='Remote monitoring')
    ax.plot(y, efa_, '-*', linewidth=3, markersize=9, label='Edge-fog architecture')
    ax.plot(y, fr_, '-*', linewidth=3, markersize=9, label='Fog Robotics')
    ax.plot(y, fc_, '-*', linewidth=3, markersize=9, label='Fog Computing')
    ax.set_xticks(y)
    ax.set_ylim(0, ylim)
    ax.grid()
    ax.legend(fontsize=fonte_legenda, loc='upper right')
    ax.set_xlabel('Anos', fontsize=fonte_axis)
    ax.set_ylabel('Numero de publicacoes (em escala)', fontsize=fonte_axis)
    plt.savefig(name, format='eps')

# Graficos de evolucao de publicacao desde 2017
# 2019 - 20 - 21 - 22 - 23
years = [2019, 2020, 2021, 2022, 2023]

## Science direct
# 3d scanning
tds = [795, 944, 1112, 1304, 455]
# remote monitoring
rm = [1040, 1323, 1899, 2073, 516]
# edge-fog architecture
efa = [81, 141, 132, 220, 54]
# fog robotics
fr = [0, 1, 4, 4, 0]
# fog computing 
fc = [470, 630, 761, 935, 238]
# plot
plot_publications(years, tds, rm, efa, fr, fc, 3100, "science_direct.eps")

## Scopus
# 3d scanning
tds = [610, 785, 976, 1180, 329]
# remote monitoring
rm = [1290, 1730, 2630, 3050, 614]
# edge-fog architecture
efa = [99, 189, 218, 345, 66]
# fog robotics
fr = [1, 1, 3, 7, 1]
# fog computing 
fc = [720, 1250, 1710, 2170, 451]
# plot
plot_publications(years, tds, rm, efa, fr, fc, 4700, "scopus.eps")

plt.show()