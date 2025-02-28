import time
import numpy as np

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

# Feedback neural networks - Ablation study on feedback gain and uncertainty degree

plt.rcParams['font.family'] = 'Calibri'
color_gray = (102 / 255, 102 / 255, 102 / 255)
color_blue = (76 / 255, 147 / 255, 173 / 255)
color_red = (1, 0, 0)

Error_ablation_FNN = np.array([[9.76916981,  5.27131033,  2.69700933, 10.86641598,  5.73902655,  7.41164732, 9.38551903, 8.05338001, 6.5255065,  5.44023752],
                               [2.7662549,  1.07926428, 0.55296433, 3.28547859, 1.88726377, 1.83750904, 2.24268603, 2.08255315, 1.64348292, 1.33384299],
                               [1.16910183, 0.57652259, 0.31804511, 1.02902365, 0.53571916, 0.57118499, 0.75071597, 0.69858891, 0.53485662, 0.50724995],
                               [0.72154504, 0.28560072, 0.12592565, 0.41471729, 0.26492083, 0.31847236, 0.46942028, 0.49508619, 0.50114757, 0.50270766],
                               [0.65060008, 0.21024537, 0.0719158,  0.17412066, 0.13501026, 0.21583609, 0.34573448, 0.41433105, 0.4625766,  0.49241999],
                               [0.7911638, 0.23101455, 0.05886561, 0.07595783, 0.117239,   0.22487572, 0.38386497, 0.47609606, 0.54510874, 0.59264684],
                               [1.02941728, 0.29626641, 0.06776927, 0.05103273, 0.13278879, 0.28108889, 0.49668434, 0.62131011, 0.71260625, 0.77577323],
                               [1.50964713, 0.4285951,  0.09598425, 0.03724421, 0.17796311, 0.39283311, 0.72465175, 0.89617395, 1.0235498,  1.09978855],
                               [2.9186089,  0.81106019, 0.17990695, 0.02905083, 0.30909851, 0.6566897, 1.16246915, 1.37887096, 1.50722861, 1.53738093],
                               # [7.93075705, 1.81639731, 0.3522152,  0.06622376, 0.48661035, 0.9189468, 1.3839016,  1.51838124, 1.52760446, 1.45669639],
                               [41.560123, 17.408058, 6.638754, 0.053141, 4.458117, 7.502489, 9.86453915, 11.73958111, 13.04064369, 14.3398838]])

Error_ablation_FNN_sub = np.array([
                               [0.57652259, 0.31804511, 1.02902365, 0.53571916, 0.57118499, 0.75071597, 0.69858891],
                               [0.28560072, 0.12592565, 0.41471729, 0.26492083, 0.31847236, 0.46942028, 0.49508619],
                               [0.21024537, 0.0719158,  0.17412066, 0.13501026, 0.21583609, 0.34573448, 0.41433105],
                               [0.23101455, 0.05886561, 0.07595783, 0.117239,   0.22487572, 0.38386497, 0.47609606],
                               [0.29626641, 0.06776927, 0.05103273, 0.13278879, 0.28108889, 0.49668434, 0.62131011],
                               [0.4285951,  0.09598425, 0.03724421, 0.17796311, 0.39283311, 0.72465175, 0.89617395],
                               [0.81106019, 0.17990695, 0.02905083, 0.30909851, 0.6566897, 1.16246915, 1.37887096],
                               ])
plot_commamd = 0  # 1-Error_ablation_FNN; 0-Error_ablation_FNN_sub

if plot_commamd:
    # Heatplot - Error_ablation_FNN：
    data = pd.DataFrame(Error_ablation_FNN)
    # tick_ = np.arange(0, 3, 0.5)
    # dict_ = {"ticks": tick_}
    plot = sns.heatmap(data, cmap="YlGnBu", linewidths=0.8)  # viridis, YlGnBu, Blues, Greens, plasma

    cbar = plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=17, labelcolor="black")
    cbar.ax.set_ylabel(ylabel="Prediction error", size=20, loc="center", labelpad=10)
    # cbar.ax.set_ylim(0, 8)

    plt.xlabel("Degree of uncertainty", size=20, labelpad=20)
    plt.ylabel("Feedback gain", size=20, rotation=90, labelpad=10)
    # plt.title("Ablation study on feedback gain and uncertainty degree", size=20)
    plot.set_xticks([])
    plot.set_yticks(np.arange(10) + 0.5, ['45', '40', '35', '30', '25', '20', '15', '10', '5', '0'])
    plot.tick_params(axis='x', labelsize=17)
    plot.tick_params(axis='y', labelsize=17)

    timestamp = time.time()
    now = time.localtime(timestamp)
    month = now.tm_mon
    day = now.tm_mday

    # Figure show
    plt.savefig('png/b_ablation_L_heatmap_full{:02d}{:02d}'.format(month, day))
    plt.show()

else:
    # Heatplot - Error_ablation_FNN_sub：
    data = pd.DataFrame(Error_ablation_FNN_sub)
    plot = sns.heatmap(data, cmap="YlGnBu", linewidths=0.8)  # viridis, YlGnBu, Blues, Greens, plasma

    cbar = plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=17, labelcolor="black")
    cbar.ax.set_ylabel(ylabel="Prediction error", size=20, loc="center", labelpad=10)
    # cbar.ax.set_ylim(0, 8)

    plt.xlabel("Degree of uncertainty", size=20, labelpad=20)
    plt.ylabel("Feedback gain", size=20, rotation=90, labelpad=10)
    # plt.title("Ablation study on feedback gain and uncertainty degree", size=20)
    plot.set_xticks([])
    plot.set_yticks(np.arange(7) + 0.5, ['35', '30', '25', '20', '15', '10', '5'])
    plot.tick_params(axis='x', labelsize=17)
    plot.tick_params(axis='y', labelsize=17)

    timestamp = time.time()
    now = time.localtime(timestamp)
    month = now.tm_mon
    day = now.tm_mday

    # Figure show
    plt.savefig('png/b_ablation_L_heatmap_sub{:02d}{:02d}'.format(month, day))
    plt.show()


    # python b_ablation_L_heatmap.py --viz





