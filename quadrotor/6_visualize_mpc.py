import numpy as np
import os
from scipy.io import loadmat, savemat
from sklearn.metrics import root_mean_squared_error

"""Visualization"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm, colors
from matplotlib import rcParams
config = {
    "font.family":'calibri',
    "font.size": 15,
}
rcParams.update(config)

from aux_module.quadrotor_visualize import Quadrotor_Visualize

vis = Quadrotor_Visualize()
vis.bodyX_alpha = 0.8

# Define Wrap-up function
def addsubplot_traj(xk_seq, refk_seq, lossk_seq, ax, title, start, end, stp):

    ax.set_title(title, pad=-5)
    ax.view_init(elev=28, azim=65)
    ax.set_box_aspect((6, 6, 2.5))
    ax.invert_zaxis()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    # ax2.set_xlim([-4, 4])
    # ax2.set_ylim([-4, 4])
    # ax.set_zlim([0, -1])
    ax.set_zticks([-1.0, -0.5, 0.0])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.plot(
        refk_seq[start:end, 0],
        refk_seq[start:end, 1],
        refk_seq[start:end, 2],
        "k",
        linestyle="-.",
    )
    poserr_seq = xk_seq[:-1, 0:3] - refk_seq

    errnorm_list = [np.linalg.norm(poserr) for poserr in poserr_seq]
    print("Max tracking err:", max(errnorm_list))
    print("RMSE:", root_mean_squared_error(xk_seq[:-1, 0:3], refk_seq))

    for lossk, x in zip(errnorm_list[start:end:stp], xk_seq[start:end:stp]):
        v = (lossk - 0) / (0.72 - 0)
        color = tuple(cm.viridis(v))
        vis.plot_quadrotorEul(ax=ax, x=x[0:12], pc_list=[color]*4, bc='k')


def plot_dx(dx_pred, dx_real, fig_, title):
    dx_predT = dx_pred.T
    dx_realT = dx_real.T

    c1 = "lightcoral"
    c2 = "darkcyan"
    s1 = "solid"
    s2 = "dotted"
    lw = 3

    fig_.subplots_adjust(0.1, 0.12, 0.95, 0.9, 0.5, 0.3)
    ax_1 = fig_.add_subplot(3, 4, 1)
    # x2_1.set_xlabel("t")
    ax_1.set_ylabel("Pos x")
    ax_1.grid()
    ax_1.plot(dx_predT[:, 0], color=c1, linestyle=s1, linewidth=lw)
    ax_1.plot(dx_realT[:, 0], color=c2, linestyle=s2, linewidth=lw)

    ax_2 = fig_.add_subplot(3, 4, 5)
    # ax2_2.set_xlabel("t")
    ax_2.set_ylabel("Pos y")
    ax_2.grid()
    ax_2.plot(dx_predT[:, 1], color=c1, linestyle=s1, linewidth=lw)
    ax_2.plot(dx_realT[:, 1], color=c2, linestyle=s2, linewidth=lw)

    ax_3 = fig_.add_subplot(3, 4, 9)
    # ax2_3.set_xlabel("t")
    ax_3.set_ylabel("Pos z")
    ax_3.grid()
    ax_3.plot(dx_predT[:, 2], color=c1, linestyle=s1, linewidth=lw)
    ax_3.plot(dx_realT[:, 2], color=c2, linestyle=s2, linewidth=lw)

    ax_4 = fig_.add_subplot(3, 4, 2)
    # x2_4.set_xlabel("t")
    ax_4.set_ylabel("Vel x")
    ax_4.grid()
    ax_4.plot(dx_predT[:, 3], color=c1, linestyle=s1, linewidth=lw)
    ax_4.plot(dx_realT[:, 3], color=c2, linestyle=s2, linewidth=lw)

    ax_5 = fig_.add_subplot(3, 4, 6)
    # ax2_5.set_xlabel("t")
    ax_5.set_ylabel("Vel y")
    ax_5.grid()
    ax_5.plot(dx_predT[:, 4], color=c1, linestyle=s1, linewidth=lw)
    ax_5.plot(dx_realT[:, 4], color=c2, linestyle=s2, linewidth=lw)

    ax_6 = fig_.add_subplot(3, 4, 10)
    # ax2_6.set_xlabel("t")
    ax_6.set_ylabel("Vel z")
    ax_6.grid()
    ax_6.plot(dx_predT[:, 5], color=c1, linestyle=s1, linewidth=lw)
    ax_6.plot(dx_realT[:, 5], color=c2, linestyle=s2, linewidth=lw)

    ax_7 = fig_.add_subplot(3, 4, 3)
    # ax2_7.set_xlabel("t")
    ax_7.set_ylabel("Pitch")
    ax_7.grid()
    ax_7.plot(dx_predT[:, 6], color=c1, linestyle=s1, linewidth=lw)
    ax_7.plot(dx_realT[:, 6], color=c2, linestyle=s2, linewidth=lw)

    ax_8 = fig_.add_subplot(3, 4, 7)
    # ax2_8.set_xlabel("t")
    ax_8.set_ylabel("Pitch")
    ax_8.grid()
    ax_8.plot(dx_predT[:, 7], color=c1, linestyle=s1, linewidth=lw)
    ax_8.plot(dx_realT[:, 7], color=c2, linestyle=s2, linewidth=lw)

    ax_9 = fig_.add_subplot(3, 4, 11)
    # ax2_9.set_xlabel("t")
    ax_9.set_ylabel("Yaw")
    ax_9.grid()
    ax_9.plot(dx_predT[:, 8], color=c1, linestyle=s1, linewidth=lw)
    ax_9.plot(dx_realT[:, 8], color=c2, linestyle=s2, linewidth=lw)

    ax_10 = fig_.add_subplot(3, 4, 4)
    # ax2_10.set_xlabel("t")
    ax_10.set_ylabel("Omega x")
    ax_10.grid()
    ax_10.plot(dx_predT[:, 9], color=c1, linestyle=s1, linewidth=lw)
    ax_10.plot(dx_realT[:, 9], color=c2, linestyle=s2, linewidth=lw)

    ax_11 = fig_.add_subplot(3, 4, 8)
    # ax2_8.set_xlabel("t")
    ax_11.set_ylabel("Omega y")
    ax_11.grid()
    ax_11.plot(dx_predT[:, 10], color=c1, linestyle=s1, linewidth=lw)
    ax_11.plot(dx_realT[:, 10], color=c2, linestyle=s2, linewidth=lw)

    ax_12 = fig_.add_subplot(3, 4, 12)
    # ax2_9.set_xlabel("t")
    ax_12.set_ylabel("Omega z")
    ax_12.grid()
    ax_12.plot(dx_predT[:, 11], color=c1, linestyle=s1, linewidth=lw)
    ax_12.plot(dx_realT[:, 11], color=c2, linestyle=s2, linewidth=lw)

    legend_elements = [
        Line2D([0], [0], linestyle=s1, color=c1, lw=2, label="Prediction"),
        Line2D([0], [0], linestyle=s2, color=c2, lw=2, label="Truth"),
    ]

    fig_.legend(handles=legend_elements, loc="lower center", ncol=3, framealpha=1)

    fig_.savefig("img/sim_dx_plot_{}.png".format(title), dpi=300)


"""Plot 3D Trajectories"""
fig1 = plt.figure(figsize=(15, 6))
fig1.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.2)
ax11 = fig1.add_subplot(241, projection="3d")
ax12 = fig1.add_subplot(242, projection="3d")
ax13 = fig1.add_subplot(243, projection="3d")
ax14 = fig1.add_subplot(244, projection="3d")
ax15 = fig1.add_subplot(245, projection="3d")
ax16 = fig1.add_subplot(246, projection="3d")
ax17 = fig1.add_subplot(247, projection="3d")
ax18 = fig1.add_subplot(248, projection="3d")
ax_list = [ax11, ax12, ax13, ax14, ax15, ax16, ax17]

# Get sim trajectories
traj_dir_list = [
    file for file in os.listdir("sim_mpc_trajs") if file.endswith(".mat")
]
title_dir_list = [
    "Nomi-MPC",
    "Neural-MPC",
    "FB-MPC",
    "FNN-MPC",
    "AdapNN-MPC",
    "MLP-MPC",
    "AdapFNN-MPC"
]
plot_ids = [1,2,4,6,5,3,7]
data_title_plot = list(zip(traj_dir_list, title_dir_list, plot_ids))
data_title_plot = sorted(data_title_plot, key= lambda x: x[2])

for bind, ax in zip(data_title_plot, ax_list):
    traj_dir, title, plot_id = bind
    simdata = loadmat("sim_mpc_trajs" + "/" + traj_dir)
    xsim_seq = simdata["xsim_seq"]
    pos_refernece = simdata["pos_reference"]
    lossk_seq = simdata["lossk_seq"]

    addsubplot_traj(xsim_seq.T, pos_refernece.T, lossk_seq, ax, title, start=0, end=300, stp=2)

legend_elements = [
    Line2D([0], [0], linestyle="-.", color="k", lw=1, label="Reference"),
    # Line2D(
    #     [0],
    #     [0],
    #     marker="o",
    #     color="w",
    #     label="MPC tracking control",
    #     markerfacecolor="lightcoral",
    #     markersize=10,
    # ),
]

norm_V = colors.Normalize(0, 0.72)
# ax11.legend(handles=legend_elements, loc='best', framealpha=1)
fig1.colorbar(cm.ScalarMappable(norm=norm_V, cmap=cm.viridis),
                cax=plt.axes([0.92, 0.3, 0.01, 0.4]), orientation='vertical', 
                label='Tracking error [m]', fraction = 0.15)

fig1.savefig("img/simulation_result.png", dpi=300)

"""Plot dx Trajectories"""
fig2 = plt.figure(figsize=(12, 8))
fig3 = plt.figure(figsize=(12, 8))
fig4 = plt.figure(figsize=(12, 8))
fig5 = plt.figure(figsize=(12, 8))
fig6 = plt.figure(figsize=(12, 8))
fig7 = plt.figure(figsize=(12, 8))
fig8 = plt.figure(figsize=(12, 8))
fig_list = [fig2, fig3, fig4, fig5, fig6, fig7, fig8]

for bind, fig_ in zip(data_title_plot, fig_list):
    traj_dir, title, ax_id = bind
    print(traj_dir, title)
    simdata = loadmat("sim_mpc_trajs" + "/" + traj_dir)
    dx_real  = simdata["dx_real"]
    dx_pred = simdata["dx_pred"]
    plot_dx(dx_pred[::], dx_real[::], fig_, title)


# Comparison AdaptFNN vs FNN
fig9 = plt.figure(figsize=(10, 4))
fig9.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.0)
ax91 = fig9.add_subplot(121, projection="3d")
ax92 = fig9.add_subplot(122, projection="3d")
ax_list = [ax91, ax92]
traj_dir_list = [
    file for file in os.listdir("sim_mpc_trajs") if file.endswith(".mat")
]
title_dir_list = [
    "Nomi-MPC",
    "Neural-MPC",
    "FB-MPC",
    "FNN-MPC",
    "AdapNN-MPC",
    "MLP-MPC",
    "AdapFNN-MPC"
]
plot_ids = [6,7,4,1,5,3,2]      # Pick 6 and 7 only
data_title_plot = list(zip(traj_dir_list, title_dir_list, plot_ids))
data_title_plot = sorted(data_title_plot, key= lambda x: x[2])

for bind, ax in list(zip(data_title_plot, ax_list))[:2]:
    traj_dir, title, plot_id = bind
    simdata = loadmat("sim_mpc_trajs" + "/" + traj_dir)
    xsim_seq = simdata["xsim_seq"]
    pos_refernece = simdata["pos_reference"]
    lossk_seq = simdata["lossk_seq"]

    addsubplot_traj(xsim_seq.T, pos_refernece.T, lossk_seq, ax, title, start=0, end=300, stp=2)

norm_V = colors.Normalize(0, 0.36)
# ax11.legend(handles=legend_elements, loc='best', framealpha=1)
fig9.colorbar(cm.ScalarMappable(norm=norm_V, cmap=cm.viridis),
                cax=plt.axes([0.92, 0.3, 0.01, 0.4]), orientation='vertical', 
                label='Tracking error [m]', fraction = 0.15)
fig9.savefig("img/simulation_FNN_AdapFNN.png", dpi=300)
