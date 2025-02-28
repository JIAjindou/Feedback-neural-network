import numpy as np
import casadi as ca
from scipy.io import loadmat
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams

config = {
    "font.family": "calibri",
}
rcParams.update(config)
from aux_module.quadrotor_visualize import *

vis = Quadrotor_Visualize()
vis.bodyX_alpha = 0.8

from model.model_nominal import Quadrotor
from model.model_NODE import Quadrotor_NODE
from model.model_sim import Quadrotor_Sim
from aux_module.trajectory_tools import *

model_nominal = Quadrotor()
model_learn = Quadrotor_NODE(discrete_h_learn=0.02)
model_sim = Quadrotor_Sim()
model_sim.aero_D = np.diag([0.7, 0.7, 0.2])

""" Plot Training Trajectories"""
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 20})
fig = plt.figure(figsize=(6.5, 6))
ax1 = plt.axes(projection="3d")
ax1.view_init(elev=28, azim=-45)
plt.gca().set_box_aspect((6, 6, 4))
ax1.invert_zaxis()

ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
ax1.set_zlabel("z [m]")
ax1.set_xlim([-4, 4])
ax1.set_ylim([-4, 4])
# ax1.set_zticks([-2.0, -1.5, -1.0, -0.5, 0.0])
# ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax1.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.2))

# Read and plot generated trajectory
folder_dir = "data"
traj_dir_list = [file for file in os.listdir(folder_dir) if file.endswith(".mat")]
for traj_dir in traj_dir_list:
    sim_traj = loadmat(folder_dir + "/" + traj_dir)
    xk_real_seq = sim_traj["xk_real_seq"].T
    refk_seq = sim_traj["aux_inputk_seq"].T
    ax1.plot(
        xk_real_seq[:, 0],
        xk_real_seq[:, 1],
        xk_real_seq[:, 2],
        c="lightcoral",
        linewidth=3,
        alpha=0.8,
    )
    ax1.plot(refk_seq[:, 0], refk_seq[:, 1], refk_seq[:, 2], c="k", linestyle="-.")

legend_elements = [
    Line2D([0], [0], linestyle="-.", color="k", lw=1, label="Reference"),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Fly results",
        markerfacecolor="lightcoral",
        markersize=10,
    ),
]
ax1.legend(handles=legend_elements, framealpha=1)
plt.savefig("img/data_sample.png", dpi=300)

""" Plot Learning Curves for NODE """
# Get filenames
fpath_list = []
for filepath, dirnames, filenames in os.walk("learning_NODE/log"):
    for filename in filenames:
        fpath_list += [filepath + "/" + filename]
# Plot
plt.rcParams.update({"font.size": 21})
fig1 = plt.figure(figsize=(7, 5))
ax1 = fig1.add_subplot(111)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
# ax1.set_yscale("log")
ax1.grid()
for i, fpath in enumerate(fpath_list):
    loss_seq = np.load(fpath)
    ax1.plot(
        loss_seq[:],
        linestyle="--",
        marker="^",
        markersize=5,
        label="Trial {}".format(i + 1),
    )
ax1.legend()
plt.savefig("img/learning_curve_NODE.png", dpi=300, bbox_inches="tight")

""" Plot Learning Curves for MLP """
# Plot
skip = 10
plt.rcParams.update({"font.size": 21})
fig1 = plt.figure(figsize=(7, 5))
ax1 = fig1.add_subplot(111)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.set_yscale("log")
ax1.grid()
loss_seq = np.load(
    "learning_MLP/log/Nov17_16-00-48_2024_minibatch_torchADAM/loss_record.npy"
)
ax1.plot(
    list(range(loss_seq.shape[0]))[::skip],
    loss_seq[::skip, 0],
    linestyle="--",
    marker="^",
    markersize=5,
    label="Train",
)
ax1.plot(
    list(range(loss_seq.shape[0]))[::skip],
    loss_seq[::skip, 1],
    linestyle="--",
    marker="^",
    markersize=5,
    label="Val",
)
ax1.plot(
    list(range(loss_seq.shape[0]))[::skip],
    loss_seq[::skip, 2],
    linestyle="--",
    marker="^",
    markersize=5,
    label="Test",
)
ax1.legend()
plt.savefig("img/learning_curve_MLP.png", dpi=300, bbox_inches="tight")


"""Visualization for Evaluation (NODE plotted)"""
test_trajectories_NODE = []
test_trajectories_MLP = []

# Load test trajectories
traj_dir_list = [file for file in os.listdir("test_eval") if file.endswith("NODE.mat")]
for traj in traj_dir_list:
    traj_data = loadmat("test_eval/" + traj)
    test_trajectories_NODE += [
        {
            "xk_pred_seq": traj_data["xk_pred_seq"].T,
            "xk_real_seq": traj_data["xk_real_seq"].T,
            "refk_seq": traj_data["refk_seq"].T,
            "drag_pred_seq": traj_data["drag_pred_seq"].T,
            "drag_real_seq": traj_data["drag_real_seq"].T,
        }
    ]

# Load test trajectories
traj_dir_list = [file for file in os.listdir("test_eval") if file.endswith("MLP.mat")]
for traj in traj_dir_list:
    traj_data = loadmat("test_eval/" + traj)
    test_trajectories_MLP += [
        {
            "xk_pred_seq": traj_data["xk_pred_seq"].T,
            "xk_real_seq": traj_data["xk_real_seq"].T,
            "refk_seq": traj_data["refk_seq"].T,
            "drag_pred_seq": traj_data["drag_pred_seq"].T,
            "drag_real_seq": traj_data["drag_real_seq"].T,
        }
    ]

# Define Wrap-up function
def addsubplot_traj(xk_pred_seq, xk_real_seq, refk_seq, ax, start, end, stp):
    ax.view_init(elev=28, azim=65)
    # ax.set_box_aspect((6, 6, 1.5))
    ax.invert_zaxis()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    # ax2.set_xlim([-4, 4])
    # ax2.set_ylim([-4, 4])
    # ax.set_zlim([0, -1])
    # ax.set_zticks([-1.0, -0.5, 0.0])
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

    for x in xk_pred_seq[start:end:stp]:
        vis.plot_quadrotorEul(
            ax=ax, x=x[0:12], pc_list=["lightcoral"] * 4, bc="lightcoral"
        )

    for x in xk_real_seq[start:end:stp]:
        vis.plot_quadrotorEul(ax=ax, x=x[0:12], pc_list=["darkcyan"] * 4, bc="darkcyan")


def plot_states(xk_pred_seq, xk_real_seq, fig_, id_):
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
    ax_1.plot(xk_pred_seq[:, 0], color=c1, linestyle=s1, linewidth=lw)
    ax_1.plot(xk_real_seq[:, 0], color=c2, linestyle=s2, linewidth=lw)

    ax_2 = fig_.add_subplot(3, 4, 5)
    # ax2_2.set_xlabel("t")
    ax_2.set_ylabel("Pos y")
    ax_2.grid()
    ax_2.plot(xk_pred_seq[:, 1], color=c1, linestyle=s1, linewidth=lw)
    ax_2.plot(xk_real_seq[:, 1], color=c2, linestyle=s2, linewidth=lw)

    ax_3 = fig_.add_subplot(3, 4, 9)
    # ax2_3.set_xlabel("t")
    ax_3.set_ylabel("Pos z")
    ax_3.grid()
    ax_3.plot(xk_pred_seq[:, 2], color=c1, linestyle=s1, linewidth=lw)
    ax_3.plot(xk_real_seq[:, 2], color=c2, linestyle=s2, linewidth=lw)

    ax_4 = fig_.add_subplot(3, 4, 2)
    # x2_4.set_xlabel("t")
    ax_4.set_ylabel("Vel x")
    ax_4.grid()
    ax_4.plot(xk_pred_seq[:, 3], color=c1, linestyle=s1, linewidth=lw)
    ax_4.plot(xk_real_seq[:, 3], color=c2, linestyle=s2, linewidth=lw)

    ax_5 = fig_.add_subplot(3, 4, 6)
    # ax2_5.set_xlabel("t")
    ax_5.set_ylabel("Vel y")
    ax_5.grid()
    ax_5.plot(xk_pred_seq[:, 4], color=c1, linestyle=s1, linewidth=lw)
    ax_5.plot(xk_real_seq[:, 4], color=c2, linestyle=s2, linewidth=lw)

    ax_6 = fig_.add_subplot(3, 4, 10)
    # ax2_6.set_xlabel("t")
    ax_6.set_ylabel("Vel z")
    ax_6.grid()
    ax_6.plot(xk_pred_seq[:, 5], color=c1, linestyle=s1, linewidth=lw)
    ax_6.plot(xk_real_seq[:, 5], color=c2, linestyle=s2, linewidth=lw)

    ax_7 = fig_.add_subplot(3, 4, 3)
    # ax2_7.set_xlabel("t")
    ax_7.set_ylabel("Pitch")
    ax_7.grid()
    ax_7.plot(xk_pred_seq[:, 6], color=c1, linestyle=s1, linewidth=lw)
    ax_7.plot(xk_real_seq[:, 6], color=c2, linestyle=s2, linewidth=lw)

    ax_8 = fig_.add_subplot(3, 4, 7)
    # ax2_8.set_xlabel("t")
    ax_8.set_ylabel("Pitch")
    ax_8.grid()
    ax_8.plot(xk_pred_seq[:, 7], color=c1, linestyle=s1, linewidth=lw)
    ax_8.plot(xk_real_seq[:, 7], color=c2, linestyle=s2, linewidth=lw)

    ax_9 = fig_.add_subplot(3, 4, 11)
    # ax2_9.set_xlabel("t")
    ax_9.set_ylabel("Yaw")
    ax_9.grid()
    ax_9.plot(xk_pred_seq[:, 8], color=c1, linestyle=s1, linewidth=lw)
    ax_9.plot(xk_real_seq[:, 8], color=c2, linestyle=s2, linewidth=lw)

    ax_10 = fig_.add_subplot(3, 4, 4)
    # ax2_10.set_xlabel("t")
    ax_10.set_ylabel("Omega x")
    ax_10.grid()
    ax_10.plot(xk_pred_seq[:, 9], color=c1, linestyle=s1, linewidth=lw)
    ax_10.plot(xk_real_seq[:, 9], color=c2, linestyle=s2, linewidth=lw)

    ax_11 = fig_.add_subplot(3, 4, 8)
    # ax2_8.set_xlabel("t")
    ax_11.set_ylabel("Omega y")
    ax_11.grid()
    ax_11.plot(xk_pred_seq[:, 10], color=c1, linestyle=s1, linewidth=lw)
    ax_11.plot(xk_real_seq[:, 10], color=c2, linestyle=s2, linewidth=lw)

    ax_12 = fig_.add_subplot(3, 4, 12)
    # ax2_9.set_xlabel("t")
    ax_12.set_ylabel("Omega z")
    ax_12.grid()
    ax_12.plot(xk_pred_seq[:, 11], color=c1, linestyle=s1, linewidth=lw)
    ax_12.plot(xk_real_seq[:, 11], color=c2, linestyle=s2, linewidth=lw)

    legend_elements = [
        Line2D([0], [0], linestyle=s1, color=c1, lw=2, label="Prediction"),
        Line2D([0], [0], linestyle=s2, color=c2, lw=2, label="Truth"),
    ]

    fig_.legend(handles=legend_elements, loc="lower center", ncol=3, framealpha=1)
    fig_.savefig("img/val_state_plot_{}.png".format(id_), dpi=300)


# Select 3 NODE trajectories to plot
plt.rcParams.update({"font.size": 15})
fig3 = plt.figure(figsize=(10, 5))
fig3.subplots_adjust(
    left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.02
)
ax21 = fig3.add_subplot(131, projection="3d")
ax22 = fig3.add_subplot(132, projection="3d")
ax23 = fig3.add_subplot(133, projection="3d")
ax_list = [ax21, ax22, ax23]

for traj, ax in zip(test_trajectories_NODE, ax_list):
    xk_pred_seq = traj["xk_pred_seq"]
    xk_real_seq = traj["xk_real_seq"]
    refk_seq = traj["refk_seq"]
    addsubplot_traj(xk_pred_seq, xk_real_seq, refk_seq, ax, start=0, end=300, stp=4)

legend_elements = [
    Line2D([0], [0], linestyle="-.", color="k", lw=1, label="Reference"),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Prediction",
        markerfacecolor="lightcoral",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Truth",
        markerfacecolor="darkcyan",
        markersize=10,
    ),
]
fig3.legend(handles=legend_elements, loc="lower center", ncol=3, framealpha=1)
fig3.savefig("img/test_traj.png", dpi=300)

"""Plot State Trajectories"""
# Select 3 trajectories to plot
plt.rcParams.update({"font.size": 15})
fig4 = plt.figure(figsize=(12, 8))
fig5 = plt.figure(figsize=(12, 8))
fig6 = plt.figure(figsize=(12, 8))
fig_list = [fig4, fig5, fig6]
id_ = 1
for traj, fig_ in zip(test_trajectories_NODE, fig_list):
    xk_pred_seq = traj["xk_pred_seq"]
    xk_real_seq = traj["xk_real_seq"]
    refk_seq = traj["refk_seq"]
    plot_states(xk_pred_seq[::4], xk_real_seq[::4], fig_, id_)
    id_ += 1