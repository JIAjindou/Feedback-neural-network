import numpy as np
import casadi as ca
from scipy.io import savemat, loadmat
import os

from model.model_NODE import Quadrotor_NODE
from model.model_sim import Quadrotor_Sim
from aux_module.trajectory_tools import *

dt = 0.02
model_learn = Quadrotor_NODE(discrete_h_learn=dt)
params_NODE = ca.DM(
    np.load("learning_NODE/temp/model_param.npy")
)  # DM is used for end2end forward rollout
model_sim = Quadrotor_Sim()
model_sim.aero_D = np.diag([0.7, 0.7, 0.2])


def rollout_and_pred(refk_seq, x0, params):
    xk_real = x0
    xk_pred = x0
    xk_real_seq = [xk_real]
    xk_pred_seq = [xk_pred]

    for i in range(refk_seq.shape[0]):
        xk1_real = model_sim.cldyn_sym_exRK4(xk_real, refk_seq[i], dt)
        xk1_pred = model_learn.augdyn_symRK4(xk_real, refk_seq[i], params)
        xk_real_seq.append(xk1_real.full())
        xk_pred_seq.append(xk1_pred.full())

        xk_real = xk1_real
        xk_pred = xk1_pred

    xk_real_seq = np.array(xk_real_seq).reshape(-1, x0.shape[0])
    xk_pred_seq = np.array(xk_pred_seq).reshape(-1, x0.shape[0])

    return (xk_real_seq, xk_pred_seq)


""" Evaluate models on training trajectories"""
dir_ = "data"
traj_dir_list = [file for file in os.listdir(dir_) if file.endswith(".mat")]

for i, traj_dir in enumerate(traj_dir_list):
    traj_i = i + 1
    print("train set eval process: {0}/{1}".format(traj_i, len(traj_dir_list)))
    learndata = loadmat("data" + "/" + traj_dir)
    xk_real_seq = learndata["xk_real_seq"].T
    refk_seq = learndata["aux_inputk_seq"].T

    x0 = ca.DM(xk_real_seq[0, :])

    (xk_real_seq, xk_pred_seq) = rollout_and_pred(refk_seq, x0, params=params_NODE)

    savemat(
        "train_eval/train_trajectory{}_NODE.mat".format(traj_i),
        {
            "xk_real_seq": xk_real_seq.T,
            "xk_pred_seq": xk_pred_seq.T,
            "refk_seq": refk_seq.T,
        },
    )


""" Evaluate models on test trajectories """
from aux_module.trajectory_tools import *
from model.model_nominal import Quadrotor

model_nominal = Quadrotor()
solver = Polynomial_TrajOpt(model=model_nominal)

# Trajectory Generation
traj_num = 10
dt = 0.02
N = 500
traj_i = 1

while traj_i <= traj_num:
    print("test set eval process: {0}/{1}".format(traj_i, traj_num))
    pos_waypoints = get_random_waypoints(
        waypoint_num=5, x_bound=[-3, 3], y_bound=[-3, 3], z_bound=[-2, 2]
    )
    # Generate random command trajectories
    t_init = 0.0
    refx0 = np.hstack([pos_waypoints[0, :], np.array([0] * 9)])
    refxf = np.hstack([pos_waypoints[-1, :], np.array([0] * 9)])

    solver.discrete_setup(N=N, h=0.02)
    solver.set_refuBoxCons(inputlb=[-1e2] * 3, inputub=[1e2] * 3)
    solver.set_posWP(pos_waypoints)
    solver.setrefxBoundCond(refx0, refxf)
    solver.NLP_Prepare()
    sol = solver.NLP_FormAndSolve(Eq_Relax=0.0)
    refx_opt = solver.get_refxopt(sol)
    refx = ca.vertcat(
        refx0.reshape((-1, refx0.shape[0])),
        refx_opt[:-1],
    )

    # Closed-loop Rollout for state trajectories
    refk_seq = refx.toarray()

    # Closed-loop Rollout for state trajectories
    x0 = ca.DM(np.hstack([pos_waypoints[0, :], [0] * 15]))

    # xk_real_seq = xk_real_seq[:, 0:12] # Select only nominal states
    # Check if accept trajectory
    vel_ref = np.linalg.norm(refk_seq[:, 3:6], axis=1)
    if max(vel_ref) <= 10:
        (xk_real_seq, xk_pred_seq) = rollout_and_pred(refk_seq, x0, params=params_NODE)

        savemat(
            "test_eval/test_trajectory{}_NODE.mat".format(traj_i),
            {
                "xk_real_seq": xk_real_seq.T,
                "xk_pred_seq": xk_pred_seq.T,
                "refk_seq": refk_seq.T,
            },
        )

        print(traj_i, "trajectories tested.")
        traj_i += 1
