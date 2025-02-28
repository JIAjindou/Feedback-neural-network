import numpy as np
import casadi as ca
from scipy.io import savemat, loadmat
import os

"""Load Model"""
from model.model_sim import Quadrotor_Sim

# Define aerodynamic drag coefficients
model_sim = Quadrotor_Sim()
model_sim.aero_D = np.diag([0.7, 0.7, 0.2])

"""Make Trajectory Data"""
from aux_module.trajectory_tools import *
from model.model_nominal import Quadrotor

model_nominal = Quadrotor()
solver = Polynomial_TrajOpt(model=model_nominal)

# Trajectory Generation
traj_num = 40
dt = 0.02
N = 200
traj_i = 1

while traj_i <= traj_num:
    pos_waypoints = get_random_waypoints(
        waypoint_num=3, x_bound=[-3, 3], y_bound=[-3, 3], z_bound=[-2, 2]
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
    refk_seq = refx.toarray()

    # Closed-loop Rollout for state trajectories
    x0 = ca.DM(np.hstack([pos_waypoints[0, :], [0] * 15]))   # the closed-loop state is 18-dimensional
    xk_real = x0
    xk_real_seq = [xk_real]
    drag_seq = [model_sim.aerodrag_sym(xk_real).full()]
    for i in range(refk_seq.shape[0]):
        xk1_real = model_sim.cldyn_sym_exRK4(xk_real, refk_seq[i], dt)
        drag = model_sim.aerodrag_sym(xk1_real)
        xk_real_seq.append(xk1_real.full())
        drag_seq.append(drag.full())
        xk_real = xk1_real

    xk_real_seq = np.array(xk_real_seq).reshape(-1, x0.shape[0])
    drag_seq = np.array(drag_seq).reshape(-1, drag.shape[0])

    # Check if accept trajectory
    vel_ref = np.linalg.norm(xk_real_seq[:, 3:6], axis=1)
    if max(vel_ref) <= 10:
        # drag seq is 
        savemat(
            "data/traj_data_{}.mat".format(traj_i),
            {"xk_real_seq": xk_real_seq.T, "aux_inputk_seq": refk_seq.T, "drag_seq":drag_seq.T}
        )
        print(traj_i, "trajectories generated.")
        traj_i += 1
    