import numpy as np
import casadi as ca

# Simulation & Control Setup
SIMSTEPS = 300
TIMESTEP = 0.02
HORIZON = 10
TRAJ_SPEEDRATE = 1.0

"""Import Models"""
from model.model_sim import Quadrotor_Sim  # FOR SIMULATION
from model.model_nominal import Quadrotor  # FOR MPC
from model.model_NODE import Quadrotor_NODE  # FOR MPC
from model.model_NODE_adapt import Quadrotor_NODE_Adapt  # FOR MPC
from mpc.solver import *

# add bias in aerodynamic effects, mass and inertia
model_sim = Quadrotor_Sim()
model_sim.disturb = np.array([0.3] * 3 + [0.0] * 3)
model_sim.aero_D = np.diag([0.6, 0.6, 0.15])
# model_sim.m_actual += 0.5
# model_sim.Ixx_actual += 2e-3
# model_sim.Iyy_actual += 2e-3

# Prepare model for model predictive control
params_NODE = ca.DM(np.load("learning_NODE/temp/model_param.npy"))
params_MLP = np.load("learning_MLP/temp/model_param.npy")
# from learning_MLP.core import Core
# from model.model_drag_MLP import Drag_MLP
# core = Core()
# model_MLP = Drag_MLP()
# params_MLP = core.pth2params_savenpy(model_MLP, 
#                 pth_path="learning_MLP/learned_models/Nov17_15-52-54_2024_minibatch_torchADAM/model_param_epoch500.pth",
#                 npy_path="learning_MLP/temp/model_param.npy") 
# print(params_NODE.shape, params_MLP.shape)

model_nominal = Quadrotor()

model_NODE = Quadrotor_NODE(discrete_h_learn=TIMESTEP)
model_NODE.load_params_for_OC(params=params_NODE)

model_MLP = Quadrotor_NODE(discrete_h_learn=TIMESTEP)
model_MLP.load_params_for_OC(params=params_MLP)

model_NODE_adapt = Quadrotor_NODE_Adapt(adaptive_gain=8.0)
model_NODE_adapt.load_params(
    paramsFull=params_NODE
)

# Prepare mpc controllers
mpc_standard_nominal = MPC(model=model_nominal, discrete_h=TIMESTEP, H=HORIZON)
mpc_standard_NODE = MPC(model=model_NODE, discrete_h=TIMESTEP, H=HORIZON)
mpc_standard_MLP = MPC(model=model_MLP, discrete_h=TIMESTEP, H=HORIZON)
mpc_multistep_nominal = MPC_MultiStep(
    model=model_nominal, discrete_h=TIMESTEP, H=HORIZON
)
mpc_multistep_NODE = MPC_MultiStep(model=model_NODE, discrete_h=TIMESTEP, H=HORIZON)
mpc_multistep_nominal.set_GainAndDecay(L_init=np.diag([3.0] * 12), decay_rate=0.1)
mpc_multistep_NODE.set_GainAndDecay(L_init=np.diag([3.0] * 12), decay_rate=0.1)
mpc_adapt = AdaptMPC(model=model_NODE_adapt, discrete_h=TIMESTEP, H=HORIZON)
mpc_adapt_multistep = AdaptMPC_MultiStep(model=model_NODE_adapt, discrete_h=TIMESTEP, H=HORIZON)
mpc_adapt_multistep.set_GainAndDecay(L_init=np.diag([3.0] * 12), decay_rate=0.1)

"""Simulation Setup: Tracking 3D Lissajous Trajectory"""
from aux_module.trajectory_tools import coeff_to_pointsLissajous

# Get initial state for trajectory tracking
# using differential flatness mapping
pvaj_0 = coeff_to_pointsLissajous(
    0.0,
    a=1.0,
    a0=TRAJ_SPEEDRATE,
    Radi=np.array([3.0, 3.0, 0.5]),
    Period=np.array([6.0, 3.0, 3.0]),
    h=-0.5,
)
x0 = model_nominal.ref2x_map(pvaj_0)

# Get position reference trajectory for plotting
pos_reference = []
for i in range(SIMSTEPS):
    pvaj = coeff_to_pointsLissajous(
        i * TIMESTEP,
        a=1.0,
        a0=TRAJ_SPEEDRATE,
        Radi=np.array([3.0, 3.0, 0.5]),
        Period=np.array([6.0, 3.0, 3.0]),
        h=-0.5,
    )
    pos_reference += [pvaj[:3].reshape((-1, 1))]
pos_reference = np.hstack(pos_reference)


# Obtaining the upcomming state and input reference trajectories
def get_XUref(t):
    xrefk_seq = []
    urefk_seq = []
    for i in range(HORIZON):
        pvaj = coeff_to_pointsLissajous(
            t + i * TIMESTEP,
            a=1.0,
            a0=TRAJ_SPEEDRATE,
            Radi=np.array([3.0, 3.0, 0.5]),
            Period=np.array([6.0, 3.0, 3.0]),
            h=-0.5,
        )
        # Compute x, u reference using differential flatness mapping (pvaj -> x, u)
        xrefk_seq += [model_nominal.ref2x_map(pvaj).reshape((-1, 1))]
        urefk_seq += [model_nominal.ref2u_map(pvaj).reshape((-1, 1))]

    return np.hstack(xrefk_seq), np.hstack(urefk_seq)


# Simulation Main Function
def sim_main(idx, controller):
    dx_real = []
    dx_pred = []
    lossk_seq = []
    t = 0
    xk = np.squeeze(x0, axis=1)
    xsim_seq = [x0]
    # Refresh in mpc_multistep and adapt_mpc
    if 3 <= idx <= 5 or idx == 7:
        controller.refresh()
    for i in range(SIMSTEPS):  # main loop for simulation
        if i % 10 == 0:
            print("Current Case: {0}, step {1}/{2}".format(idx, i, SIMSTEPS))
        # Get reference trajectory
        xrefk_seq, urefk_seq = get_XUref(t)
        # Solve OC
        if idx <= 2 or idx == 6:
            u_opt_seq = controller.solve_OC(xk, xrefk_seq, urefk_seq)[0]
        elif 3 <= idx <= 4 or idx == 7:
            u_opt_seq = controller.solve_OC_MultiStepPred(xk, xrefk_seq, urefk_seq)[0]
        elif idx == 5:
            u_opt_seq = controller.solve_OC_adapt(xk, xrefk_seq, urefk_seq)[0]
        uk = u_opt_seq[:, 0]
        # Compute losskx
        lossk_seq += [controller.loss_kx(xk, xrefk_seq[:, 0])]
        # Rollout out simulation model
        xk1 = model_sim.openloop_sym_exRK4(xk, uk, TIMESTEP)
        # Get predicted dx v.s. real dx
        dx_real_i = model_sim.nominal_sym(xk, uk)
        dx_pred_i = controller.dx_modelpredict(xk, uk)
        dx_real += [dx_real_i]
        dx_pred += [dx_pred_i]
        # Store sim state
        xsim_seq += [xk1.reshape((-1, 1))]
        # Update sim state
        xk = xk1
        # Update time step
        t += TIMESTEP

    return (
        np.hstack(xsim_seq),
        np.hstack(dx_real),
        np.hstack(dx_pred),
        np.array(lossk_seq),
    )


"""Go Simulation and Visualization"""
from scipy.io import savemat

controller_list = [
    # (1, mpc_standard_nominal),
    # (2, mpc_standard_NODE),
    # (3, mpc_multistep_nominal),
    # (4, mpc_multistep_NODE),
    # (5, mpc_adapt),
    # (6, mpc_standard_MLP),
    # (7, mpc_adapt_multistep)
]


for idx, controller in controller_list:
    # Simulate
    xsim_seq, dx_real, dx_pred, lossk_seq = sim_main(idx, controller)

    # Save Trajectory
    savemat(
        "sim_mpc_trajs/sim_case{}.mat".format(idx),
        {
            "xsim_seq": xsim_seq,
            "pos_reference": pos_reference,
            "dx_real": dx_real,
            "dx_pred": dx_pred,
            "lossk_seq": lossk_seq,
        },
    )
