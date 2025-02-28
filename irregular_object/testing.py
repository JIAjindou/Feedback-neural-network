import os
import argparse
import time
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim
import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams['font.family'] = 'Calibri'

# Global parameters
parser = argparse.ArgumentParser('Trajectory Prediction')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=10)  # maximum predicted time of mimi_batch
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=1000)  # Maximum number of iterations
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

# ODE solver
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# Load testing dataset
with torch.no_grad():
    Testing_dataset = torch.tensor(loadmat('Dataset_construction/Testing_dataset.mat')['Testing_dataset'])
    # Acc_truth = torch.zeros([3, 1058, 1, 1])

    case_num = Testing_dataset.size(0)
    data_size = Testing_dataset.size(1)
    variable_num = Testing_dataset.size(2) - 3

    Acc_truth = Testing_dataset[:, :data_size, 0:3].float().to(device)
    PVA_truth = Testing_dataset[:, :data_size, 3:].float().to(device)
    PVA0 = Testing_dataset[:, 0, 3:].float().to(device)
    t = torch.linspace(0., data_size * 0.001, data_size).to(device)

    sample_time = 0.001
    pre_steps = 500  # prediction steps
    pre_time = sample_time * pre_steps


# Make a new folder
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# Neural networks
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(variable_num, 100),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(100, 100),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(100, 100),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(100, variable_num),
        )

        for m in self.net.modules():  # 参数初始化
            if isinstance(m, nn.Linear):  # 判断是否为线性层
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


if args.viz:
    makedirs('png')


def visualize1(Postion_truth, pre_results, pre_results_FNN, k):
    index = k + 1
    position_plot = fig1.add_subplot(3, 3, index, projection='3d')

    color_gray = (102 / 255, 102 / 255, 102 / 255)
    color_blue = (76 / 255, 147 / 255, 173 / 255)
    color_red = (1, 0, 0)

    # Position plot
    position_plot.cla()
    # position_plot.set_title('Test trajectory - {:01d}'.format(k+1), fontsize=15, pad=10)
    position_plot.set_xlabel('x [m]', fontsize=12)
    position_plot.set_ylabel('y [m]', fontsize=12)
    position_plot.set_zlabel('z [m]', fontsize=12)
    position_plot.tick_params(axis='x', labelsize=12)
    position_plot.tick_params(axis='y', labelsize=12)
    position_plot.tick_params(axis='z', labelsize=12)
    x_true = Postion_truth.detach().numpy()[k, :, 0]
    y_true = Postion_truth.detach().numpy()[k, :, 1]
    z_true = Postion_truth.detach().numpy()[k, :, 2]
    x_learnt = pre_results.detach().numpy()[k, :, 0]
    y_learnt = pre_results.detach().numpy()[k, :, 1]
    z_learnt = pre_results.detach().numpy()[k, :, 2]
    x_FNN = pre_results_FNN.detach().numpy()[k, :, 0]
    y_FNN = pre_results_FNN.detach().numpy()[k, :, 1]
    z_FNN = pre_results_FNN.detach().numpy()[k, :, 2]

    position_plot.plot(x_true, y_true, z_true, '--', color=color_gray, linewidth=1.5, label='Truth')
    # position_plot.plot(x_learnt, y_learnt, z_learnt,  color=color_blue, linewidth=1.5, label='Neural ODE')
    position_plot.plot(x_FNN, y_FNN, z_FNN, color=color_red, linewidth=1.5, label='Feedback NN')
    if k == 0:
        position_plot.legend(loc='lower left', fontsize=13)
    # position_plot.grid(False)

    timestamp = time.time()
    now = time.localtime(timestamp)
    month = now.tm_mon
    day = now.tm_mday

    # Figure show
    # fig1.tight_layout()
    plt.savefig('png/testing_trajectories{:02d}{:02d}'.format(month, day))
    plt.draw()
    plt.pause(0.0001)


def visualize2(acc_truth, acc_pre, acc_pre_FNN, k):
    index = k + 1
    acc_plot = fig2.add_subplot(3, 3, index)

    color_gray = (102 / 255, 102 / 255, 102 / 255)
    color_blue = (76 / 255, 147 / 255, 173 / 255)
    color_red = (1, 0, 0)
    color_green = (0, 1, 0)

    # Figure: Acceleration
    acc_plot.cla()
    # acc_plot.set_title('Trained acceleration on testing set', fontsize=15, pad=10)
    acc_plot.set_xlabel('t [$s$]', fontsize=15)
    acc_plot.set_ylabel('Acceleration [$m^2/s$]', fontsize=15)
    acc_plot.tick_params(axis='x', labelsize=14)
    acc_plot.tick_params(axis='y', labelsize=14)
    x_true = acc_truth.detach().numpy()[k, :, 0]
    y_true = acc_truth.detach().numpy()[k, :, 1]
    z_true = acc_truth.detach().numpy()[k, :, 2]
    x_learnt = acc_pre.detach().numpy()[k, :, 0]
    y_learnt = acc_pre.detach().numpy()[k, :, 1]
    z_learnt = acc_pre.detach().numpy()[k, :, 2]
    x_FNN = acc_pre_FNN.detach().numpy()[k, :, 0]
    y_FNN = acc_pre_FNN.detach().numpy()[k, :, 1]
    z_FNN = acc_pre_FNN.detach().numpy()[k, :, 2]

    tt = t[:data_size - pre_steps].detach().numpy()
    acc_plot.plot(tt, x_true, '--', color=color_blue, alpha=0.5, linewidth=1.5, label='Truth x')
    # acc_plot.plot(tt, x_learnt, linewidth=1.5, label='Neural ODE x')
    acc_plot.plot(tt, x_FNN, color=color_blue, linewidth=1.5, label='Learnt x')
    acc_plot.plot(tt, y_true, '--', color=color_red, alpha=0.5, linewidth=1.5, label='Truth y')
    # acc_plot.plot(tt, y_learnt, linewidth=1.5, label='Neural ODE y')
    acc_plot.plot(tt, y_FNN, color=color_red, linewidth=1.5, label='Learnt y')
    acc_plot.plot(tt, z_true, '--', color=color_gray, alpha=0.5, linewidth=1.5, label='Truth z')
    # acc_plot.plot(tt, z_learnt, linewidth=1.5, label='Neural ODE z')
    acc_plot.plot(tt, z_FNN, color=color_gray, linewidth=1.5, label='Learnt z')

    if k == 0:
        acc_plot.legend(loc='lower left', fontsize=13)

    timestamp = time.time()
    now = time.localtime(timestamp)
    month = now.tm_mon
    day = now.tm_mday

    # Figure show
    fig2.tight_layout()
    plt.savefig('png/testing_acc{:02d}{:02d}'.format(month, day))
    plt.draw()
    plt.pause(0.0001)


def visualize3(pre_error, pre_error_FNN, pre_error_gradient, pre_error_drag):
    error_plot = fig3.add_subplot(111)

    color_gray = (102 / 255, 102 / 255, 102 / 255)
    color_blue = (76 / 255, 147 / 255, 173 / 255)
    color_red = (1, 0, 0)
    color_green = (0, 1, 0)

    # Figure-1: Error
    error_plot.cla()
    error_plot.set_title('Trajectory prediction results', fontsize=15, pad=10)
    error_plot.set_xlabel('t [s]', fontsize=15)
    error_plot.set_ylabel('Prediction error [m]', fontsize=15)
    error_plot.tick_params(axis='x', labelsize=14)
    error_plot.tick_params(axis='y', labelsize=14)
    tt = t[:data_size - pre_steps].detach().numpy()
    average_error = torch.mean(pre_error, dim=0).detach().numpy()
    std_error = torch.std(pre_error, dim=0).detach().numpy()
    average_error_FNN = torch.mean(pre_error_FNN, dim=0).detach().numpy()
    std_error_FNN = torch.std(pre_error_FNN, dim=0).detach().numpy()
    average_error_drag = torch.mean(pre_error_drag, dim=0).detach().numpy()
    std_error_drag = torch.std(pre_error_drag, dim=0).detach().numpy()
    average_error_gradient = torch.mean(pre_error_gradient, dim=0).detach().numpy()
    std_error_gradient = torch.std(pre_error_gradient, dim=0).detach().numpy()

    error_plot.plot(tt, average_error,  color=color_blue, linewidth=1.5, label='Neural ODE')
    error_plot.fill_between(tt, average_error + std_error, average_error - std_error, color=color_blue, alpha=0.1)
    error_plot.fill_between(tt, average_error - std_error, average_error + std_error, color=color_blue, alpha=0.1)
    error_plot.plot(tt, average_error_FNN, color=color_red, linewidth=1.5, label='Feedback NN')
    error_plot.fill_between(tt, average_error_FNN + std_error_FNN, average_error_FNN - std_error_FNN, color=color_red, alpha=0.1)
    error_plot.fill_between(tt, average_error_FNN - std_error_FNN, average_error_FNN + std_error_FNN, color=color_red,
                            alpha=0.1)

    # error_plot.plot(tt, average_error_gradient, color=color_gray, linewidth=1.5, label='Gradient')
    # error_plot.fill_between(tt, average_error_gradient + std_error_gradient, average_error_gradient - std_error_gradient, color=color_gray, alpha=0.1)
    # error_plot.fill_between(tt, average_error_gradient - std_error_gradient, average_error_gradient + std_error_gradient, color=color_gray, alpha=0.1)

    error_plot.plot(tt, average_error_drag, color=color_gray, linewidth=1.5, label='Model based')
    error_plot.fill_between(tt, average_error_drag + std_error_drag, average_error_drag - std_error_drag, color=color_gray, alpha=0.1)
    error_plot.fill_between(tt, average_error_drag - std_error_drag, average_error_drag + std_error_drag, color=color_gray, alpha=0.1)
    error_plot.set_xlim(tt.min(), tt.max())

    error_plot.legend(loc='lower right', fontsize=13)

    timestamp = time.time()
    now = time.localtime(timestamp)
    month = now.tm_mon
    day = now.tm_mday

    # Figure show
    fig3.tight_layout()
    plt.savefig('png/testing_errors{:02d}{:02d}'.format(month, day))
    plt.draw()
    plt.pause(0.0001)


if __name__ == '__main__':

    funcODE = torch.load('trained_model/Neural_ODE_ite1000_loss0.048128.pt', weights_only=False).to(device)

    # 1.Prediction - Neural ODE
    pre_results = torch.zeros(case_num, data_size - pre_steps, 3)
    pre_error = torch.zeros(case_num, data_size - pre_steps)
    pre_acc = torch.zeros(case_num, data_size - pre_steps, 3)
    tamp = t[:pre_steps]
    count = 0
    for jj in range(data_size - pre_steps):
        count += 1
        print('Predicting wit Neural ODE: {:02d}%'.format(count*100//(data_size - pre_steps)))
        pre = odeint(funcODE, PVA_truth[:, jj, :], tamp)
        pre_last = pre[pre_steps - 1, :, :3]
        pre_results[:, jj, :] = pre_last
        pre_acc[:, jj, :] = funcODE(0, PVA_truth[:, jj, :])[:, 3:6]
        pre_error[:, jj] = torch.sqrt(torch.sum((pre_last - PVA_truth[:, jj + pre_steps, :3])**2, dim=1))

    # 2.Prediction - feedback NN
    pre_results_FNN = torch.zeros(case_num, data_size - pre_steps, 3)
    pre_error_FNN = torch.zeros(case_num, data_size - pre_steps)
    pre_acc_FNN = torch.zeros(case_num, data_size - pre_steps, 3)

    temp = odeint(funcODE, PVA0, t[:pre_steps + 1])
    hat_NN = temp[1:, :, :]
    count = 0
    decay_rate = 0.02
    feedback_gain = 5.
    last_output = torch.zeros(case_num, variable_num)
    for jj in range(data_size - pre_steps):
        count += 1
        print('Predicting wit Feedback NN: {:02d}%'.format(count*100//(data_size - pre_steps)))
        for kk in range(pre_steps):
            if kk == 0:
                input_NN = PVA_truth[:, jj, :]
            else:
                input_NN = last_output
            # L decays as the prediction depth increases
            L_decay = (torch.eye(variable_num) * feedback_gain * math.exp(-kk * decay_rate))
            # Predict next state
            output_NN = funcODE(0, input_NN)
            output_FNN = output_NN + torch.mm(input_NN - hat_NN[kk, :, :], L_decay)
            if kk == 0:
                pre_acc_FNN[:, jj, :] = output_FNN[:, 3:6]
            last_output[:, :3] = input_NN[:, :3] + input_NN[:, 3:] * sample_time + 1/2 * output_FNN[:, 3:] * sample_time ** 2
            last_output[:, 3:] = input_NN[:, 3:] + output_FNN[:, 3:] * sample_time
            hat_NN[kk, :, :] = hat_NN[kk, :, :] + output_FNN * sample_time
        pre_results_FNN[:, jj, :] = last_output[:, :3]
        summ = torch.sum((last_output[:, :3] - PVA_truth[:, jj + pre_steps, :3]) ** 2, dim=1)
        pre_error_FNN[:, jj] = torch.sqrt(summ)

    # 3.Prediction - Gradient
    pre_results_gradient = torch.zeros(case_num, data_size - pre_steps, 3)
    pre_error_gradient = torch.zeros(case_num, data_size - pre_steps)
    count = 0
    # for jj in range(data_size - pre_steps):
    #     count += 1
    #     print('Predicting wit gradient based method: {:02d}%'.format(count*100//(data_size - pre_steps)))
    #     pre_results_gradient[:, jj, :2] = PVA_truth[:, jj, :2] + PVA_truth[:, jj, 3:5] * pre_time
    #     pre_results_gradient[:, jj, 2] = (PVA_truth[:, jj, 2] + PVA_truth[:, jj, 5] * pre_time + 1 / 2 *
    #                                       (-9.8) * pre_time ** 2)
    #     pre_error_gradient[:, jj] = torch.sqrt(torch.sum((pre_results_gradient[:, jj, :3] -
    #                                                       PVA_truth[:, jj + pre_steps, :3]) ** 2, dim=1))

    # 4.Prediction - Drag model
    coef_ETH = torch.tensor([-0.096222441713880, -0.103286781439618, - 0.104106381096502])
    pre_results_drag = torch.zeros(case_num, data_size - pre_steps, 3)
    pre_error_drag = torch.zeros(case_num, data_size - pre_steps)
    count = 0
    for jj in range(data_size - pre_steps):
        count += 1
        print('Predicting wit model based method: {:02d}%'.format(count*100//(data_size - pre_steps)))
        for kk in range(case_num):
            pos = PVA_truth[kk, jj, 0:3]
            vel = PVA_truth[kk, jj, 3:6]
            # norm = torch.sqrt(torch.sum(vel ** 2))
            # acc_ETH = coef_ETH * norm * vel
            # acc_ETH[2] = acc_ETH[2] - 9.8
            # pos = pos + vel * pre_time + 1 / 2 * acc_ETH * pre_time ** 2
            for ii in range(pre_steps):
                norm = torch.sqrt(torch.sum(vel ** 2))
                acc_ETH = coef_ETH * norm * vel
                acc_ETH[2] = acc_ETH[2] - 9.8
                pos = pos + vel * sample_time + 1 / 2 * acc_ETH * sample_time ** 2
                vel = vel + acc_ETH * sample_time
            pre_error_drag[kk, jj] = torch.sqrt(torch.sum((pos - PVA_truth[kk, jj + pre_steps, :3]) ** 2))

    '''plot testing results'''
    # plot the predicted position
    fig1 = plt.figure(figsize=(11, 9), facecolor='white')
    for k in range(case_num):
        visualize1(PVA_truth[:, pre_steps:, :3], pre_results, pre_results_FNN, k)

    # plot the learnt acceleration
    fig2 = plt.figure(figsize=(10, 9), facecolor='white')
    for k in range(case_num):
        visualize2(Acc_truth[:, :data_size - pre_steps, :], pre_acc[:, :, :],
              pre_acc_FNN[:, :, :], k)

    # plot the prediction error
    fig3 = plt.figure(figsize=(5, 4), facecolor='white')
    visualize3(pre_error, pre_error_FNN, pre_error_gradient, pre_error_drag)


    # python training.py --viz


