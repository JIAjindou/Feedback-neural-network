import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams['font.family'] = 'Calibri'

# Global parameters
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--niters', type=int, default=1000)  # Maximum number of iterations
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')

parser.add_argument('--start_point', type=float, default=9.)
parser.add_argument('--end_time', type=int, default=20)
parser.add_argument('--prediction_step', type=int, default=50)  # Prediction steps
parser.add_argument('--case_num', type=int, default=20)  # The number of cases through domain randomization
args = parser.parse_args()

# ODE solver
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# Initial value of trajectory
true_y0 = torch.tensor([[args.start_point, 0.]]).to(device)
t = torch.linspace(0., args.end_time, args.data_size).to(device)  # Observation moments
# true_A_nominal = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)  # Parameter of training function
true_A_nominal = torch.tensor([[-0.12, 2.4], [-2.4, -0.12]]).to(device)  # Parameter of training function


# Training function
class Lambda_nominal(nn.Module):
    def forward(self, t, y):
        return torch.mm(y, true_A_nominal)
        # return torch.mm(y, true_A_nominal) + 8.


# Training functions - domain randomization
class Lambda_train(nn.Module):
    def forward(self, t, y):
        return torch.mm(y, true_A_domran) + add_dis


# Observations in training
with torch.no_grad():
    true_y_nominal = odeint(Lambda_nominal(), true_y0, t, method='dopri5').to(
        device)  # Observations on nominal function
    true_y_domran = torch.zeros(args.case_num, args.data_size, 1, 2).to(device)  # Observations on multiple functions
    true_dydt_domran = torch.zeros(args.case_num, args.data_size, 2).to(device)
    for i in range(args.case_num):
        A_decay = 0.04 + i * 0.005  # [0.04 1 0.14]
        A_period = 0.8 + i * 0.12  # [0.8 2 3.2]
        add_dis = -24. + i * 2.4
        true_A_domran = torch.tensor([[-A_decay, A_period], [-A_period, -A_decay]]).to(device)
        temp = odeint(Lambda_train(), true_y0, t, method='dopri5').to(device)
        true_y_domran[i] = temp
        #  true_dydt_domran[i] = torch.mm(temp.reshape(-1, 2), true_A_domran) + add_dis  # [case_num, data_size, 2]


# Make a new folder
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# Neural networks
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():  # 参数初始化
            if isinstance(m, nn.Linear):  # 判断是否为线性层
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


# Neural networks
class BackFunc(nn.Module):
    def __init__(self):
        super(BackFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():  # 参数初始化
            if isinstance(m, nn.Linear):  # 判断是否为线性层
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if args.viz:
    makedirs('png')
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    plt.show(block=False)


# Visualize the testing performance
def visualize(dydt_true_test, dydt_ODEnn_test, dydt_FNN_test, kk):
    color_gray = (102 / 255, 102 / 255, 102 / 255)
    color_blue = (76 / 255, 147 / 255, 173 / 255)
    color_red = (1, 0, 0)
    index = (kk-4) * 2 + 1
    ax_dydt_ODEnn_test = fig.add_subplot(4, 6, index, frameon=False)
    ax_dydt_FNN_test = fig.add_subplot(4, 6, index + 1, frameon=False)

    # Figure-1: Training performance of Neural ODE
    ax_dydt_ODEnn_test.cla()
    ax_dydt_ODEnn_test.set_title('Neural ODE'.format(kk-3), fontsize=15, pad=10)
    x_true1 = dydt_true_test.detach().numpy()[:, 0]
    y_true1 = dydt_true_test.detach().numpy()[:, 1]
    x_test1 = dydt_ODEnn_test.cpu().detach().numpy()[:, 0, 0]
    y_test1 = dydt_ODEnn_test.cpu().detach().numpy()[:, 0, 1]
    Error_xy1 = np.sqrt((x_test1 - x_true1) ** 2 + (y_test1 - y_true1) ** 2)
    upper1 = Error_xy1.max()

    ax_dydt_ODEnn_test.plot(x_true1, y_true1, '--', color=color_gray, linewidth=1.5, label='Truth')
    ax_dydt_ODEnn_test.scatter(x=x_true1[0], y=y_true1[0], s=100, marker='*', color=color_gray)
    ax_dydt_ODEnn_test.scatter(x=x_test1[0], y=y_test1[0], s=100, marker='*', color=color_red)

    # Figure-2: Feedback neural networks
    ax_dydt_FNN_test.cla()
    ax_dydt_FNN_test.set_title('Feedback neural network', fontsize=15, pad=10)

    x_true2 = dydt_true_test.detach().numpy()[:args.data_size - 1, 0]
    y_true2 = dydt_true_test.detach().numpy()[:args.data_size - 1, 1]
    x_test2 = dydt_FNN_test.detach().numpy()[:, 0, 0]
    y_test2 = dydt_FNN_test.detach().numpy()[:, 0, 1]
    Error_xy2 = np.sqrt((x_test2 - x_true2) ** 2 + (y_test2 - y_true2) ** 2)
    upper2 = Error_xy2.max()

    ax_dydt_FNN_test.plot(x_true2, y_true2, '--', color=color_gray, linewidth=1.5, label='Truth')
    ax_dydt_FNN_test.scatter(x=x_true2[0], y=y_true2[0], s=100, marker='*', color=color_gray)
    ax_dydt_FNN_test.scatter(x=x_test2[0], y=y_test2[0], s=100, marker='*', color=color_red)

    # 颜色映射
    points1 = np.array([x_test1, y_test1]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points1[:-1], points1[1:]], axis=1)
    norm = plt.Normalize(0, 4.5)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(Error_xy1)
    lc.set_linewidth(2)
    line = ax_dydt_ODEnn_test.add_collection(lc)
    ax_dydt_ODEnn_test.set_aspect('equal', adjustable='box')
    ax_dydt_ODEnn_test.set_xticks([])
    ax_dydt_ODEnn_test.set_yticks([])
    ax_dydt_ODEnn_test.set_xlim(min(x_true1.min(), x_test1.min())-1, max(x_true1.max(), x_test1.max())+1)
    ax_dydt_ODEnn_test.set_ylim(min(y_true1.min(), y_test1.min())-1, max(y_true1.max(), y_test1.max())+1)

    points2 = np.array([x_test2, y_test2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points2[:-1], points2[1:]], axis=1)
    norm = plt.Normalize(0, max(upper1, upper2))
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(Error_xy2)
    lc.set_linewidth(2)
    line = ax_dydt_FNN_test.add_collection(lc)
    cbar = fig.colorbar(line, shrink=1)
    cbar.ax.tick_params(labelsize=14)
    # ax_dydt_FNN_test.legend(loc='lower right', fontsize=13)
    ax_dydt_FNN_test.set_aspect('equal', adjustable='box')
    ax_dydt_FNN_test.set_xticks([])
    ax_dydt_FNN_test.set_yticks([])
    ax_dydt_FNN_test.set_xlim(min(x_true2.min(), x_test2.min())-1, max(x_true2.max(), x_test2.max())+1)
    ax_dydt_FNN_test.set_ylim(min(y_true2.min(), y_test2.min())-1, max(y_true2.max(), y_test2.max())+1)

    # Obtain current time
    timestamp = time.time()
    now = time.localtime(timestamp)
    month = now.tm_mon
    day = now.tm_mday

    # Figure show
    fig.tight_layout()
    plt.savefig('png/d_FeedbackNN_test{:02d}{:02d}'.format(month, day))
    plt.draw()
    plt.pause(0.0001)


if __name__ == '__main__':
    # load trained_model
    funcODE = torch.load('trained_model/Neural_ODE_ite400_loss0.2.pt', weights_only=False).to(device)
    funcBack = torch.load('trained_model/FeedbackNN_ite477_loss0.494530.pt', weights_only=False).to(device)
    sample_time = args.end_time / args.data_size  # Sample time
    Lambda_train = Lambda_train().to(device)

    for kk in range(4, args.case_num-4):
        # 1) True dydt
        A_decay = 0.04 + kk * 0.005  # [0.04 1 0.14]
        A_period = 0.8 + kk * 0.12  # [0.8 2 3.2]
        add_dis = -24. + kk * 2.4
        true_A_domran = torch.tensor([[-A_decay, A_period], [-A_period, -A_decay]]).to(device)
        dydt_true_test = Lambda_train(0, true_y_domran[kk].reshape(-1, 2)).to(device)

        # 2) Learned dydt of Neural ODE
        dydt_ODEnn_test = funcODE(0, true_y_domran[kk]).to(device)

        # 2) Learned dydt of feedback neural network
        y_hat_test = torch.tensor([[args.start_point, 0.]])  # Estimated observation
        dydt_FNN_test = torch.zeros((args.data_size - 1, 1, 2))
        for j in range(args.data_size - 1):
            y0_test = true_y_domran[kk, j, :, :]  # Initial value at each moment
            # label_pre_test[j, :, :] = funcBack(0, (y0_test - y_hat))
            dydt_hat = dydt_ODEnn_test[j, :, :] + funcBack(0, (y0_test - y_hat_test))  # Correct dydt by feedback
            y_hat_test = y_hat_test + dydt_hat * sample_time
            dydt_FNN_test[j, :, :] = dydt_hat

        visualize(dydt_true_test, dydt_ODEnn_test, dydt_FNN_test, kk)

    # python d_FeedbackNN_test.py --viz

