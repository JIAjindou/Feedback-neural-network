import os
import argparse
import time
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim

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

# Load training dataset
with torch.no_grad():
    Training_dataset = torch.tensor(loadmat('Dataset_construction/Training_dataset.mat')['Training_dataset'])
    # Acc_truth = torch.zeros([3, 1058, 1, 1])
    Acc_truth = Training_dataset[:, :, 0:3].float().to(device)
    PVA_truth = Training_dataset[:, :, 3:].float().to(device)

    case_num = Training_dataset.size(0)
    data_size = Training_dataset.size(1)
    variable_num = Training_dataset.size(2) - 3

    t = torch.linspace(0., 1.058, 1058).to(device)
    PVA0 = PVA_truth[:, 0, :].to(device)


# Constructing the mini-bach dataset for training
def get_batch():
    s1 = torch.from_numpy(
        np.random.choice(np.arange(data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_t = t[:args.batch_time]
    batch_PVA0 = PVA_truth[:, s1, :] # (case, variables, data_size)
    batch_PVA_pre = torch.stack([PVA_truth[:, s1 + i, :] for i in range(args.batch_time)], dim=0)  # (T, M, D) [10, 20, 1, 2]
    return batch_PVA0.to(device), batch_PVA_pre.to(device), batch_t.to(device)


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


def visualize(true, learnt, Acc_NN, Acc_truth):
    if args.viz:
        makedirs('png')

        fig = plt.figure(figsize=(15, 5), facecolor='white')

        position_plot = fig.add_subplot(131, projection='3d')
        velocity_plot = fig.add_subplot(132)
        acc_plot = fig.add_subplot(133)

        color_gray = (102 / 255, 102 / 255, 102 / 255)
        color_blue = (76 / 255, 147 / 255, 173 / 255)
        color_red = (1, 0, 0)

        # Figure-1: Position
        position_plot.cla()
        position_plot.set_title('Trained position on training set', fontsize=15, pad=10)
        position_plot.set_xlabel('${x}$', fontsize=15)
        position_plot.set_ylabel('${y}$', fontsize=15)
        position_plot.set_ylabel('${z}$', fontsize=15)
        position_plot.tick_params(axis='x', labelsize=14)
        position_plot.tick_params(axis='y', labelsize=14)
        position_plot.tick_params(axis='z', labelsize=14)
        x_true = true.detach().numpy()[:, 0]
        y_true = true.detach().numpy()[:, 1]
        z_true = true.detach().numpy()[:, 2]
        x_learnt = learnt.detach().numpy()[:, 0]
        y_learnt = learnt.detach().numpy()[:, 1]
        z_learnt = learnt.detach().numpy()[:, 2]

        position_plot.plot(x_true, y_true, z_true, '--', color=color_gray, linewidth=1.5, label='Truth')
        position_plot.plot(x_learnt, y_learnt, z_learnt,  color=color_red, linewidth=1.5, label='Learnt')
        # ax_dydt_train_NN.scatter(x=x_true[0], y=y_true[0], z=z_true[0], s=100, marker='*', color=color_gray)
        # ax_dydt_train_NN.scatter(x=x_learnt[0], y=y_learnt[0], z=z_learnt[0], s=100, marker='*', color=color_red)

        position_plot.legend(loc='lower right', fontsize=13)
        position_plot.set_aspect('equal', adjustable='box')

        # Figure-2: Velocity
        velocity_plot.cla()
        velocity_plot.set_title('Trained velocity on training set', fontsize=15, pad=10)
        velocity_plot.set_xlabel('$t$', fontsize=15)
        velocity_plot.set_ylabel('$velocity$', fontsize=15)
        velocity_plot.tick_params(axis='x', labelsize=14)
        velocity_plot.tick_params(axis='y', labelsize=14)
        tt = t.detach().numpy()
        x_true = true.detach().numpy()[:, 3]
        y_true = true.detach().numpy()[:, 4]
        z_true = true.detach().numpy()[:, 5]
        x_learnt = learnt.detach().numpy()[:, 3]
        y_learnt = learnt.detach().numpy()[:, 4]
        z_learnt = learnt.detach().numpy()[:, 5]

        velocity_plot.plot(tt, x_true, '--', color=color_gray, linewidth=1.5, label='Truth velx')
        velocity_plot.plot(tt, x_learnt, linewidth=1.5, label='Learnt velx')
        velocity_plot.plot(tt, y_true, '--', color=color_gray, linewidth=1.5, label='Truth vely')
        velocity_plot.plot(tt, y_learnt, linewidth=1.5, label='Learnt vely')
        velocity_plot.plot(tt, z_true, '--', color=color_gray, linewidth=1.5, label='Truth velz')
        velocity_plot.plot(tt, z_learnt, linewidth=1.5, label='Learnt velz')
        # ax_dydt_train_NN.scatter(x=x_true[0], y=y_true[0], z=z_true[0], s=100, marker='*', color=color_gray)
        # ax_dydt_train_NN.scatter(x=x_learnt[0], y=y_learnt[0], z=z_learnt[0], s=100, marker='*', color=color_red)

        velocity_plot.legend(loc='lower right', fontsize=13)

        # Figure-3: Acceleration
        acc_plot.cla()
        acc_plot.set_title('Trained acceleration on training set', fontsize=15, pad=10)
        acc_plot.set_xlabel('$t$', fontsize=15)
        acc_plot.set_ylabel('$Acceleration$', fontsize=15)
        acc_plot.tick_params(axis='x', labelsize=14)
        acc_plot.tick_params(axis='y', labelsize=14)
        tt = t.detach().numpy()
        x_true = Acc_truth.detach().numpy()[:, 0]
        y_true = Acc_truth.detach().numpy()[:, 1]
        z_true = Acc_truth.detach().numpy()[:, 2]
        x_learnt = Acc_NN.detach().numpy()[:, 3]
        y_learnt = Acc_NN.detach().numpy()[:, 4]
        z_learnt = Acc_NN.detach().numpy()[:, 5]

        acc_plot.plot(tt, x_true, '--', color=color_gray, linewidth=1.5, label='Truth accx')
        acc_plot.plot(tt, x_learnt, linewidth=1.5, label='Learnt accx')
        acc_plot.plot(tt, y_true, '--', color=color_gray, linewidth=1.5, label='Truth accy')
        acc_plot.plot(tt, y_learnt, linewidth=1.5, label='Learnt accy')
        acc_plot.plot(tt, z_true, '--', color=color_gray, linewidth=1.5, label='Truth accz')
        acc_plot.plot(tt, z_learnt, linewidth=1.5, label='Learnt accz')
        # ax_dydt_train_NN.scatter(x=x_true[0], y=y_true[0], z=z_true[0], s=100, marker='*', color=color_gray)
        # ax_dydt_train_NN.scatter(x=x_learnt[0], y=y_learnt[0], z=z_learnt[0], s=100, marker='*', color=color_red)

        acc_plot.legend(loc='lower right', fontsize=13)

        timestamp = time.time()
        now = time.localtime(timestamp)
        month = now.tm_mon
        day = now.tm_mday

        # Figure show
        fig.tight_layout()
        plt.savefig('png/trained_results{:02d}{:02d}'.format(month, day))
        plt.show()


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    optimizer = optim.Adam(func.parameters(), lr=1e-3)

    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    # Training
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_PVA0, batch_PVA_pre, batch_t = get_batch()
        NN_PVA_pre = odeint(func, batch_PVA0, batch_t).to(device)
        loss = torch.mean(torch.abs(NN_PVA_pre - batch_PVA_pre))
        if itr % args.test_freq == 0:
            pre = torch.zeros([case_num, data_size, variable_num])
            for iii in range(case_num):
                pre[iii] = odeint(func, PVA0[iii, :], t)
            total_loss = torch.mean(torch.abs(pre[:, :, :6] - PVA_truth[:, :, :6]))
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, total_loss.item()))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        end = time.time()
    torch.save(func, 'trained_model/Neural_ODE_ite{:0004d}_loss{:.6f}.pt'.format(args.niters, total_loss.item()))

    # Learnt performance on a trained trajectory
    trajectory_choose = 2
    pre = odeint(func, PVA0[trajectory_choose, :], t)
    Acc_NN = func(t, PVA_truth)
    visualize(PVA_truth[trajectory_choose, :, :], pre, Acc_NN[trajectory_choose, :, :], Acc_truth[trajectory_choose, :, :])

    # python training.py --viz


