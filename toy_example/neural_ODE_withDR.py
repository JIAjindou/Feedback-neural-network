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
parser.add_argument('--data_size', type=int, default=2000)
parser.add_argument('--batch_time', type=int, default=10)  # maximum predicted time of mimi_batch
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=400)  # Maximum number of iterations
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')

parser.add_argument('--start_point', type=float, default=9.)
parser.add_argument('--end_time', type=int, default=20)
parser.add_argument('--uncer_deg_num', type=int, default=20)  # Numbers of parameter randomization
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
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)  # Parameter of nominal function
# true_A = torch.tensor([[-0.08, 1.5], [-1.5, -0.08]]).to(device)  # Parameter of nominal function


# Norminal function
class Lambda_nominal(nn.Module):
    def forward(self, t, y):
        return torch.mm(y, true_A)


# Training functions - domain randomization
class Lambda_train(nn.Module):
    def forward(self, t, y):
        return torch.mm(y, true_A_test) + add_dis


# Observations in training and testing
true_y = torch.zeros(args.uncer_deg_num, args.data_size, 1, 2).to(device)  # Observations on training functions
with torch.no_grad():
    true_y_nominal = odeint(Lambda_nominal(), true_y0, t, method='dopri5')   # Observations on nominal function
    if args.uncer_deg_num == 1:
        A_decay = 0.1
        A_period = 2.0
        true_A_test = torch.tensor([[-A_decay, A_period], [-A_period, -A_decay]])
        true_y[0] = odeint(Lambda_train(), true_y0, t, method='dopri5')
    else:
        for i in range(args.uncer_deg_num):
            A_decay = 0.04 + i * 0.005  # [0.04 1 0.14]
            A_period = 0.8 + i * 0.12  # [0.8 2 3.2]
            add_dis = -24. + i * 2.4
            true_A_test = torch.tensor([[-A_decay, A_period], [-A_period, -A_decay]])
            true_y[i] = odeint(Lambda_train(), true_y0, t, method='dopri5')


# Constructing the mini-bach dataset for training
def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    ii = np.random.choice(np.arange(args.uncer_deg_num, dtype=np.int64), 1, replace=True)
    batch_y0 = true_y[ii, s, :, :]  # (M, D) [20, 1, 2]
    batch_t = t[:args.batch_time]  # (T) [10]
    batch_y = torch.stack([true_y[ii, s + i, :, :] for i in range(args.batch_time)], dim=0)  # (T, M, D) [10, 20, 1, 2]
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


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

def visualize(dydt_train_true, dydt_train_NN):
    if args.viz:
        makedirs('png')

        fig = plt.figure(figsize=(5, 5), facecolor='white')

        ax_dydt_train_NN = fig.add_subplot(111, frameon=True)


        color_gray = (102 / 255, 102 / 255, 102 / 255)
        color_blue = (76 / 255, 147 / 255, 173 / 255)
        color_red = (1, 0, 0)

        # Figure-4: Training performance of Neural ODE
        ax_dydt_train_NN.cla()
        ax_dydt_train_NN.set_title('Trained performance with domain randomization', fontsize=15, pad=10)
        ax_dydt_train_NN.set_xlabel('$\dot{x}$', fontsize=15)
        ax_dydt_train_NN.set_ylabel('$\dot{y}$', fontsize=15)
        ax_dydt_train_NN.tick_params(axis='x', labelsize=14)
        ax_dydt_train_NN.tick_params(axis='y', labelsize=14)
        x_true = dydt_train_true.detach().numpy()[:, 0]
        y_true = dydt_train_true.detach().numpy()[:, 1]
        x_test = dydt_train_NN.detach().numpy()[:, 0, 0]
        y_test = dydt_train_NN.detach().numpy()[:, 0, 1]
        Error_xy = np.sqrt((x_test - x_true) ** 2 + (y_test - y_true) ** 2)
        # print(Error_xy.max())

        ax_dydt_train_NN.plot(x_true, y_true, '--', color=color_gray, linewidth=1.5, label='Truth')
        ax_dydt_train_NN.scatter(x=x_true[0], y=y_true[0], s=100, marker='*', color=color_gray)
        ax_dydt_train_NN.scatter(x=x_test[0], y=y_test[0], s=100, marker='*', color=color_red)

        points = np.array([x_test, y_test]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, 14)

        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(Error_xy)
        lc.set_linewidth(2)
        line = ax_dydt_train_NN.add_collection(lc)
        cbar = fig.colorbar(line, shrink=0.78)
        cbar.ax.tick_params(labelsize=14)
        ax_dydt_train_NN.legend(loc='lower right', fontsize=13)
        ax_dydt_train_NN.set_aspect('equal', adjustable='box')

        timestamp = time.time()
        now = time.localtime(timestamp)
        month = now.tm_mon
        day = now.tm_mday

        # Figure show
        fig.tight_layout()
        plt.savefig('png/neural_ODE_withDR{:02d}{:02d}'.format(month, day))
        plt.show()


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    # Training
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Training with DR | Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1
        end = time.time()
    # torch.save(func, 'trained_model/Neural_ODE_domran.pt')

    # 1) True dydt on nominal function
    Lambda_nominal = Lambda_nominal().to(device)
    dydt_train_true = Lambda_nominal(0, true_y_nominal.reshape(-1, 2))
    # 2) Learned dydt of Neural ODE on nominal function
    dydt_train_NN = func(0, true_y_nominal)
    visualize(dydt_train_true, dydt_train_NN)

    # python neural_ODE_withDR.py --viz

