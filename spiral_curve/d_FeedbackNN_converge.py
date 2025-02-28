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
parser.add_argument('--niters', type=int, default=2000)  # Maximum number of iterations
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
true_A_nominal = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)  # Parameter of training function


# true_A_nominal = torch.tensor([[-0.12, 2.4], [-2.4, -0.12]]).to(device)  # Parameter of training function

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
    true_y_domran = torch.zeros(args.case_num, args.data_size, 1, 2).to(
        device)  # Observations on multiple functions
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
            nn.Linear(2, 100),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
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


# Visualize the testing performance
def visualize(dydt_true_nominal, dydt_ODEnn_test, dydt_FNN_test, Total_loss_sequence, count):
    if args.viz:
        makedirs('png')

        fig = plt.figure(figsize=(8, 4), facecolor='white')

        ax_dydt_FNN_test = fig.add_subplot(121, frameon=True)
        ax_loss = fig.add_subplot(122, frameon=True)

        color_gray = (102 / 255, 102 / 255, 102 / 255)
        color_blue = (76 / 255, 147 / 255, 173 / 255)
        color_red = (1, 0, 0)

        # Figure-1: Feedback neural networks on nominal task
        ax_dydt_FNN_test.cla()
        ax_dydt_FNN_test.set_title('Training with a neural feedback', fontsize=15, pad=10)
        ax_dydt_FNN_test.set_xlabel('$\dot{x}$', fontsize=15)
        ax_dydt_FNN_test.set_ylabel('$\dot{y}$', fontsize=15)
        ax_dydt_FNN_test.tick_params(axis='x', labelsize=14)
        ax_dydt_FNN_test.tick_params(axis='y', labelsize=14)

        x_true = dydt_true_nominal.detach().numpy()[:args.data_size - 1, 0]
        y_true = dydt_true_nominal.detach().numpy()[:args.data_size - 1, 1]
        x_test = dydt_FNN_test.detach().numpy()[:, 0, 0]
        y_test = dydt_FNN_test.detach().numpy()[:, 0, 1]
        Error_xy = np.sqrt((x_test - x_true) ** 2 + (y_test - y_true) ** 2)

        ax_dydt_FNN_test.plot(x_true, y_true, '--', color=color_gray, linewidth=1.5, label='Truth')
        ax_dydt_FNN_test.scatter(x=x_true[0], y=y_true[0], s=100, marker='*', color=color_gray)
        ax_dydt_FNN_test.scatter(x=x_test[0], y=y_test[0], s=100, marker='*', color=color_red)

        points = np.array([x_test, y_test]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, 14)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(Error_xy)
        lc.set_linewidth(2)
        line = ax_dydt_FNN_test.add_collection(lc)
        cbar = fig.colorbar(line, shrink=0.78)
        cbar.ax.tick_params(labelsize=14)
        ax_dydt_FNN_test.legend(loc='lower right', fontsize=13)
        ax_dydt_FNN_test.set_aspect('equal', adjustable='box')

        # Figure-2: Loss
        ax_loss.cla()
        ax_loss.set_title('Loss variation', fontsize=15)
        ax_loss.set_xlabel('Iteration', fontsize=15)
        ax_loss.set_ylabel('Loss', fontsize=15)
        N_axis = [1 + i * 1 for i in range(count)]
        ax_loss.plot(N_axis, Total_loss_sequence[:count], linewidth=1.5, label='Total_loss')
        ax_loss.set_xlim(1, count)
        ax_loss.set_ylim(-1, 30)
        ax_loss.tick_params(axis='x', labelsize=14)
        ax_loss.tick_params(axis='y', labelsize=14)
        ax_loss.grid(True)
        # ax_loss.legend(loc='upper right', fontsize=13)

        # Obtain current time
        timestamp = time.time()
        now = time.localtime(timestamp)
        month = now.tm_mon
        day = now.tm_mday

        # Figure show
        fig.tight_layout()
        plt.savefig('png/d_FeedbackNN{:02d}{:02d}_ite{:0003d}_conver'.format(month, day, count))
        plt.show()


if __name__ == '__main__':
    # load trained_model
    funcODE = torch.load('trained_model/Neural_ODE_ite400_loss0.2.pt', weights_only=False).to(device)

    '''Leanring feedback neural networks through domain randomization'''

    # Initialization NN
    itr = 0
    funcBack = BackFunc().to(device)
    optimizer = optim.RMSprop(funcBack.parameters(), lr=1e-2)
    # optimizer = optim.Adam(funcBack.parameters(), lr=1e-2)
    # optimizer = optim.SGD(funcBack.parameters(), lr=0.01, momentum=0.8)  # not work
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    # Initialization parameters
    dydt_ODEnn = funcODE(0, true_y_domran)  # [case_num, data_size, 1, 2] output dydt from network
    sample_time = args.end_time / args.data_size  # Sample time
    label_true = (true_y_domran[:, 1:, :, :] - true_y_domran[:, :args.data_size - 1, :, :]
                  - sample_time * dydt_ODEnn[:, :args.data_size - 1, :,
                                  :]) / sample_time  # [case_num, data_size - 1, 1, 2]

    label_pre = torch.zeros(args.case_num, args.data_size - 1, 1, 2).to(device)  # Store output of BackFunc
    pre_yhat = torch.zeros(args.case_num, args.data_size - 1, 1, 2).to(device)  # Store y_hat
    stable_index = 0
    Total_loss_sequence = np.zeros(args.niters)
    count = 0

    # Modifying the initial data to represent the convergence procedure
    converge_time = 0.5
    converge_samples = int(converge_time / sample_time)
    for i in range(converge_samples):
        label_true[:, i, :, :] = label_true[:, i, :, :] * (1 - np.exp(-i * 2))

    # training
    for itr in range(1, args.niters + 1):
        with torch.no_grad():
            for i in range(args.case_num):
                y_hat = torch.tensor([[args.start_point, 0.]])  # Estimated observation
                for j in range(args.data_size - 1):
                    y0_test = true_y_domran[i, j, :, :]  # Initial value at each moment
                    label_pre[i, j, :, :] = funcBack(0, (y0_test - y_hat))
                    pre_yhat[i, j, :, :] = y_hat  # Store y_hat
                    dydt_hat = dydt_ODEnn[i, j, :, :] + funcBack(0, (y0_test - y_hat))  # Correct dydt by feedback
                    y_hat = y_hat + dydt_hat * sample_time
            Feature = true_y_domran[:, :args.data_size - 1, :, :] - pre_yhat  # [case_num, data_size - 1, 1, 2]
            # Obtain mini-batch training data
            s1 = torch.from_numpy(np.random.choice(np.arange(stable_index, args.data_size - 1, dtype=np.int64),
                                                   args.batch_size, replace=False))
            batch_Feature = Feature[:, :, :, :]  # [case_num, s, 1, 2]
            batch_label_true = label_true[:, :, :, :]

            # for j in range(10):
            #     with torch.no_grad():
            #         s1 = torch.from_numpy(np.random.choice(np.arange(stable_index, args.data_size - 1, dtype=np.int64),
            #                                                args.batch_size, replace=False))
            #         batch_Feature = Feature[:, s1, :, :]  # [case_num, s, 1, 2]
            #         batch_label_true = label_true[:, s1, :, :]

        # 前向传播
        optimizer.zero_grad()
        batch_label_NN = funcBack(0, batch_Feature)
        # loss = torch.mean(torch.abs(batch_label_true - batch_label_NN))
        loss = (batch_label_NN - batch_label_true).pow(2).sum()
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        with torch.no_grad():
            count += 1
            Total_loss = torch.mean(torch.abs(label_true - label_pre))
            Total_loss_sequence[itr - 1] = Total_loss
            # print('Iter {:04d} | Local Loss {:.6f}'.format(itr, loss.item()))
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, Total_loss.item()))
            if Total_loss < 0.5:
                break
        end = time.time()
    torch.save(funcBack, 'trained_model/FeedbackNN_ite{:003d}_loss{:.6f}_conver.pt'.format(count, Total_loss.item()))
    # funcBack = torch.load('trained_model/FeedbackNN0901.pt', weights_only=False).to(device)

    '''Test the Neural ODE'''
    # 1) True dydt
    Lambda_nominal = Lambda_nominal().to(device)
    dydt_true_nominal = Lambda_nominal(0, true_y_nominal.reshape(-1, 2)).to(device)

    # 2) Learned dydt of Neural ODE
    dydt_ODEnn_test = funcODE(0, true_y_nominal).to(device)

    '''Test the Feedback Neural ODE'''
    y_hat_test = torch.tensor([[args.start_point, 0.]])  # Estimated observation
    dydt_FNN_test = torch.zeros((args.data_size - 1, 1, 2))
    for j in range(args.data_size - 1):
        y0_test = true_y_nominal[j, :, :]  # Initial value at each moment
        # label_pre_test[j, :, :] = funcBack(0, (y0_test - y_hat))
        dydt_hat = dydt_ODEnn_test[j, :, :] + funcBack(0, (y0_test - y_hat_test))  # Correct dydt by feedback
        y_hat_test = y_hat_test + dydt_hat * sample_time
        dydt_FNN_test[j, :, :] = dydt_hat

    with open("iteration_ite{:003d}_conver.txt".format(count), "w") as file:
        np.set_printoptions(threshold=np.inf)
        file.write(str(Total_loss_sequence))

    visualize(dydt_true_nominal, dydt_ODEnn_test, dydt_FNN_test, Total_loss_sequence, count)

    # python d_FeedbackNN.py --viz

