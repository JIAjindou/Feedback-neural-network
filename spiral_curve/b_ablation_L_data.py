import os
import argparse
import time
import numpy as np

import math

import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Feedback neural networks - Ablation study on feedback gain and uncertainty degree

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
parser.add_argument('--prediction_step', type=int, default=50)  # Prediction steps
parser.add_argument('--uncer_deg_num', type=int, default=10)
# parser.add_argument('--gain_num', type=int, default=3)
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
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)  # Parameter of training function


# Training function
class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y, true_A)


# True observations in training and testing
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


# Constructing the mini-bach dataset for training
def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D) [20, 1, 2]
    batch_t = t[:args.batch_time]  # (T) [10]
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D) [10, 20, 1, 2]
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
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1
        end = time.time()

    '''--------------------------------Performance Test----------------------------------'''
    # Initialization parameters
    sample_time = args.end_time / args.data_size  # Sample time
    prediction_time = args.prediction_step * sample_time  # Multi-steps prediction time

    # 1) parameters - Neural ODE
    pre_test_NN = torch.zeros(args.data_size - args.prediction_step, 1, 2)  # Prediction results

    # 2) parameters - Feedback neural network
    decay_rate = 0.02
    y_hat = torch.tensor([[args.start_point, 0.]])  # Estimated observation
    temp = odeint(func, y_hat, t[:args.prediction_step + 1])
    y_hat_N = temp[1:, :]  # Initialize estimated observation in multi-steps prediction with Neural ODE
    Error_ablation_FNN = np.zeros(args.uncer_deg_num)

    '''Ablation study on feedback gain and uncertainty degree'''
    dydt_test_true = torch.zeros(args.data_size, 1, 2)  # Real dydt on testing set
    true_y_test = torch.zeros(args.data_size, 1, 2)  # Real observation on testing set
    true_y_test[0, :, :] = true_y0

    for i in range(args.uncer_deg_num):  # Test on each uncertainty
        # True observation in testing set
        A_decay = 0.04 + i * 0.02
        A_period = 0.8 + i * 0.4
        # A_decay = 0.16 + i * 0.02
        # A_period = 3.2 + i * 0.4
        true_A_test = torch.tensor([[-A_decay, A_period], [-A_period, -A_decay]])
        add_dis = -24. + i * 8.
        # add_dis = 32 + i * 8.
        # true_A_test = true_A + delta_A
        for j in range(args.data_size):
            dydt_test_true[j, :, :] = torch.mm(true_y_test[j, :, :], true_A_test) + add_dis
            # Predict next state by 4-order Runge-kutta
            k1 = torch.mm(true_y_test[j, :, :], true_A_test) + add_dis
            k2 = torch.mm(true_y_test[j, :, :] + sample_time * k1 / 2, true_A_test) + add_dis
            k3 = torch.mm(true_y_test[j, :, :] + sample_time * k2 / 2, true_A_test) + add_dis
            k4 = torch.mm(true_y_test[j, :, :] + sample_time * k3, true_A_test) + add_dis
            if j < args.data_size - 1:
                true_y_test[j + 1, :, :] = true_y_test[j, :, :] + (k1 + 2 * k2 + 2 * k3 + k4) * sample_time / 6

        # Sore real observation in multi-steps prediction
        x_true = true_y_test.cpu().numpy()[args.prediction_step:, 0, 0]
        y_true = true_y_test.cpu().numpy()[args.prediction_step:, 0, 1]

        # Multi-steps prediction of feedback neural network
        last_output = torch.zeros(1, 2)  # Output of last layer prediction of feedback neural network
        # print('Ablation number: uncer-{:02d}'.format(i))
        gain = 5  # (0, 45, 5)
        pre_test_FNN = torch.zeros(args.data_size - args.prediction_step, 1, 2)  # Store prediction results
        for j in range(args.data_size - args.prediction_step):
            for k in range(args.prediction_step):
                if k == 0:
                    input_N = true_y_test[j, :, :]
                else:
                    input_N = last_output
                # L decays as the prediction depth increases
                L_decay = torch.tensor([[gain, 0.], [0., gain]]) * math.exp(
                    -k * decay_rate)
                # Predict next state by 4-order Runge-kutta
                k1 = func(0, input_N) + torch.mm(input_N - y_hat_N[k, :, :], L_decay)
                k2 = func(0, input_N + sample_time * k1 / 2) + torch.mm(input_N - y_hat_N[k, :, :], L_decay)
                k3 = func(0, input_N + sample_time * k2 / 2) + torch.mm(input_N - y_hat_N[k, :, :], L_decay)
                k4 = func(0, input_N + sample_time * k3) + torch.mm(input_N - y_hat_N[k, :, :], L_decay)
                y_hat_N[k, :, :] = y_hat_N[k, :, :] + (k1 + 2 * k2 + 2 * k3 + k4) * sample_time / 6
                last_output = input_N + (k1 + 2 * k2 + 2 * k3 + k4) * sample_time / 6
            pre_test_FNN[j, :, :] = last_output

        # Calculate the prediction error at each moment
        x_test_FNN = pre_test_FNN.detach().numpy()[:, 0, 0]
        y_tes_FNN = pre_test_FNN.detach().numpy()[:, 0, 1]
        Error_FNN = np.sqrt((x_test_FNN - x_true) ** 2 + (y_tes_FNN - y_true) ** 2)
        # Evaluate the prediction on the last 5s
        Error_ablation_FNN[i] = np.average(Error_FNN[(len(Error_FNN) - int(10 / sample_time)):])
        print('L {:02d} | Uncertainty {:02d} | error: {:.6f}'.format(gain, i, Error_ablation_FNN[i]))
    print(Error_ablation_FNN)


    # python b_ablation_L_data.py

