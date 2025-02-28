import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams['font.family'] = 'Calibri'

# Global parameters
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=100)  # Maximum number of iterations
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')

parser.add_argument('--start_point', type=float, default=9.)
parser.add_argument('--end_time', type=int, default=20)
parser.add_argument('--prediction_step', type=int, default=50)  # Prediction steps
parser.add_argument('--case_num', type=int, default=10)  # The number of cases through domain randomization
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
true_A_norminal = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)  # Parameter of training function


# Training function
class Lambda_norminal(nn.Module):
    def forward(self, t, y):
        return torch.mm(y, true_A_norminal)


# Training functions - domain randomization
class Lambda_train(nn.Module):
    def forward(self, t, y):
        return torch.mm(y, true_A_domran) + add_dis


# Observations in training
with torch.no_grad():
    true_y_norminal = odeint(Lambda_norminal(), true_y0, t, method='dopri5').to(
        device)  # Observations on norminal function
    true_y_domran = torch.zeros(args.case_num, args.data_size, 1, 2).to(device)  # Observations on multiple functions
    true_dydt_domran = torch.zeros(args.case_num, args.data_size, 2).to(device)
    for i in range(args.case_num):
        # A_decay = 0.1 + (i - 10) * 0.005
        # A_period = 2.0 + (i - 10) * 0.1
        A_decay = 0.04 + i * 0.02
        A_period = 0.8 + i * 0.4
        true_A_domran = torch.tensor([[-A_decay, A_period], [-A_period, -A_decay]]).to(device)
        add_dis = -24. + i * 8.
        temp = odeint(Lambda_train(), true_y0, t, method='dopri5').to(device)
        true_y_domran[i] = temp
        true_dydt_domran[i] = torch.mm(temp.reshape(-1, 2), true_A_domran) + add_dis  # [case_num, data_size, 2]


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


# Visualize the testing performance
def visualize(dydt_true_norminal, dydt_learn_norminal, gain_x_sequence, gain_y_sequence, Total_loss_sequence):
    if args.viz:
        makedirs('png')

        fig = plt.figure(figsize=(12, 4), facecolor='white')

        ax_dydt_train_NN = fig.add_subplot(131, frameon=True)
        ax_gain = fig.add_subplot(132, frameon=True)
        ax_loss = fig.add_subplot(133, frameon=True)

        color_gray = (102 / 255, 102 / 255, 102 / 255)
        color_blue = (76 / 255, 147 / 255, 173 / 255)
        color_red = (1, 0, 0)

        # Figure-1: Training performance of Neural ODE
        ax_dydt_train_NN.cla()
        ax_dydt_train_NN.set_title('Trained performance', fontsize=15, pad=10)
        ax_dydt_train_NN.set_xlabel('$\dot{x}$', fontsize=15)
        ax_dydt_train_NN.set_ylabel('$\dot{y}$', fontsize=15)
        ax_dydt_train_NN.tick_params(axis='x', labelsize=14)
        ax_dydt_train_NN.tick_params(axis='y', labelsize=14)
        ax_dydt_train_NN.plot(dydt_true_norminal.detach().numpy()[:, 0], dydt_true_norminal.detach().numpy()[:, 1],
                              '--', color=color_gray, linewidth=1.5, label='Truth')
        ax_dydt_train_NN.plot(dydt_learn_norminal.cpu().detach().numpy()[:, :, 0],
                              dydt_learn_norminal.cpu().detach().numpy()[:, :, 1], 'b-', linewidth=1.5,
                              label='Neural ODE')

        ax_dydt_train_NN.legend(loc='lower right', fontsize=13)
        ax_dydt_train_NN.set_aspect('equal', adjustable='box')

        # Figure-2: Gain
        ax_gain.cla()
        ax_gain.set_title('Gain variation', fontsize=15)
        ax_gain.set_xlabel('Iteration', fontsize=15)
        ax_gain.set_ylabel('Gain', fontsize=15)
        N_axis = [1 + i * 1 for i in range(args.niters)]
        ax_gain.plot(N_axis, gain_x_sequence, linewidth=1.5, label='Gain x')
        ax_gain.plot(N_axis, gain_y_sequence, linewidth=1.5, label='Gain y')
        ax_gain.set_xlim(0, args.niters)
        # ax_gain.set_ylim(-1, 1)
        ax_gain.tick_params(axis='x', labelsize=14)
        ax_gain.tick_params(axis='y', labelsize=14)
        ax_gain.grid(True)
        ax_gain.legend(loc='lower right', fontsize=13)

        # Figure-3: Loss
        ax_loss.cla()
        ax_loss.set_title('Loss variation', fontsize=15)
        ax_loss.set_xlabel('Iteration', fontsize=15)
        ax_loss.set_ylabel('Loss', fontsize=15)
        N_axis = [0 + i * 1 for i in range(args.niters)]
        ax_loss.plot(N_axis, Total_loss_sequence, linewidth=1.5, label='Total_loss')
        ax_loss.set_xlim(0, args.niters)
        # ax_gain.set_ylim(-1, 1)
        ax_loss.tick_params(axis='x', labelsize=14)
        ax_loss.tick_params(axis='y', labelsize=14)
        ax_loss.grid(True)
        ax_loss.legend(loc='lower right', fontsize=13)

        timestamp = time.time()
        now = time.localtime(timestamp)
        month = now.tm_mon
        day = now.tm_mday

        # Figure show
        fig.tight_layout()
        plt.savefig('png/spiral_L_learn{:02d}{:02d}'.format(month, day))
        plt.show()


if __name__ == '__main__':
    # load trained_model
    func = torch.load('spiral_model.pt', weights_only=False).to(device)

    '''Test the load model'''
    # 1) True dydt of
    Lambda_norminal = Lambda_norminal().to(device)
    dydt_true_norminal = Lambda_norminal(0, true_y_norminal.reshape(-1, 2)).to(device)
    # 2) Learned dydt of Neural ODE
    dydt_learn_norminal = func(0, true_y_norminal).to(device)

    '''Leanring L through domain randomization'''

    # Initialization parameters
    itr = 0
    gain_x, gain_y = 1., 10.  # Initialization gain
    alpha, miu = 20, 0.5  # alpha: updata step, miu: hyperparameter
    dydt_NN = func(0, true_y_domran)  # [case_num, data_size, 1, 2] output dydt from network
    sample_time = args.end_time / args.data_size  # Sample time
    pre_y_domran = torch.zeros(args.case_num, args.data_size - 1, 1, 2).to(device)  # Store one-step prediction
    pre_yhat_domran = torch.zeros(args.case_num, args.data_size - 1, 1, 2).to(device)  # Store y_hat
    pre_dydt_domran = torch.zeros(args.case_num, args.data_size - 1, 1, 2).to(device)  # Store dydt
    stable_index = 1000

    gain_x_sequence = np.zeros(args.niters)
    gain_y_sequence = np.zeros(args.niters)
    Total_loss_sequence = np.zeros(args.niters)

    for itr in range(1, args.niters + 1):
        # One-step prediction by FNN
        y_hat = torch.tensor([[args.start_point, 0.]])  # Estimated observation
        L = torch.tensor([[gain_x, 0.], [0., gain_y]])  # Feedback gain
        gain_x_sequence[itr - 1] = gain_x
        gain_y_sequence[itr - 1] = gain_y

        for i in range(args.case_num):
            for j in range(args.data_size - 1):
                noise = np.random.randn(2) * 0.1
                y0_test = true_y_domran[i, j, :, :]  # Initial value at each moment
                dydt_hat = dydt_NN[i, j, :, :] + torch.mm((y0_test.float() - y_hat), L)  # Correct dydt by feedback
                y_hat = y_hat + dydt_hat * sample_time
                pre_dydt_domran[i, j, :, :] = dydt_hat  # Store the correct dydt
                pre_y_domran[i, j, :, :] = y0_test + dydt_hat * sample_time  # Predict next state
                pre_yhat_domran[i, j, :, :] = y_hat

        Total_loss = torch.mean(torch.abs(pre_y_domran[:, stable_index:, :, :]
                                          - true_y_domran[:, stable_index + 1:, :, :]))
        Total_loss_sequence[itr - 1] = Total_loss

        # Obtain mini-batch training data
        s = torch.from_numpy(
            np.random.choice(np.arange(stable_index, args.data_size - 1, dtype=np.int64), args.batch_size,
                             replace=True))
        batch_y0 = true_y_domran[:, s, :, :]  # [case_num, s, 1, 2]
        batch_y_true = true_y_domran[:, s + 1, :, :]  # [case_num, s, 1, 2]
        batch_y_pre = pre_y_domran[:, s, :, :]  # [case_num, s, 1, 2]
        batch_yhat_last = pre_yhat_domran[:, s, :, :]  # [case_num, s, 1, 2]

        # Calculate loss on mini-batch training data
        Local_loss = torch.mean(torch.abs(batch_y_true - batch_y_pre))
        if 1:
            print('Iter {:04d} | Local Loss {:.6f}'.format(itr, Local_loss.item()))
            print('Iter {:04d} | Total Loss {:.6f} | gain_x {:.2f} | gain_y {:.2f}'
                  .format(itr, Total_loss.item(), gain_x, gain_y))

        # Updata gain through gradient descent
        prod_x = ((batch_y_true[:, :, :, 0] - batch_y_pre[:, :, :, 0]) *
                  (batch_y0[:, :, :, 0] - batch_yhat_last[:, :, :, 0]))
        prod_y = ((batch_y_true[:, :, :, 1] - batch_y_pre[:, :, :, 1]) *
                  (batch_y0[:, :, :, 1] - batch_yhat_last[:, :, :, 1]))
        sum_x = torch.sum(prod_x)
        sum_y = torch.sum(prod_y)
        gradient_lx = - sum_x * sample_time - miu / (gain_x - 1 / 2)
        gradient_ly = - sum_y * sample_time - miu / (gain_y - 1 / 2)
        # gain_x = gain_x - alpha * gradient_lx
        # gain_y = gain_y - alpha * gradient_ly
        gain_x = - miu / sum_x / sample_time
        gain_y = - miu / sum_y / sample_time

    visualize(dydt_true_norminal, dydt_learn_norminal, gain_x_sequence, gain_y_sequence, Total_loss_sequence)


