import os
import argparse
import time
import numpy as np

import math

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
parser.add_argument('--feedback_gain', type=float, default=20.)
parser.add_argument('--end_time', type=int, default=20)
parser.add_argument('--prediction_step', type=int, default=50)  # Prediction steps
parser.add_argument('--add_dis', type=int, default=10)  # Additive disturbance
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
true_A_test = torch.tensor([[-0.05, 3], [-3, -0.05]]).to(device)  # Parameter of testing function


# Training function
class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y, true_A)


# Testing function
class Lambda_test(nn.Module):
    def forward(self, t, y):
        zz = torch.mm(y, true_A_test) + args.add_dis
        return zz


# True observations in training and testing
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    true_y_test = odeint(Lambda_test(), true_y0, t, method='dopri5')


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


# Visualize the testing performance
def visualize(true_y, true_y_test, pre_test_NN, pred_y_N_feedback, pre_test_FNN_nodecay, dydt_train_true,
              dydt_test_true, dydt_test_NN, dydt_feedback, Error_diffN_NN, Error_diffN_FNN, odefunc):
    if args.viz:
        makedirs('png')

        fig = plt.figure(figsize=(14, 10), facecolor='white')
        ax_curve_train = fig.add_subplot(331, frameon=False)
        ax_curve_test = fig.add_subplot(333, frameon=False)

        ax_dydt_train_NN = fig.add_subplot(334, frameon=True)
        ax_dydt_test_NN = fig.add_subplot(335, frameon=True)
        ax_dydt_test_FNN = fig.add_subplot(336, frameon=True)

        ax_stepN_test = fig.add_subplot(337, frameon=True)
        ax_stepN_test_decay = fig.add_subplot(338, frameon=True)
        ax_diff_N_test = fig.add_subplot(339, frameon=True)

        color_gray = (102 / 255, 102 / 255, 102 / 255)
        color_blue = (76 / 255, 147 / 255, 173 / 255)
        color_red = (1, 0, 0)

        # Figure-1: Training trajectory
        ax_curve_train.cla()
        ax_curve_train.set_title('(a) The training set', fontsize=17)
        ax_curve_train.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'k-', linewidth=2)
        ax_curve_train.plot(0, 0, 'ro', label='Origin')
        temp_x = (abs(true_y_test.cpu().numpy()[:, 0, 0].min()) + true_y_test.cpu().numpy()[:, 0, 0].max()) / 2
        temp_y = (abs(true_y_test.cpu().numpy()[:, 0, 1].min()) + true_y_test.cpu().numpy()[:, 0, 1].max()) / 2
        ax_curve_train.set_xlim(-temp_x - 1, temp_x + 1)
        ax_curve_train.set_ylim(-temp_y - 1, temp_y + 1)
        ax_curve_train.set_xticks([])
        ax_curve_train.set_yticks([])
        ax_curve_train.set_aspect('equal', adjustable='box')
        ax_curve_train.legend(loc='lower right', fontsize=15)

        # Figure-3: Testing trajectory
        ax_curve_test.cla()
        ax_curve_test.set_title('(b) The test set', fontsize=17)

        ax_curve_test.plot(true_y_test.cpu().numpy()[:, 0, 0], true_y_test.cpu().numpy()[:, 0, 1], linewidth=2)
        ax_curve_test.plot(0, 0, 'ro', label='Origin')
        ax_curve_test.set_xlim(true_y_test.cpu().numpy()[:, 0, 0].min() - 1,
                               true_y_test.cpu().numpy()[:, 0, 0].max() + 1)
        ax_curve_test.set_ylim(true_y_test.cpu().numpy()[:, 0, 1].min() - 1,
                               true_y_test.cpu().numpy()[:, 0, 1].max() + 1)
        ax_curve_test.set_xticks([])
        ax_curve_test.set_yticks([])
        ax_curve_test.set_aspect('equal', adjustable='box')
        ax_curve_test.legend(loc='lower right', fontsize=15)

        # Figure-4: Training performance of Neural ODE
        ax_dydt_train_NN.cla()
        ax_dydt_train_NN.set_title('(c) Trained performance', fontsize=17, pad=10)
        dydt_NN_train = odefunc(0, true_y).cpu().detach().numpy()
        ax_dydt_train_NN.set_xlabel('$\dot{x}$', fontsize=17)
        ax_dydt_train_NN.set_ylabel('$\dot{y}$', fontsize=17)
        ax_dydt_train_NN.tick_params(axis='x', labelsize=16)
        ax_dydt_train_NN.tick_params(axis='y', labelsize=16)

        x_true = dydt_train_true.detach().numpy()[:, 0]
        y_true = dydt_train_true.detach().numpy()[:, 1]
        x_test = dydt_NN_train[:, 0, 0]
        y_test = dydt_NN_train[:, 0, 1]
        Error_xy = np.sqrt((x_test - x_true) ** 2 + (y_test - y_true) ** 2)

        ax_dydt_train_NN.plot(x_true, y_true, '--', color=color_gray, linewidth=1.5, label='Truth')
        ax_dydt_train_NN.scatter(x=x_true[0], y=y_true[0], s=100, marker='*', color=color_gray)
        ax_dydt_train_NN.scatter(x=x_test[0], y=y_test[0], s=100, marker='*', color=color_red)

        points = np.array([x_test, y_test]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, 22)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(Error_xy)
        lc.set_linewidth(2)
        line = ax_dydt_train_NN.add_collection(lc)
        cbar = fig.colorbar(line)
        cbar.ax.tick_params(labelsize=14)
        ax_dydt_train_NN.legend(loc='lower right', fontsize=15)
        ax_dydt_train_NN.set_aspect('equal', adjustable='box')

        # Figure-5: Testing performance of Neural ODE
        ax_dydt_test_NN.cla()
        ax_dydt_test_NN.set_title('(d) Testing performance of Neural ODE', fontsize=17, pad=10)
        ax_dydt_test_NN.set_xlabel('$\dot{x}$', fontsize=17)
        ax_dydt_test_NN.set_ylabel('$\dot{y}$', fontsize=17)
        ax_dydt_test_NN.tick_params(axis='x', labelsize=14)
        ax_dydt_test_NN.tick_params(axis='y', labelsize=14)

        x_true = dydt_test_true.detach().numpy()[:args.data_size - 1, 0]
        y_true = dydt_test_true.detach().numpy()[:args.data_size - 1, 1]
        x_test = dydt_test_NN.detach().numpy()[:args.data_size - 1, 0, 0]
        y_test = dydt_test_NN.detach().numpy()[:args.data_size - 1, 0, 1]
        Error_xy = np.sqrt((x_test - x_true) ** 2 + (y_test - y_true) ** 2)

        ax_dydt_test_NN.plot(x_true, y_true, '--', color=color_gray, linewidth=1.5, label='Truth')
        ax_dydt_test_NN.scatter(x=x_true[0], y=y_true[0], s=100, marker='*', color=color_gray)
        ax_dydt_test_NN.scatter(x=x_test[0], y=y_test[0], s=100, marker='*', color=color_red)

        points = np.array([x_test, y_test]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, 22)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(Error_xy)
        lc.set_linewidth(2)
        line = ax_dydt_test_NN.add_collection(lc)
        cbar = fig.colorbar(line)
        cbar.ax.tick_params(labelsize=14)
        ax_dydt_test_NN.legend(loc='lower right', fontsize=15)
        ax_dydt_test_NN.set_aspect('equal', adjustable='box')

        # Figure-6: Testing performance of Feedback NN
        ax_dydt_test_FNN.cla()
        ax_dydt_test_FNN.set_title('(e) Testing performance of Feedback NN', fontsize=17, pad=10)
        ax_dydt_test_FNN.set_xlabel('$\dot{x}$', fontsize=17)
        ax_dydt_test_FNN.set_ylabel('$\dot{y}$', fontsize=17)
        ax_dydt_test_FNN.tick_params(axis='x', labelsize=14)
        ax_dydt_test_FNN.tick_params(axis='y', labelsize=14)

        x_true = dydt_test_true.detach().numpy()[:args.data_size - 1, 0]
        y_true = dydt_test_true.detach().numpy()[:args.data_size - 1, 1]
        x_test = dydt_feedback.detach().numpy()[:, 0, 0]
        y_test = dydt_feedback.detach().numpy()[:, 0, 1]
        Error_xy = np.sqrt((x_test - x_true) ** 2 + (y_test - y_true) ** 2)

        ax_dydt_test_FNN.plot(x_true, y_true, '--', color=color_gray, linewidth=1.5, label='Truth')
        ax_dydt_test_FNN.scatter(x=x_true[0], y=y_true[0], s=100, marker='*', color=color_gray)
        ax_dydt_test_FNN.scatter(x=x_test[0], y=y_test[0], s=100, marker='*', color=color_red)

        points = np.array([x_test, y_test]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, 22)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(Error_xy)
        lc.set_linewidth(2)
        line = ax_dydt_test_FNN.add_collection(lc)
        cbar = fig.colorbar(line)
        cbar.ax.tick_params(labelsize=14)
        ax_dydt_test_FNN.legend(loc='lower right', fontsize=15)
        ax_dydt_test_FNN.set_aspect('equal', adjustable='box')

        # Figure-7: Multi-steps prediction errors in testing
        ax_stepN_test.cla()
        ax_stepN_test.set_title('(f) Multi-step prediction errors in testing', fontsize=17, pad=10)
        ax_stepN_test.set_xlabel('t [s]', fontsize=17)
        ax_stepN_test.set_ylabel('Prediction error', fontsize=17)
        time_f8 = t.cpu().numpy()[args.prediction_step:]

        x_true = true_y_test.cpu().numpy()[args.prediction_step:, 0, 0]
        y_true = true_y_test.cpu().numpy()[args.prediction_step:, 0, 1]
        x_test_NN = pre_test_NN.detach().numpy()[:, 0, 0]
        y_test_NN = pre_test_NN.detach().numpy()[:, 0, 1]
        x_test_FNN = pred_y_N_feedback.detach().numpy()[:, 0, 0]
        y_test_FNN = pred_y_N_feedback.detach().numpy()[:, 0, 1]
        Error_NN = np.sqrt((x_test_NN - x_true) ** 2 + (y_test_NN - y_true) ** 2)
        Error_FNN = np.sqrt((x_test_FNN - x_true) ** 2 + (y_test_FNN - y_true) ** 2)

        ax_stepN_test.plot(time_f8, Error_NN, linewidth=1.5, label='Neural ODE')
        ax_stepN_test.plot(time_f8, Error_FNN, 'r-', linewidth=1.5, label='Feedback NN')

        ax_stepN_test.set_xlim(time_f8.min(), time_f8.max())
        ax_stepN_test.set_ylim(-2, 12)
        ax_stepN_test.legend(loc='lower right', fontsize=15)
        ax_stepN_test.tick_params(axis='x', labelsize=14)
        ax_stepN_test.tick_params(axis='y', labelsize=14)
        ax_stepN_test.grid(True)

        # Figure-8: Performance with decay strategy
        ax_stepN_test_decay.cla()
        ax_stepN_test_decay.set_title('(g) Performance with decay strategy', fontsize=17, pad=10)
        ax_stepN_test_decay.set_xlabel('t [s]', fontsize=17)
        ax_stepN_test_decay.set_ylabel('x', fontsize=17)
        time_f3 = t.cpu().numpy()[args.prediction_step:]
        ax_stepN_test_decay.plot(time_f3, true_y_test.cpu().numpy()[args.prediction_step:, 0, 0], color=color_gray,
                                 linewidth=1.5,
                                 label='Truth')
        ax_stepN_test_decay.plot(time_f3, pre_test_FNN_nodecay.detach().numpy()[:, 0, 0], '--', linewidth=1.5,
                                 label='Without decay')
        ax_stepN_test_decay.plot(time_f3, pred_y_N_feedback.detach().numpy()[:, 0, 0], 'r--',
                                 linewidth=1.5, label='With decay')
        ax_stepN_test_decay.set_xlim(time_f3.min(), time_f3.max())
        ax_stepN_test_decay.set_ylim(-args.start_point - args.add_dis - 10, args.start_point + args.add_dis + 2)
        ax_stepN_test_decay.legend(loc='lower right', fontsize=15)
        ax_stepN_test_decay.tick_params(axis='x', labelsize=14)
        ax_stepN_test_decay.tick_params(axis='y', labelsize=14)
        ax_stepN_test_decay.grid(True)

        # Figure-9: Performance with different prediction steps
        ax_diff_N_test.cla()
        ax_diff_N_test.set_title('(f) Performance with different N', fontsize=17)
        ax_diff_N_test.set_xlabel('N', fontsize=17)
        ax_diff_N_test.set_ylabel('Prediction error', fontsize=17)
        N_axis = [0 + i * 1 for i in range(len(Error_diffN_FNN))]
        ax_diff_N_test.plot(N_axis, Error_diffN_NN, linewidth=1.5, label='Neural ODE')
        ax_diff_N_test.plot(N_axis, Error_diffN_FNN, 'r-', linewidth=1.5, label='Feedback NN')
        ax_diff_N_test.set_xlim(0, len(Error_diffN_FNN) - 1)
        ax_diff_N_test.set_ylim(-1, Error_diffN_NN.max() + 1)
        ax_diff_N_test.tick_params(axis='x', labelsize=14)
        ax_diff_N_test.tick_params(axis='y', labelsize=14)
        ax_diff_N_test.grid(True)
        ax_diff_N_test.legend(loc='lower right', fontsize=15)

        timestamp = time.time()
        now = time.localtime(timestamp)
        month = now.tm_mon
        day = now.tm_mday

        # Figure show
        fig.tight_layout()
        plt.savefig('png/a_multi_steps_pre{:02d}{:02d}'.format(month, day))
        plt.show()


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
    bias = 31

    # 1) parameters - Neural ODE
    pre_test_NN = torch.zeros(args.data_size - args.prediction_step, 1, 2)  # Prediction results
    Error_diffN_NN = np.zeros(args.prediction_step + bias)  # Prediction errors with different prediction steps N

    # 2) parameters - Feedback neural network
    dydt_test_FNN = torch.zeros(args.data_size - 1, 1, 2)  # Learned dydt of testing set
    L = torch.tensor([[args.feedback_gain, 0.], [0., args.feedback_gain]])  # Feedback gain
    decay_rate = 0.02
    y_hat = torch.tensor([[args.start_point, 0.]])  # Estimated observation
    temp = odeint(func, y_hat, t[:args.prediction_step + 1])
    y_hat_N = temp[1:, :]  # Initialize estimated observation in multi-steps prediction with Neural ODE
    pre_test_FNN = torch.zeros(args.data_size - args.prediction_step, 1, 2)  # Prediction results
    pre_test_FNN_nodecay = torch.zeros(args.data_size - args.prediction_step, 1, 2)  # Without L decay strategy
    Error_diffN_FNN = np.zeros(args.prediction_step + bias)  # Prediction errors with different prediction steps N

    '''dydt Performance'''

    # 1) True dydt of training set
    Lambda_train = Lambda().to(device)
    dydt_train_true = Lambda_train(0, true_y.reshape(-1, 2))
    # 2) True dydt of testing set
    Lambda_fun = Lambda_test().to(device)
    dydt_test_true = Lambda_fun(0, true_y_test.reshape(-1, 2))
    # 3) Learned dydt of Neural ODE in testing set
    dydt_test_NN = func(0, true_y_test)
    # 4) Learned dydt of feedback neural network in testing set
    for kk in range(args.data_size - 1):
        y0_test = true_y_test[kk, :, :]  # Initial value at each moment
        dydt_hat = dydt_test_NN[kk, :, :] + torch.mm((y0_test - y_hat), L)  # correct dydt by feedback
        k1 = func(0, y0_test) + torch.mm(y0_test - y_hat, L)
        y_hat_new = y_hat + k1 * sample_time
        y_hat = y_hat_new
        dydt_test_FNN[kk, :, :] = dydt_hat  # store the learned dydt of testing set

    '''Multi-steps Prediction Performance'''

    last_output = torch.zeros(1, 2)  # Output of last layer prediction of feedback neural network
    for jj in range(args.data_size - args.prediction_step):
        # 1) Feedback neural network
        for ii in range(args.prediction_step):
            if ii == 0:
                input_N = true_y_test[jj, :, :]
            else:
                input_N = last_output
            # L decays as the prediction depth increases
            L_decay = torch.tensor([[args.feedback_gain, 0.], [0., args.feedback_gain]]) * math.exp(-ii * decay_rate)
            k1 = func(0, input_N) + torch.mm(input_N - y_hat_N[ii, :, :], L_decay)
            y_hat_N[ii, :, :] = y_hat_N[ii, :, :] + k1 * sample_time
            last_output = input_N + k1 * sample_time
        pre_test_FNN[jj, :, :] = last_output

        # 2) Neural ODE
        y0_test = true_y_test[jj, :, :]
        temp = odeint(func, y0_test, t[:args.prediction_step + 1])
        pre_test_NN[jj, :, :] = temp[args.prediction_step, :, :]

    '''Gain Decay Performance'''

    temp = odeint(func, y_hat, t[:args.prediction_step + 1])
    y_hat_N = temp[1:, :]  # Initialize estimated observation in multi-steps prediction with Neural ODE
    for jj in range(args.data_size - args.prediction_step):
        for ii in range(args.prediction_step):
            if ii == 0:
                input_N = true_y_test[jj, :, :]
            else:
                input_N = last_output
            k1 = func(0, input_N) + torch.mm(input_N - y_hat_N[ii, :, :], L)
            y_hat_N[ii, :, :] = y_hat_N[ii, :, :] + k1 * sample_time
            last_output = input_N + k1 * sample_time
        pre_test_FNN_nodecay[jj, :, :] = last_output

    '''Prediction Performance with Different Prediction Steps N'''

    # for kk in range(1, args.prediction_step + bias):
    #     print('Prediction steps {:03d}'.format(kk))
    #     y_hat_N = torch.zeros(kk, 1, 2)
    #     temp_FNN = torch.zeros(args.data_size - kk, 1, 2)
    #     temp_NN = torch.zeros(args.data_size - kk, 1, 2)
    #     for jj in range(args.data_size - kk):
    #         # 1) Feedback neural network
    #         for ii in range(kk):
    #             if ii == 0:
    #                 input_N = true_y_test[jj, :, :]
    #             else:
    #                 input_N = last_output
    #             L_decay = (torch.tensor([[args.feedback_gain, 0.], [0., args.feedback_gain]]) *
    #                        math.exp(-ii * decay_rate))
    #             # Predict next state by 4-order Runge-kutta
    #             k1 = func(0, input_N) + torch.mm(input_N - y_hat_N[ii, :, :], L_decay)
    #             y_hat_N[ii, :, :] = y_hat_N[ii, :, :] + k1 * sample_time
    #             last_output = input_N + k1 * sample_time
    #         temp_FNN[jj, :, :] = last_output
    #
    #         # 2) Neural ODE
    #         temp = odeint(func, true_y_test[jj, :, :], t[:kk + 1])
    #         temp_NN[jj, :, :] = temp[kk, :, :]
    #
    #     # Calculate the prediction error at each moment
    #     x_true = true_y_test.cpu().numpy()[kk:, 0, 0]
    #     y_true = true_y_test.cpu().numpy()[kk:, 0, 1]
    #     x_test_FNN = temp_FNN.detach().numpy()[:, 0, 0]
    #     y_tes_FNN = temp_FNN.detach().numpy()[:, 0, 1]
    #     Error_FNN = np.sqrt((x_test_FNN - x_true) ** 2 + (y_tes_FNN - y_true) ** 2)
    #     x_test_NN = temp_NN.detach().numpy()[:, 0, 0]
    #     y_tes_NN = temp_NN.detach().numpy()[:, 0, 1]
    #     Error_NN = np.sqrt((x_test_NN - x_true) ** 2 + (y_tes_NN - y_true) ** 2)
    #
    #     # Evaluate the prediction on the last 5s
    #     Error_diffN_FNN[kk] = np.average(Error_FNN[(len(Error_FNN) - int(5 / sample_time)):])
    #     Error_diffN_NN[kk] = np.average(Error_NN[(len(Error_NN) - int(5 / sample_time)):])

    # plot
    visualize(true_y, true_y_test, pre_test_NN, pre_test_FNN, pre_test_FNN_nodecay, dydt_train_true,
              dydt_test_true, dydt_test_NN, dydt_test_FNN, Error_diffN_NN, Error_diffN_FNN, func)


    # python a_multi_steps_pre.py --viz
