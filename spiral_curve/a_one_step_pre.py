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

# 分离参数与代码 用于频繁修改参数
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=2000)
parser.add_argument('--batch_time', type=int, default=10)   # mimi_batch 训练时最长预测时间
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=400)      # 最大迭代次数
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')

parser.add_argument('--start_point', type=float, default=9.)
parser.add_argument('--feedback_gain', type=float, default=10)
parser.add_argument('--end_time', type=int, default=20)
parser.add_argument('--prediction_step', type=int, default=2)  # N - Multi-steps prediction
parser.add_argument('--dis', type=int, default=10)  # N - Multi-steps prediction
args = parser.parse_args()

# 选取ODE求解器
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# 检查是否有可用的GPU
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# 初值
true_y0 = torch.tensor([[args.start_point, 0.]]).to(device)
# 观察时刻
t = torch.linspace(0., args.end_time, args.data_size).to(device)
# 原函数参数
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)
# 测试函数参数
true_A_test = torch.tensor([[-0.05, 5], [-5, -0.05]]).to(device)
# true_A_test = true_A


class Lambda(nn.Module):
    # 训练函数
    def forward(self, t, y):
        return torch.mm(y, true_A)

class Lambda_test(nn.Module):
    # 测试函数
    def forward(self, t, y):
        zz = torch.mm(y, true_A_test) + args.dis
        return zz

# 在各个观测时刻的真实值
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    true_y_test = odeint(Lambda_test(), true_y0, t, method='dopri5')


# 创建mini batch训练数据集
def get_batch():
    # 随机从[0, data_size-batch_time]范围内选取batch_size个索引
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D) [20, 1, 2]
    batch_t = t[:args.batch_time]  # (T) [10]
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D) [10, 20, 1, 2]
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


# 创建新目录
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def visualize(true_y, true_y_test, dydt_true_train, dydt_true, dydt_NN, dydt_feedback, Error_NN, Error_FNN, odefunc, itr):
    if args.viz:
        makedirs('png')  # 创建png文件夹，存储中间训练的图像

        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax_dydt_train_NN = fig.add_subplot(221, frameon=True)
        ax_dydt_test_NN = fig.add_subplot(222, frameon=True)
        ax_dydt_test_FNN = fig.add_subplot(223, frameon=True)
        ax_one_step_pre = fig.add_subplot(224, frameon=True)


        color_gray = (102 / 255, 102 / 255, 102 / 255)
        color_blue = (76 / 255, 147 / 255, 173 / 255)
        color_red = (1, 0, 0)

        # Figure-1
        ax_dydt_train_NN.cla()
        ax_dydt_train_NN.set_title('Trained performance of Neural ODE', fontsize=15)
        dydt_NN_train = odefunc(0, true_y).cpu().detach().numpy()
        ax_dydt_train_NN.set_xlabel('dx', fontsize=15)
        ax_dydt_train_NN.set_ylabel('dy', fontsize=15)
        ax_dydt_train_NN.tick_params(axis='x', labelsize=14)
        ax_dydt_train_NN.tick_params(axis='y', labelsize=14)

        x_true = dydt_true_train.detach().numpy()[:, 0]
        y_true = dydt_true_train.detach().numpy()[:, 1]
        x_test = dydt_NN_train[:, 0, 0]
        y_test = dydt_NN_train[:, 0, 1]
        Error_xy = np.sqrt((x_test - x_true) ** 2 + (y_test - y_true) ** 2)

        ax_dydt_train_NN.plot(x_true, y_true, '--', color=color_gray, linewidth=1.5, label='Truth')
        ax_dydt_train_NN.scatter(x=x_true[0], y=y_true[0], s=100, marker='*', color= color_gray)
        ax_dydt_train_NN.scatter(x=x_test[0], y=y_test[0], s=100, marker='*', color= color_red)

        points = np.array([x_test, y_test]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, 22)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(Error_xy)
        lc.set_linewidth(2)
        line =ax_dydt_train_NN.add_collection(lc)
        ax_dydt_train_NN.legend(loc='lower right', fontsize=15)
        ax_dydt_train_NN.set_aspect('equal', adjustable='box')

        # Figure-2
        ax_dydt_test_NN.cla()
        ax_dydt_test_NN.set_title('Testing performance of Neural ODE', fontsize=15)
        ax_dydt_test_NN.set_xlabel('dx', fontsize=15)
        ax_dydt_test_NN.set_ylabel('dy', fontsize=15)
        ax_dydt_test_NN.tick_params(axis='x', labelsize=14)
        ax_dydt_test_NN.tick_params(axis='y', labelsize=14)

        x_true = dydt_true.detach().numpy()[:args.data_size-1, 0]
        y_true = dydt_true.detach().numpy()[:args.data_size-1, 1]
        x_test = dydt_NN.detach().numpy()[:args.data_size-1, 0, 0]
        y_test = dydt_NN.detach().numpy()[:args.data_size-1, 0, 1]
        Error_xy = np.sqrt((x_test - x_true) ** 2 + (y_test - y_true) ** 2)

        ax_dydt_test_NN.plot(x_true, y_true, '--', color=color_gray, linewidth=1.5, label='Truth')
        ax_dydt_test_NN.scatter(x=x_true[0], y=y_true[0], s=100, marker='*', color=color_gray)
        ax_dydt_test_NN.scatter(x=x_test[0], y=y_test[0], s=100,  marker='*', color=color_red)

        points = np.array([x_test, y_test]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0,22)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(Error_xy)
        lc.set_linewidth(2)
        line = ax_dydt_test_NN.add_collection(lc)
        ax_dydt_test_NN.legend(loc='lower right', fontsize=15)
        ax_dydt_test_NN.set_aspect('equal', adjustable='box')

        # Figure-3
        ax_dydt_test_FNN.cla()
        ax_dydt_test_FNN.set_title('Testing performance of Feedback NN', fontsize=15)
        ax_dydt_test_FNN.set_xlabel('$\dot{x}$', fontsize=15)
        ax_dydt_test_FNN.set_ylabel('$\dot{y}$', fontsize=15)
        ax_dydt_test_FNN.tick_params(axis='x', labelsize=14)
        ax_dydt_test_FNN.tick_params(axis='y', labelsize=14)

        x_true = dydt_true.detach().numpy()[:args.data_size-1, 0]
        y_true = dydt_true.detach().numpy()[:args.data_size-1, 1]
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
        cbar_ax = fig.add_axes([0.97, 0.57, 0.01, 0.38])
        cbar = fig.colorbar(line, cax=cbar_ax)
        # cbar.set_label('', fontsize=20)
        cbar.ax.tick_params(labelsize=14)
        ax_dydt_test_FNN.legend(loc='lower right', fontsize=15)
        ax_dydt_test_FNN.set_aspect('equal', adjustable='box')


        # Figure-4
        ax_one_step_pre.cla()
        ax_one_step_pre.set_title('One step prediction', fontsize=15)
        ax_one_step_pre.set_xlabel('N', fontsize=15)
        ax_one_step_pre.set_ylabel('error', fontsize=15)
        time_f3 = t.cpu().numpy()[1:]
        ax_one_step_pre.plot(time_f3, Error_NN, '--', linewidth=1.5, label='Error_NN')
        ax_one_step_pre.plot(time_f3, Error_FNN, '--', linewidth=1.5, label='Error_FNN')
        ax_one_step_pre.legend()

        timestamp = time.time()
        now = time.localtime(timestamp)
        month = now.tm_mon
        day = now.tm_mday

        # Figure show
        fig.tight_layout()
        plt.savefig('png/a_one_step_pre{:02d}{:02d}'.format(month, day))
        plt.show()

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()  # 子类继承了父类的所有属性和方法

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

    # Initial NN
    func = ODEFunc().to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

    end = time.time()  # 获取当前时间戳

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    # Training
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()  # 清空梯度
        batch_y0, batch_t, batch_y = get_batch()  # 获取mini batch训练集
        pred_y = odeint(func, batch_y0, batch_t).to(device)  # 当前网络预测
        loss = torch.mean(torch.abs(pred_y - batch_y))  # loss function
        loss.backward()  # 计算梯度
        optimizer.step()  # 根据计算出来梯度更新参数

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())  # 将单个元素标量转换为标量

        if itr % args.test_freq == 0:
            with torch.no_grad():  # 禁用梯度计算
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1
        end = time.time()

    ##################Performance test#####################
    sample_freq = args.end_time/args.data_size           # sample time
    prediction_time = args.prediction_step * sample_freq # Multi-steps prediction time

    # Initial Neural ODE
    pred_y_1 = torch.zeros(args.data_size - 1, 1, 2)
    pred_y_1_test = torch.zeros(args.data_size - 1, 1, 2)
    dydt_NN = func(0, true_y_test)  # 测试集下神经网络dydt  [1000, 1, 2]

    Lambda_train = Lambda().to(device)
    dydt_true_train = Lambda_train(0, true_y.reshape(-1, 2))

    # Initial Neural ODE Feedback NN
    Lambda_fun = Lambda_test().to(device)
    dydt_true = Lambda_fun(0, true_y_test.reshape(-1, 2))  # 真实值dydt [1000, 2]

    pred_y_1_Feedback = torch.zeros(args.data_size - 1, 1, 2)  # 初始化一步预测值
    dydt_Feedback = torch.zeros(args.data_size - 1, 1, 2)  # 初始化dydt_Feedback
    L = torch.tensor([[args.feedback_gain, 0.], [0., args.feedback_gain]])  # 反馈增益
    current_y_hat = torch.tensor([[args.start_point, 0.]])  # 初始化估计值

    for kk in range(args.data_size-1):
        # 1. Training performance on training data
        current_y0 = true_y[kk, :, :]
        # 1) Neural ODE
        temp0 = odeint(func, current_y0, t[:2])
        pred_y_1[kk, :, :] = temp0[1, :, :]

        # 2. Training performance on testing data
        current_y0_test = true_y_test[kk, :, :]
        # 1) Neural ODE
        temp3 = odeint(func, current_y0_test, t[:2])
        pred_y_1_test[kk, :, :] = temp3[1, :, :]
        print('Neural error:' + str(true_y_test[kk+1, :, :] - pred_y_1_test[kk, :, :]))

        # 2) Feedback NN
        dydt_hat = dydt_NN[kk, :, :] + torch.mm((current_y0_test - current_y_hat), L)  # 修正导数
        current_y_hat_new = current_y_hat + sample_freq * dydt_hat  # 一步预测
        # pred_y_1_Feedback[kk, :, :] = current_y_hat_new
        pred_y_1_Feedback[kk, :, :] = current_y0_test + sample_freq * dydt_hat
        print('FNN error:' + str(true_y_test[kk+1, :, :] - pred_y_1_Feedback[kk, :, :]))
        print('One loop')
        current_y_hat = current_y_hat_new
        dydt_Feedback[kk, :, :] = dydt_hat  # 存储修正后导数

    # Calculate prediction error
    x_true = true_y_test.cpu().numpy()[1:, 0, 0]
    y_true = true_y_test.cpu().numpy()[1:, 0, 1]
    x_test_FNN = pred_y_1_Feedback.detach().numpy()[:, 0, 0]
    y_tes_FNN = pred_y_1_Feedback.detach().numpy()[:, 0, 1]
    Error_FNN = np.sqrt((x_test_FNN - x_true) ** 2 + (y_tes_FNN - y_true) ** 2)
    x_test_NN = pred_y_1_test.detach().numpy()[:, 0, 0]
    y_tes_NN = pred_y_1_test.detach().numpy()[:, 0, 1]
    Error_NN = np.sqrt((x_test_NN - x_true) ** 2 + (y_tes_NN - y_true) ** 2)

    # plot
    visualize(true_y, true_y_test, dydt_true_train, dydt_true, dydt_NN, dydt_Feedback, Error_NN, Error_FNN, func, 3)

    # python a_one_step_pre.py --viz
