import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

# 分离参数与代码 用于频繁修改参数
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=400)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

# 选取ODE求解器
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# 检查是否有可用的GPU
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# 初值
true_y0 = torch.tensor([[2., 0.]]).to(device)
# 观察时刻
t = torch.linspace(0., 25., args.data_size).to(device)
# 函数参数
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)
true_A_test = torch.tensor([[-0.1, 3.0], [-3.0, -0.1]]).to(device)


class Lambda(nn.Module):
    # 真实函数
    def forward(self, t, y):
        return torch.mm(y, true_A)

class Lambda_test(nn.Module):
    # 真实函数
    def forward(self, t, y):
        zz = torch.mm(y, true_A_test)
        return zz


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


if args.viz:
    makedirs('png')  # 创建png文件夹，存储中间训练的图像
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)  # frameon-图形边框
    ax_phase = fig.add_subplot(132, frameon=True)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)  # 关闭图形，保证代码继续运行下去


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_traj.cla()  # 移除坐标轴显示的内容
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
                     'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(),
                     pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        #ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.0001)

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()  # 子类继承了父类的所有属性和方法

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
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


def Feedback_enhanced(current_y, current_y_enganced, current_dydt_NN):
    L = np.array([[1, 0], [0, 1]])
    dydt_enhanced = current_dydt_NN + np.dot((current_y - current_y_enganced), L)
    sample_time = 25/args.data_size
    current_y_enganced_new = current_y_enganced + sample_time * dydt_enhanced
    return dydt_enhanced, current_y_enganced_new

def plot_dydt(dydt_true, dydt_NN, dydt_Feedback):
    fig = plt.figure(figsize=(12, 6), facecolor='white')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t.cpu().numpy(), dydt_true.detach().numpy()[:, 0], label='dydt_true')
    ax1.plot(t.cpu().numpy(), dydt_NN.detach().numpy()[:, 0], label='dydt_NN')
    ax1.plot(t.cpu().numpy(), dydt_Feedback[:, 0], label='dydt_Feedback')
    ax1.set_title('dydt_0')
    ax1.set_xlabel('x-time')
    ax1.set_ylabel('y-dydt')
    plt.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t.cpu().numpy(), dydt_true.detach().numpy()[:, 1], label='dydt_true')
    ax2.plot(t.cpu().numpy(), dydt_NN.detach().numpy()[:, 1], label='dydt_NN')
    ax2.plot(t.cpu().numpy(), dydt_Feedback[:, 1], label='dydt_Feedback')
    ax2.set_title('dydt_1')
    ax2.set_xlabel('x-time')
    ax2.set_ylabel('y-dydt')
    plt.legend()

    # 显示图形
    plt.show()


if __name__ == '__main__':

    ii = 0

    # 初始化NN和优化器
    func = ODEFunc().to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

    end = time.time()  # 获取当前时间戳

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)


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
                print(batch_t.size())
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()

    # 评估反馈引入的作用
    Lambda_fun = Lambda_test().to(device)
    dydt_true = Lambda_fun(0, true_y_test.reshape(-1, 2))    # 真实dydt
    dydt_NN = func(0, true_y_test.reshape(-1, 2))            # 神经网络dydt

    # 反馈修正神经网络dydt
    dydt_Feedback = np.zeros((args.data_size, 2))      #初始化dydt_Feedback
    current_y_enganced = torch.tensor([[2., 0.]])
    L = np.array([[10, 0], [0, 10]])                     #反馈增益
    for jj in range(args.data_size):
        current_y = true_y_test.reshape(-1, 2)[jj, :]
        current_dydt_NN = dydt_NN[jj, :]
        dydt_enhanced = current_dydt_NN.detach().numpy() + np.dot((current_y - current_y_enganced), L)
        sample_time = 25 / args.data_size
        current_y_enganced_new = current_y_enganced + sample_time * dydt_enhanced
        current_y_enganced = current_y_enganced_new
        dydt_Feedback[jj, :] = dydt_enhanced
    # 绘制图形
    #plot_dydt(dydt_true, dydt_NN, dydt_Feedback)

    # 预测测试
    time_prediction = torch.tensor([5.])
    pre_true = np.zeros((args.data_size-200, 2))
    pre_NN = np.zeros((args.data_size - 200, 2))
    pre_Feedback = np.zeros((args.data_size - 200, 2))

    func_feeback = ODEFunc_change().to(device)

    for kk in range(args.data_size-200):
        pre_true[kk, :] = true_y_test.reshape(-1, 2)[200+kk, :]
        pre_NN[kk, :] = odeint(func, true_y_test[kk, :], time_prediction)
        pre_Feedback[kk, :] = odeint(func_feeback, true_y_test[kk, :], time_prediction)

    fig = plt.figure(figsize=(12, 6), facecolor='white')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(pre_true[:, 0], pre_true[:, 1], label='true')
    ax1.plot(pre_NN[:, 0], pre_NN[:, 1], label='NN')
    ax1.plot(pre_Feedback[:, 0], pre_Feedback[:, 1], label='Feedback')
    ax1.set_title('prediction at 5s')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.legend()

    t2 = torch.linspace(0., 20., args.data_size-200).to(device)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t2.cpu().numpy(), pre_true[:, 0], label='true')
    ax2.plot(t2.cpu().numpy(), pre_NN[:, 0], label='NN')
    ax2.set_title('Trajectory')
    ax2.set_xlabel('x-time')
    ax2.set_ylabel('y')
    plt.legend()
    # 显示图形
    plt.show()

    # python my_code.py --viz
