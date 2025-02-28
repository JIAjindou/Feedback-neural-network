import numpy as np
import casadi as ca
import torch
import time
import os
from scipy.io import loadmat
import random


class Core:
    def __init__(self):
        # Get learning framework directory
        self.frame_dir = os.path.split(os.path.realpath(__file__))[0]

    """Recording and Saving"""

    def get_epoch_num(self):
        epoch_num = np.loadtxt(self.frame_dir + '/' + "epoch_num.txt")
        return int(epoch_num)

    def record_run(self, type, method):

        current_time = time.strftime("%b%d_%H-%M-%S_%Y", time.localtime())
        self.case_name = "{0}_{1}".format(current_time, type + "_" + method)
        self.log_path = self.frame_dir + "/log/" + self.case_name
        self.learned_models_path = self.frame_dir + "/learned_models/{0}_{1}".format(
            current_time, type + "_" + method
        )
        os.mkdir(self.log_path)
        os.mkdir(self.learned_models_path)

    def record_loss(self, loss_seq):
        path = self.log_path
        np.save(path + "/loss_record.npy", loss_seq)

    def save_model(self, model, epoch):
        torch.save(
            model.state_dict(),
            self.learned_models_path + "/model_param_epoch{}.pth".format(epoch),
        )
        torch.save(model.state_dict(), self.frame_dir+"/temp/model_param.pth")
        params = self.get_modelparam_vec(model.parameters())
        np.save(self.frame_dir+"/temp/model_param.npy", params)

    def get_modelparam_vec(self, model_param_gen):
        # Pytorch model parameter to a vector
        params = []
        for param in model_param_gen:
            # print(param)
            param_cpu = param.cpu()
            param_array = param_cpu.detach().numpy().reshape(-1, 1)
            params += [param_array]
        params = np.vstack(params)
        
        return params

    def pth2params_savenpy(self, model_class, pth_path, npy_path=None):
        model = model_class()
        model.load_state_dict(torch.load(pth_path))
        # print(model.state_dict())
        params = self.get_modelparam_vec(model.parameters())
        if npy_path:
            np.save(npy_path, params)

        return params
    