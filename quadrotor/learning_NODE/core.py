import numpy as np
import casadi as ca
import time
import os
from scipy.io import loadmat
import random


class Core:
    def __init__(self, model_wrapper, param_reg):
        """
        End-to-End Learning Framework: Core Functions

        Basic Functions for both learning and evaluation.
        1- Integrators
        2- Loss functions
        3- Data-process functions

        Problem:
        params = argmin \sum_0^N-1 (xk - xk_real)^T Q (xk - xk_real)
                    s.t. xk+1 = Phi(xk, pi(xk, aux_inputk), h, params)

        len xk: N
        len xk_real: N
        len aux_inputk: N-1

        Note:
        1. All vectors are in size (dim, 1), Matrices that store them: (dim, N)
        2. For batches: each xk_real batch has a batch-size of M while that of aux_inputk batch is M-1.
        """
        self.model = model_wrapper
        self.Q = np.diag([1.0] * 12 + [0.0] * 6)
        self.reg = param_reg  # Regulation for params

        # Get learning framework directory
        self.frame_dir = os.path.split(os.path.realpath(__file__))[0]

    """Basic Functions"""

    # Rollout / Integration
    def forward_rollout(self, x0, aux_inputk_seq, params):
        xk_seq = x0
        xk = x0
        for i in range(aux_inputk_seq.shape[1]):
            aux_inputk = aux_inputk_seq[:, i]
            xk1 = self.model.augdyn_symRK4(xk, aux_inputk, params)
            xk_seq = ca.horzcat(xk_seq, xk1)
            xk = xk1

        return xk_seq

    # def forward_rollout_num(self, x0, aux_inputk_seq, params):
    #     xk_seq = x0
    #     xk = x0
    #     for i in range(aux_inputk_seq.shape[1]):
    #         aux_inputk = aux_inputk_seq[:, i]
    #         xk1 = self.model.augdyn_numRK4(xk, aux_inputk, params)
    #         xk_seq = np.vstack([xk_seq, xk1])
    #         xk = xk1

    #     return xk_seq.T

    # Loss Function at step k
    def loss_k(self, xk, xk_real, params):
        loss = 0.5 * (xk - xk_real).T @ self.Q @ (xk - xk_real)
        return loss

    def loss_N(self, xk, xk_real, params):
        loss = 0.5 * (xk - xk_real).T @ self.Q @ (xk - xk_real)
        return loss

    def get_loss(self, xk_seq, xk_real_seq, params):
        loss = 0.0
        for i in range(xk_seq.shape[1]):
            loss += self.loss_k(xk_seq[:, i], xk_real_seq[:, i], params)

        loss += self.loss_N(xk_seq[:, -1], xk_real_seq[:, -1], params)
        return loss

    """Load trajectories for batch learning"""

    def load_trajectories(self, folder_dir="data"):
        # Add frame dir
        # folder_dir = self.frame_dir + '/' + folder_dir
        # Get trajectories and create batch loader
        self.traj_dir_list = [
            file for file in os.listdir(folder_dir) if file.endswith(".mat")
        ]
        print("{} trajectories loaded.".format(len(self.traj_dir_list)))
        self.xk_real_loader = []
        self.aux_inputk_loader = []
        size_ = []
        for traj_dir in self.traj_dir_list:
            learndata = loadmat(folder_dir + "/" + traj_dir)
            xk_real_seq = learndata["xk_real_seq"]
            aux_inputk_seq = learndata["aux_inputk_seq"]
            self.xk_real_loader += [xk_real_seq]
            self.aux_inputk_loader += [aux_inputk_seq]
            size_ += [aux_inputk_seq.shape[1]]
        # Count the maximum batch size avaliable: smallest trajectory horizon
        self.max_btchsize = min(size_)
        print("Max batch size: ", self.max_btchsize)

    def get_batch(self, batch_size):
        """Random choose a trajectory batch for training"""
        # Bounded batch size
        batch_size = min(batch_size, self.max_btchsize)
        # Get avaliable batch indices: (traj_id, aux_input_id)
        self.btchid_options = []
        for i in range(len(self.aux_inputk_loader)):
            for j in range(self.max_btchsize - batch_size + 1):
                self.btchid_options += [(i, j)]
        # Get batch
        num_items = len(self.btchid_options)
        random_index = random.randrange(num_items)
        btch_start = self.btchid_options[random_index]
        print("Batch Selected:", btch_start, end=" ")
        traj_id, state_id = btch_start[0], btch_start[1]
        xk_real_seq = self.xk_real_loader[traj_id][
            :, state_id : state_id + batch_size + 2
        ]
        aux_inputk_seq = self.aux_inputk_loader[traj_id][:, state_id : state_id + batch_size + 1]
        return xk_real_seq, aux_inputk_seq

    """Mini-batching for End2End"""

    def get_minibatches_singletraj(self, xk_real_seq, aux_inputk_seq, minibatch_size):
        # remove initial state
        xk_real_mbs = np.array_split(
            xk_real_seq[:, 1:],
            xk_real_seq[:, 1:].shape[1] // minibatch_size,
            axis=1,
        )
        aux_inputk_mbs = np.array_split(aux_inputk_seq, aux_inputk_seq.shape[1] // minibatch_size, axis=1)
        # Add initial states (reshape required for hstack in numpy)
        last_xk_real = xk_real_seq[:, 0].reshape((-1, 1))
        for i in range(len(xk_real_mbs)):
            xk_real_mbs[i] = np.hstack([last_xk_real, xk_real_mbs[i]])
            last_xk_real = xk_real_mbs[i][:, -1].reshape((-1, 1))

        return xk_real_mbs, aux_inputk_mbs

    def create_minibatches(self, minibatch_size, shuffle=True):
        """Go throught all mini-batches for E2E training"""
        # Bounded minibatch size
        minibatch_size = min(minibatch_size, self.max_btchsize)
        # Create mini-batches
        self.xk_real_minibatches = []
        self.aux_inputk_minibatches = []
        # Divide trajectory full-batches into mini-batches
        for xk_real_seq, aux_inputk_seq in zip(self.xk_real_loader, self.aux_inputk_loader):
            xk_real_mbs, aux_inputk_mbs = self.get_minibatches_singletraj(
                xk_real_seq, aux_inputk_seq, minibatch_size
            )
            # Add to mini-batches
            self.xk_real_minibatches += xk_real_mbs
            self.aux_inputk_minibatches += aux_inputk_mbs
        print(
            "mini-batches: ", len(self.xk_real_minibatches), len(self.aux_inputk_minibatches)
        )
        self.minibatch_Wrapper = list(
            zip(self.xk_real_minibatches, self.aux_inputk_minibatches)
        )
        if shuffle:
            # Shuffle Minibatches
            random.shuffle(self.minibatch_Wrapper)
            self.xk_real_minibatches = [tup[0] for tup in self.minibatch_Wrapper]
            self.aux_inputk_minibatches = [tup[1] for tup in self.minibatch_Wrapper]

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

    def record_epoch(self, loss_seq, hyperparam_seq, epoch):
        path = self.log_path
        # Record loss
        self.record_loss(loss_seq)
        # Record Hyperparam
        try:
            if hyperparam_seq == None:
                pass
        except ValueError:
            np.save(
                path + "/hyperparam_record.npy",
                hyperparam_seq,
            )
            np.save("temp/hyperparam.npy", hyperparam_seq[:, -1])
        # Record Epoch
        np.save("temp/last_epoch.npy", epoch)

    def save_model(self, params, epoch):
        path = self.learned_models_path
        np.save(path + "/" + "model_param_epoch{}.npy".format(epoch), params)
        np.save(self.frame_dir + '/' + "temp/model_param.npy", params)
