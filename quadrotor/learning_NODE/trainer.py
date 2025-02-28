import numpy as np
import casadi as ca
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import time

from learning_NODE.core import Core


class E2E_Trainer(Core):
    def __init__(self, model_wrapper, param_reg):
        """
        End-to-End Learning Framework for: E2E_Trainer

        Features:

        (1) A neural ODE / neural ODE augmented dynamics learning method
        based on variational method.

        (2) End-to-End Learning of model-mismatch from tracking error

        (3) Symbolic Represenation + Gradient Computation using CasADi

        (4) ADAM optimizer

        Problem:
        params = argmin \sum_0^N-1 (xk - xk_real)^T Q (xk - xk_real)
                    s.t. xk+1 = Phi(xk, pi(xk, aux_inputk), h, params)

        len xk: N
        len xk_real: N
        len aux_inputk: N-1

        Note:
        1. All vectors are in size (dim, 1), Matrices that store them: (dim, N)
        2. For batches: each xk_real batch has a batch-size of M while that of aux_inputk batch is M-1.s
        """

        super(E2E_Trainer, self).__init__(model_wrapper, param_reg)

        # Build Symbolic Expression of Grads and Jacobians
        self.build_symFuncs()

        # Prepare Data
        self.load_trajectories()

    """Setup Basic Symbolic"""

    def build_symFuncs(self):
        # Symbolics Variable
        state = ca.SX.sym("state", self.model.x_dim, 1)
        aux_input = ca.SX.sym("aux_input", self.model.aux_input_dim, 1)
        params = ca.SX.sym("params", self.model.params_dim, 1)
        state_real = ca.SX.sym("state_real", self.model.x_dim, 1)

        # Closed-Loop Dynamics with Params
        dynRK4_rhs = self.model.augdyn_symRK4(state, aux_input, params)
        # self.dyn_RK4func = ca.Function("dynRK4", [state, aux_input, params], [dynRK4_rhs])

        # Jacobian pPhi_px
        dynRK4_rhs = self.model.augdyn_symRK4(state, aux_input, params)
        pPhi_px_jac = ca.jacobian(dynRK4_rhs, state)
        self.pPhi_px = ca.Function("pPhi_px", [state, aux_input, params], [pPhi_px_jac])

        # Jacobian pPhi_pParams
        pPhi_pParams_jac = ca.jacobian(dynRK4_rhs, params)
        self.pPhi_pParams = ca.Function(
            "pPhi_pParams", [state, aux_input, params], [pPhi_pParams_jac]
        )

        # Gradient pl_px
        plk_px_rhs = ca.gradient(self.loss_k(state, state_real, params), state)
        plN_px_rhs = ca.gradient(self.loss_N(state, state_real, params), state)
        self.plk_px = ca.Function("plk_px", [state, state_real, params], [plk_px_rhs])
        self.plN_px = ca.Function("plN_px", [state, state_real, params], [plN_px_rhs])

        # Gradient pl_pParams
        plk_pParams_rhs = ca.gradient(self.loss_k(state, state_real, params), params)
        self.plk_pParams = ca.Function(
            "plk_pParams", [state, state_real, params], [plk_pParams_rhs]
        )
        self.plN_pParams = self.plk_pParams

        # # Hessian grad pl_pParams
        # plk_pParams_hess = ca.jacobian(plk_pParams_rhs, params)
        # self.plk_pParams_hess = ca.Function(
        #     "plk_pParams_hess", [state, state_real, params], [plk_pParams_hess]
        # )
        # self.plN_pParams_hess = self.plk_pParams_hess

        # # Hessian pPhi_pParams
        # pPhi_pParams_hess = ca.jacobian(pPhi_pParams_jac, params)
        # self.pPhi_pParams_hess = ca.Function("pPhi_pParams_hess", [state, aux_input, params], [pPhi_pParams_hess])

    """Functions for learning"""

    # Hamiltonian
    # def get_hamiltonian(self, xk_seq, xk_real_seq, aux_inputk_seq, lambdak_seq, params):
    #     H = 0
    #     for i in range(aux_inputk_seq.shape[1]):
    #         H += self.loss_k(xk_seq[:, i], xk_real_seq[:, i], params) + lambdak_seq[
    #             :, i
    #         ].T @ self.dynamics_RK4(xk_seq[:, i], aux_inputk_seq[:, i], params)
    #     H += self.loss_N(xk_seq[:, -1], xk_real_seq[:, -1], params)
    #     return H

    def adjoint_solve(self, xk_seq, xk_real_seq, aux_inputk_seq, params):
        """Solve Adjoint ODE's Initial Value Problem"""
        lambda_f = self.plN_px(xk_seq[:, -1], xk_real_seq[:, -1], params)
        lambda_k = lambda_f
        lambdak_seq = [lambda_f]
        for i in range(aux_inputk_seq.shape[1]):
            # Reverse Index
            k = aux_inputk_seq.shape[1] - 1 - i
            # Get current state, state_real and aux_input
            xk = xk_seq[:, k]
            xk_real = xk_real_seq[:, k]
            aux_inputk = aux_inputk_seq[:, k]
            # Compute Previous
            lambda_k0 = (
                self.plk_px(xk, xk_real, params)
                + self.pPhi_px(xk, aux_inputk, params).T @ lambda_k
            )
            lambdak_seq.append(lambda_k0)
            # Update lambda_k
            lambda_k = lambda_k0

        lambdak_seq = ca.horzcat(*lambdak_seq[::-1])  # Reverse Lambda Sequence
        return lambdak_seq

    def form_grad(self, xk_seq, xk_real_seq, aux_inputk_seq, lambdak_seq, params):
        gradient = 0
        for k in range(aux_inputk_seq.shape[1]):
            # Get current state, state_real, aux_input and co-state
            xk = xk_seq[:, k]
            xk_real = xk_real_seq[:, k]
            aux_inputk = aux_inputk_seq[:, k]
            lambdak = lambdak_seq[:, k]
            gradient += (
                self.plk_pParams(xk, xk_real, params)
                + self.pPhi_pParams(xk, aux_inputk, params).T @ lambdak
            )
        gradient += self.plN_pParams(xk_seq[:, -1], xk_real_seq[:, -1], params)
        gradient += self.reg * params  # Gradient with regularization
        return gradient

    def get_loss_linesearch(self, x0, aux_inputk_seq, xk_real_seq, params_ls):
        # Get loss during linesearch: rollout and get loss
        xk_seq_ls = self.forward_rollout(x0, aux_inputk_seq, params_ls)
        loss_ls = self.get_loss(xk_seq_ls, xk_real_seq, params_ls)
        return loss_ls

    """Optimization / Learning"""

    def optimize_minibatch(
        self, params_init, step_init, minibatch_size, epochs
    ):
        """
        Learn from multiple trajectories using mini-batching.
        Currently: ADAM optimizer
        """
        params_ = params_init
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        mk_ = 0
        vk_ = 0
        loss_seq = []

        self.create_minibatches(minibatch_size)
        btchlen = len(self.minibatch_Wrapper)

        self.record_run("minibatch", "adam")

        ii = 0
        while ii != epochs:
            # Go through the mini-batches for training
            btch_id = 1
            loss = 0.0
            for xk_real_seq, aux_inputk_seq in self.minibatch_Wrapper:
                # Get Current Initial State
                x0 = xk_real_seq[:, 0]
                # Forward Rollout
                xk_seq = self.forward_rollout(x0, aux_inputk_seq, params_)
                # Adjoint Solve
                lambdak_seq = self.adjoint_solve(xk_seq, xk_real_seq, aux_inputk_seq, params_)
                # Form Gradient
                grad = self.form_grad(
                    xk_seq, xk_real_seq, aux_inputk_seq, lambdak_seq, params_
                )
                # Get Loss
                loss += self.get_loss(xk_seq, xk_real_seq, params_).full()[0][0]

                # Adam Parameter Compute
                mk1 = beta1 * mk_ + (1 - beta1) * grad
                vk1 = beta2 * vk_ + (1 - beta2) * grad**2
                mkhat = mk1 / (1 - beta1 ** (ii + 1))
                vkhat = vk1 / (1 - beta2 ** (ii + 1))

                # Update Params
                params_ = params_ - step_init * mkhat / ((vkhat) ** 0.5 + eps)
                # Update Adam Parameters
                mk_ = mk1
                vk_ = vk1
                # Display Epoch and loss
                print(
                    f"\rEpoch [{ii+1}/{epochs}] ",
                    "BatchProgress: {}% ".format(int(btch_id / btchlen * 100)),
                    end="",
                )

                btch_id += 1

            loss_seq += [loss / btchlen]

            # Display Epoch and loss
            print("Current Loss: {}".format(loss / btchlen))

            # Record Epoch and Current Model
            self.record_loss(np.vstack(loss_seq))
            self.save_model(params_, epoch=ii + 1)
            ii += 1

        return params_
