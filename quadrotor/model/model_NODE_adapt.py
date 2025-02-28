import numpy as np
import casadi as ca


class Quadrotor_NODE_Adapt:
    def __init__(self, adaptive_gain):
        """
        Open-loop model of a quadrotor:

        1. Open loop model of a quadrotor with NN ode as part of the model.
            dp = v
            dv = a + NNode(v, rpy, u)
            dEul = W(Eul) @ pqr
            dpqr = J^-1 (J pqr x pqr + tau)

            x = [p, v, eul, pqr]
            u = [T1, T2, T3, T4]
            dx = f(x, u)

        2. An adaptive mechanism on the full-connected layer is developed for online model adaptation.
            Ref: Y. Cheng, et al. "Human motion prediction using semi-adaptable neural networks." 2019 American Control Conference (ACC). IEEE, 2019.

        This model accepts a learned parameter of `Quadrotor_NODE`
        and is designed for model predictive control as well as other online application.
        """
        self.__dynamic_param()
        # self.__control_param()
        self.__saturation_params()

        # Model Information
        self.x_dim = 18
        self.ol_x_dim = 12
        self.u_dim = 4
        self.aux_input_dim = 12

        # NN ode setting
        self.input_size = 6
        self.hidden_size = 36
        self.output_size = 3

        # Adaptive module setting
        self.adaptive_gain = adaptive_gain

        # Initialize and Compute param dimension
        x_test = ca.DM.ones(6)
        self.NN_base(x_test, params=ca.DM.ones(10000))  # sufficient param dim
        print("param size: ", self.params_base_dim + self.output_size * self.hidden_size)

    def load_params(self, paramsFull):
        # load all params for the complete NN
        self.paramsBase = paramsFull[: -self.hidden_size * self.output_size]
        self.FC_vec = paramsFull[-self.hidden_size * self.output_size :]

    def __dynamic_param(self):
        self.m = 0.83
        self.Ixx = 3e-3
        self.Iyy = 3e-3
        self.Izz = 4e-3
        self.__compute_J()

        torque_coef = 0.01
        arm_length = 0.150
        self.CoeffM = np.array(
            [
                [
                    0.25,
                    -0.353553 / arm_length,
                    0.353553 / arm_length,
                    0.25 / torque_coef,
                ],
                [
                    0.25,
                    0.353553 / arm_length,
                    -0.353553 / arm_length,
                    0.25 / torque_coef,
                ],
                [
                    0.25,
                    0.353553 / arm_length,
                    0.353553 / arm_length,
                    -0.25 / torque_coef,
                ],
                [
                    0.25,
                    -0.353553 / arm_length,
                    -0.353553 / arm_length,
                    -0.25 / torque_coef,
                ],
            ]
        )  # Ttau -> MF
        self.CoeffM_inv = np.linalg.inv(self.CoeffM)  # Ttau -> MF

    def __compute_J(self):
        self.J = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))
        self.J_inv = np.diag(np.array([1 / self.Ixx, 1 / self.Iyy, 1 / self.Izz]))

    def __saturation_params(self):
        self.u_lb = np.array([0.0, 0.0, 0.0, 0.0])
        self.u_ub = np.array([4.0, 4.0, 4.0, 4.0]) * 1.5

    def __linear_layer(self, x, param, dim_in, dim_out):
        W = param[0 : dim_in * dim_out].reshape((dim_out, dim_in))
        b = param[dim_in * dim_out : dim_in * dim_out + dim_out]
        return W @ x + b

    def __linear_layer_nobias(self, x, param, dim_in, dim_out):
        W = param[0 : dim_in * dim_out].reshape((dim_out, dim_in))
        return W @ x

    def NN_base(self, x, params):
        """
        NN ode for learning the translational residual dynamics of a quadrotor
        """
        x_ = x.reshape((-1, 1))
        param_idx = 0
        param_idx_1 = 0
        # Input Layer
        dim_in = self.input_size
        dim_out = self.hidden_size
        param_idx_1 += dim_in * dim_out + dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        x_ = ca.tanh(x_)

        # Hidden Layer 1
        dim_in = self.hidden_size
        dim_out = self.hidden_size
        param_idx_1 += dim_in * dim_out + dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        x_ = ca.tanh(x_)

        # Hidden Layer 2
        dim_in = self.hidden_size
        dim_out = self.hidden_size
        param_idx_1 += dim_in * dim_out + dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer(x_, param_, dim_in, dim_out)
        x_ = ca.tanh(x_)

        self.params_base_dim = param_idx_1

        return x_

    def adaptive_NN(self, x_in, param_FC):
        x_ = self.NN_base(x_in, self.paramsBase)
        return param_FC @ x_

    def openloop_augmented(self, x, MF, FC_vec):
        Ttau = self.CoeffM_inv @ MF

        dp = x[3:6]

        dvx = (
            -Ttau[0]
            / self.m
            * (ca.cos(x[8]) * ca.sin(x[7]) * ca.cos(x[6]) + ca.sin(x[8]) * ca.sin(x[6]))
        )
        dvy = (
            -Ttau[0]
            / self.m
            * (ca.sin(x[8]) * ca.sin(x[7]) * ca.cos(x[6]) - ca.cos(x[8]) * ca.sin(x[6]))
        )
        dvz = 9.8 - Ttau[0] / self.m * (ca.cos(x[6]) * ca.cos(x[7]))
        param_FC = FC_vec.reshape((self.output_size, self.hidden_size))
        dv = ca.vertcat(dvx, dvy, dvz) - self.adaptive_NN(x[3:9], param_FC)

        deul = (
            ca.vertcat(
                ca.horzcat(1, ca.tan(x[7]) * ca.sin(x[6]), ca.tan(x[7]) * ca.cos(x[6])),
                ca.horzcat(0, ca.cos(x[6]), -ca.sin(x[6])),
                ca.horzcat(0, ca.sin(x[6]) / ca.cos(x[7]), ca.cos(x[6]) / ca.cos(x[7])),
            )
            @ x[9:12]
        )

        domega = self.J_inv @ (-ca.cross(x[9:12], self.J @ x[9:12]) + Ttau[1:4])

        return ca.vertcat(dp, dv, deul, domega)

    def openloop_forCtrl(self, xk, uk, FC_vec):
        dx = self.openloop_augmented(xk, uk, FC_vec)
        return dx

    def adaptive_rate(self, xk_in, ek):
        temp = [self.NN_base(xk_in, self.paramsBase).full() * ek[i] for i in range(self.output_size)]
        dvec = self.adaptive_gain * np.vstack(temp)
        return dvec

    def adaptor_update(self, xk, ek, ts):
        self.FC_vec += ts * self.adaptive_rate(xk, ek)
