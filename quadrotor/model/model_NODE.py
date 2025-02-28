import numpy as np
import casadi as ca


class Quadrotor_NODE:
    def __init__(self, discrete_h_learn):
        """
        Open-loop + Closed-loop model of a quadrotor:

        1. Open loop model of a quadrotor with NN ode as part of the model.
            dp = v
            dv = a + NNode(v, rpy, u)
            dEul = W(Eul) @ pqr
            dpqr = J^-1 (J pqr x pqr + tau)

            x = [p, v, eul, pqr]
            u = [T1, T2, T3, T4]
            dx = f(x, u)
            
        2. Mellinger Controller / Differential Flatness-based Controller as Control Policy
            u, d(aux_x)/dt = ctrl(x, ref, aux_x)
            aux_x are dynamic variable in mellinger controller, serves as integrator and
            differentiator in PID control.

        The model is differentiable and used for adjoint-based learning and model predictive control.
        The controller is for stable integration during adjoint-based learning.
        """
        self.__dynamic_param()
        self.__control_param()
        self.__saturation_params()

        # Model Information
        self.x_dim = 18
        self.ol_x_dim = 12
        self.u_dim = 4
        self.aux_input_dim = 12
        self.h = discrete_h_learn

        # NN ode setting
        self.input_size = 6
        self.hidden_size = 36
        self.output_size = 3

        # Initialize and Compute param dimension
        x_test = ca.DM.ones(6)
        self.NN_model(x_test, params=ca.DM.ones(10000))  # sufficient param dim
        print("param size: ", self.params_dim)

    def load_params_for_OC(self, params):
        # This is for the usage of model
        self.params = params

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

    def __control_param(self):
        self.pos_gain = np.diag(np.array([1.0, 1.0, 0.7])) * 2
        self.vel_gain = self.pos_gain * 4
        self.eul_gain = np.diag(np.array([10.0, 10.0, 4.0]))
        self.omega_P = np.diag(np.array([40.0, 40.0, 16.0]))
        self.omega_I = np.diag(np.array([10.0, 10.0, 5.0])) 
        self.omgea_D = np.diag(np.array([0.5, 0.5, 0.0]))

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
    
    def __dEul2omega_sym(self, dEul_des, Eul):
        # Strap Down Equations
        domega_xdes = dEul_des[0] - (ca.sin(Eul[1]) * dEul_des[2])
        domega_ydes = (dEul_des[1] * ca.cos(Eul[0])) + (
            dEul_des[2] * ca.sin(Eul[0]) * ca.cos(Eul[1])
        )
        domega_zdes = -(dEul_des[1] * ca.sin(Eul[0])) + (
            dEul_des[2] * ca.cos(Eul[0]) * ca.cos(Eul[1])
        )
        return ca.vertcat(domega_xdes, domega_ydes, domega_zdes)

    def __dEul2omega_num(self, dEul_des, Eul):
        # Strap Down Equations
        domega_xdes = dEul_des[0] - (np.sin(Eul[1]) * dEul_des[2])
        domega_ydes = (dEul_des[1] * np.cos(Eul[0])) + (
            dEul_des[2] * np.sin(Eul[0]) * np.cos(Eul[1])
        )
        domega_zdes = -(dEul_des[1] * np.sin(Eul[0])) + (
            dEul_des[2] * np.cos(Eul[0]) * np.cos(Eul[1])
        )
        return np.hstack([domega_xdes, domega_ydes, domega_zdes])

    def __invert_eul_sym(self, moment_des, omega):
        m1 = moment_des[0] + omega[1] * omega[2] * (self.Izz - self.Iyy)
        m2 = moment_des[1] + omega[0] * omega[2] * (self.Ixx - self.Izz)
        m3 = moment_des[2] + omega[0] * omega[1] * (self.Iyy - self.Ixx)
        return ca.vertcat(m1, m2, m3)

    def __invert_eul_num(self, moment_des, omega):
        m1 = moment_des[0] + omega[1] * omega[2] * (self.Izz - self.Iyy)
        m2 = moment_des[1] + omega[0] * omega[2] * (self.Ixx - self.Izz)
        m3 = moment_des[2] + omega[0] * omega[1] * (self.Iyy - self.Ixx)
        return np.hstack([m1, m2, m3])

    def __derative3(self, interstate, x):
        # c=0.05 tf = s/(c*s+1)
        d_interstate = -10 * np.eye(3) @ interstate + 8 * np.eye(3) @ x
        x_der = -12.5 * np.eye(3) @ interstate + 10 * np.eye(3) @ x
        return d_interstate, x_der
    
    def NN_model(self, x, params):
        """
        NN ode for learing the translational dynamics of a quadrotor
        """
        x_ = x.reshape((-1,1))
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
        
        # Full Connected Layer
        dim_in = self.hidden_size
        dim_out = self.output_size
        # param_idx_1 += dim_in * dim_out + dim_out
        param_idx_1 += dim_in * dim_out
        param_ = params[param_idx:param_idx_1]
        param_idx = param_idx_1

        x_ = self.__linear_layer_nobias(x_, param_, dim_in, dim_out)

        self.params_dim = param_idx_1
        return x_
    
    def openloop_augmented(self, x, MF, params):
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
        dv = ca.vertcat(dvx, dvy, dvz) - self.NN_model(x[3:9], params)

        deul = (
            ca.vertcat(
                ca.horzcat(1, ca.tan(x[7]) * ca.sin(x[6]), ca.tan(x[7]) * ca.cos(x[6])),
                ca.horzcat(0, ca.cos(x[6]), -ca.sin(x[6])),
                ca.horzcat(0, ca.sin(x[6]) / ca.cos(x[7]), ca.cos(x[6]) / ca.cos(x[7])),
            )
            @ x[9:12]
        )

        domega = self.J_inv @ (
            -ca.cross(x[9:12], self.J @ x[9:12]) + Ttau[1:4]
        )

        return ca.vertcat(dp, dv, deul, domega)

    def ctrlmap_sym(self, x, ref, omega_err_inte, omega_err_ds):
        # demux
        pos = x[0:3]
        vel = x[3:6]
        eul = x[6:9]
        omega = x[9:12]
        pos_ref = ref[0:3]
        vel_ref = ref[3:6]
        acc_ref = ref[6:9]
        jer_ref = ref[9:12]

        # Translational Loop
        a_des = acc_ref + self.vel_gain @ (
            self.pos_gain @ (pos_ref - pos) + vel_ref - vel
        )

        # Obtain Desire Rebs
        Zb = ca.vertcat(-a_des[0], -a_des[1], 9.8 - a_des[2]) / ca.norm_2(
            ca.vertcat(-a_des[0], -a_des[1], 9.8 - a_des[2])
        )

        Xc = ca.vertcat(ca.cos(0.0), ca.sin(0.0), 0.0)
        Yb_ = ca.cross(Zb, Xc)
        Yb = Yb_ / ca.norm_2(Yb_)
        Xb = ca.cross(Yb, Zb)
        # Reb_des = np.vstack([Xb, Yb, Zb]).T
        Reb_des = ca.horzcat(Xb, Yb, Zb)

        # Obtain Desire Eul, Omega and dOmega
        # eul_des = 'ZYX' + 1,3 Switch
        eul_des = ca.vertcat(
            ca.arctan2(Reb_des[2, 1], Reb_des[2, 2]),
            ca.arctan2(
                -Reb_des[2, 0], ca.sqrt(Reb_des[1, 0] ** 2 + Reb_des[0, 0] ** 2)
            ),
            ca.arctan2(Reb_des[1, 0], Reb_des[0, 0]),
        )

        T = -self.m * ca.dot(Zb, (a_des - np.array([0.0, 0.0, 9.8])))
        h1 = -self.m / T * (jer_ref - ca.dot(Zb, jer_ref) * Zb)

        omega_des = ca.vertcat(-ca.dot(h1, Yb), ca.dot(h1, Xb), 0.0)

        # h2 = (
        #     -ca.cross(omega_des, ca.cross(omega_des, Zb))
        #     + self.m / T * ca.dot(jer_ref, Zb) * ca.cross(omega_des, Zb)
        #     + 2 * self.m / T * ca.dot(Zb, jer_ref) * ca.cross(omega_des, Zb)
        # )
        # domega_des = ca.vertcat(-ca.dot(h2, Yb), ca.dot(h2, Xb), 0.0)

        # Attitude Loop
        dEul_des = self.eul_gain @ (eul_des - eul)
        omega_err = omega_des - omega + self.__dEul2omega_sym(dEul_des, eul)
        d_omega_err_ds, omega_err_der = self.__derative3(omega_err_ds, omega_err)
        att_out = (
            self.omega_P @ omega_err
            + self.omega_I @ omega_err_inte
            + self.omgea_D @ omega_err_der
            + ca.cross(omega, self.J @ omega)
            # + self.J @ domega_des
        )
        moment_des = self.J @ att_out
        tau = self.__invert_eul_sym(moment_des, omega)

        return (
            self.CoeffM @ ca.vertcat(T, tau[0], tau[1], tau[2]),
            omega_err,
            d_omega_err_ds,
        )

    def cldyn_sym(self, x, ref, params):
        MF, omega_err, d_omega_err_ds = self.ctrlmap_sym(
            x[0:12], ref, x[12:15], x[15:18]
        )
        dx = self.openloop_augmented(x, MF, params)
        return ca.vertcat(dx, omega_err, d_omega_err_ds)
        
    def augdyn_symRK4(self, xk, ref, params):
        # Closed-loop Augmented Dynamics: RK4 Discrete
        k1 = self.cldyn_sym(xk, ref, params)
        k2 = self.cldyn_sym((xk + 0.5 * self.h * k1), ref, params)
        k3 = self.cldyn_sym((xk + 0.5 * self.h * k2), ref, params)
        k4 = self.cldyn_sym((xk + self.h * k3), ref, params)
        xk1 = xk + self.h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def openloop_forCtrl(self, xk, uk):
        dx = self.openloop_augmented(xk, uk, self.params)
        return dx