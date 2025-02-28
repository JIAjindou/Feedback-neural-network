import numpy as np
import casadi as ca


class MPC:
    def __init__(self, model, discrete_h, H):
        self.model = model
        self.h = discrete_h
        self.H = H
        # Quadratic Objectives
        self.Q_k = 1e2 * np.diag(
            np.array([1.0] * 3 + [0.5] * 3 + [0.5] * 3 + [0.01] * 3)
        )
        self.R_k = np.diag(np.array([1.0] * 4))
        self.Q_f = self.Q_k
        # State Bound
        self.x_lb = [-ca.inf] * 6 + [-np.pi / 2] * 3 + [-ca.inf] * 3
        self.x_ub = [ca.inf] * 6 + [np.pi / 2] * 3 + [ca.inf] * 3
        # Input Bound
        self.u_lb = list(self.model.u_lb)
        self.u_ub = list(self.model.u_ub)

        # NLP Setup
        self.build_symbolic()

    def loss_kx(self, xk, xk_ref):
        return 0.5 * ca.transpose(xk - xk_ref) @ self.Q_k @ (xk - xk_ref)

    def loss_ku(self, uk, uk_ref):
        return 0.5 * ca.transpose(uk - uk_ref) @ self.R_k @ (uk - uk_ref)

    def loss_k(self, xk, uk, xk_ref, uk_ref):
        return self.loss_kx(xk, xk_ref) + self.loss_ku(uk, uk_ref)

    def loss_N(self, xf, xf_ref):
        return self.loss_kx(xf, xf_ref)

    def set_stateBoxCons(self, lb_list, ub_list):
        self.x_lb = lb_list
        self.x_ub = ub_list

    def set_inputBoxCons(self, lb_list, ub_list):
        self.u_lb = lb_list
        self.u_ub = ub_list

    def discrete_sys(self, xk, uk):
        h = self.h
        k1 = self.model.openloop_forCtrl(xk, uk)
        k2 = self.model.openloop_forCtrl((xk + 0.5 * h * k1), uk)
        k3 = self.model.openloop_forCtrl((xk + 0.5 * h * k2), uk)
        k4 = self.model.openloop_forCtrl((xk + h * k3), uk)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def build_symbolic(self):
        x = ca.SX.sym("x", 12)
        u = ca.SX.sym("u", 4)
        rhs = self.discrete_sys(x, u)
        self.discrete_dynfunc = ca.Function("dyn", [x, u], [rhs])
        x_ref = ca.SX.sym("x_ref", 12)
        u_ref = ca.SX.sym("u_ref", 4)
        loss_k = self.loss_k(x, u, x_ref, u_ref)
        loss_N = self.loss_N(x, x_ref)
        self.lk = ca.Function("lkx", [x, u, x_ref, u_ref], [loss_k])
        self.lN = ca.Function("lku", [x, x_ref], [loss_N])

    def solve_OC(self, x0, xref_seq, uref_seq):
        """
        Solve u with Nominal State

        refk_seq: comming up reference trajectory: state_dim * H

        nonlinear program (NLP):
            min          F(x, p)
            x

            subject to
            LBX <=   x    <= UBX
            LBG <= G(x, p) <= UBG
            p  == P

            nx: number of decision variables
            ng: number of constraints
            np: number of parameters
        """

        w = []  # Solution
        w0 = []  # Init guess
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # Optimize N*states+N*inputs = 16*N parameters
        Xk_ = ca.DM(x0)  # The first states
        # H = xref_seq.shape[1]
        for k in range(self.H):
            # Add control param
            Uk = ca.MX.sym("U_" + str(k), self.model.u_dim)
            w += [Uk]
            w0 += list(uref_seq[:, k])
            lbw += self.u_lb
            ubw += self.u_ub

            # Add state param and constraint (index + 1)
            Xk1 = ca.MX.sym("X_" + str(k + 1), self.model.ol_x_dim)
            w += [Xk1]
            w0 += list(xref_seq[:, k])
            lbw += self.x_lb
            ubw += self.x_ub

            # Add continous constraint
            g += [self.discrete_dynfunc(Xk_, Uk) - Xk1]
            lbg += [0] * self.model.ol_x_dim
            ubg += [0] * self.model.ol_x_dim

            # Add Obj
            if k == self.H - 1:
                J += self.lk(Xk_, Uk, xref_seq[:, k], uref_seq[:, k])
            else:
                J += self.lN(Xk_, xref_seq[:, k])

            # Refresh Xk_
            Xk_ = Xk1

        # Build Prob
        prob = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
        # options = {}
        options = {
            "verbose": False,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
        }
        solver = ca.nlpsol("solver", "ipopt", prob, options)

        # Solve
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # Get control and state in mpc
        result = (
            sol["x"]
            .full()
            .flatten()
            .reshape(-1, self.model.ol_x_dim + self.model.u_dim)
        )
        result = result.T
        u_opt_seq = result[0 : self.model.u_dim, :]
        x_opt_seq = result[self.model.u_dim :, :]

        return u_opt_seq, x_opt_seq

    def dx_modelpredict(self, xk, uk):
        dx = self.model.openloop_forCtrl(xk, uk)
        return dx


class MPC_MultiStep(MPC):
    def __init__(self, model, discrete_h, H):
        super(MPC_MultiStep, self).__init__(model, discrete_h, H)

        self.BOOL_xhat = False

    def set_GainAndDecay(self, L_init, decay_rate):
        self.L_init = L_init  # Initial Feedback Gain
        self.decay_rate = decay_rate

    def feedback_NNode(self, xk, xhatk, uk, L):
        # here xk is used for update the NN series / corrector states
        return self.model.openloop_forCtrl(xk, uk) + L @ (xk - xhatk)

    def feedback_NNode_RK4(self, xk, xhatk, uk, L):
        h = self.h
        k1 = self.feedback_NNode(xk, xhatk, uk, L)
        k2 = self.feedback_NNode((xk + 0.5 * h * k1), xhatk, uk, L)
        k3 = self.feedback_NNode((xk + 0.5 * h * k2), xhatk, uk, L)
        k4 = self.feedback_NNode((xk + h * k3), xhatk, uk, L)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def predictor_RK4(self, xhatk, xk, uk, L):
        h = self.h
        k1 = self.feedback_NNode(xk, xhatk, uk, L)
        k2 = self.feedback_NNode(xk, (xhatk + 0.5 * h * k1), uk, L)
        k3 = self.feedback_NNode(xk, (xhatk + 0.5 * h * k2), uk, L)
        k4 = self.feedback_NNode(xk, (xhatk + h * k3), uk, L)
        xhatk1 = xhatk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xhatk1

    def build_symbolic(self):
        x = ca.SX.sym("x", 12)
        xhat = ca.SX.sym("xhat", 12)
        u = ca.SX.sym("u", 4)
        L = ca.SX.sym("L", 12, 12)
        rhs_openloop = self.discrete_sys(x, u)
        self.discrete_dynfunc = ca.Function("dyn", [x, u], [rhs_openloop])
        rhs_feedback = self.feedback_NNode_RK4(x, xhat, u, L)
        self.discrete_FBdynfunc = ca.Function(
            "dyn_feedback", [x, xhat, u, L], [rhs_feedback]
        )
        rhs_predictorRK4 = self.predictor_RK4(xhat, x, u, L)
        self.discrete_predfunc = ca.Function(
            "dyn_predictor", [xhat, x, u, L], [rhs_predictorRK4]
        )
        x_ref = ca.SX.sym("x_ref", 12)
        u_ref = ca.SX.sym("u_ref", 4)
        loss_k = self.loss_k(x, u, x_ref, u_ref)
        loss_N = self.loss_N(x, x_ref)
        self.lk = ca.Function("lkx", [x, u, x_ref, u_ref], [loss_k])
        self.lN = ca.Function("lku", [x, x_ref], [loss_N])

    def solve_OC_IFxkhat(self, x0, xref_seq, uref_seq):
        """
        Solve u with Multi-step Predicted State

        refk_seq: comming up reference trajectory: state_dim * H

        nonlinear program (NLP):
            min          F(x, p)
            x

            subject to
            LBX <=   x    <= UBX
            LBG <= G(x, p) <= UBG
            p  == P

            nx: number of decision variables
            ng: number of constraints
            np: number of parameters
        """

        w = []  # Solution
        w0 = []  # Init guess
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # Optimize N*corrector states+N*predicted states+N*inputs = 16*N parameters
        Xk_ = ca.DM(x0)  # The first corrector / NN input state
        L = self.L_init  # The first Feedback Gain
        # H = xref_seq.shape[1]
        # print(L @ (Xk_ - self.xhatk_seq[:, 0]))
        for k in range(self.H):
            # Add control param
            Uk = ca.MX.sym("U_" + str(k), self.model.u_dim)
            w += [Uk]
            w0 += list(uref_seq[:, k])
            lbw += self.u_lb
            ubw += self.u_ub

            # Add corrector / NN input state param and constraint (index + 1)
            Xk1 = ca.MX.sym("X_" + str(k + 1), self.model.ol_x_dim)
            w += [Xk1]
            w0 += list(xref_seq[:, k])
            lbw += self.x_lb
            ubw += self.x_ub

            # Add continous constraint on Feedback NN Dynamics
            g += [self.discrete_FBdynfunc(Xk_, self.xhatk_seq[:, k], Uk, L) - Xk1]
            lbg += [0] * self.model.ol_x_dim
            ubg += [0] * self.model.ol_x_dim

            # Add Obj
            if k == self.H - 1:
                J += self.lk(Xk_, Uk, xref_seq[:, k], uref_seq[:, k])
            else:
                J += self.lN(Xk_, xref_seq[:, k])

            # Refresh Xk_ (the corrector state)
            Xk_ = Xk1

            # Update Feedback Gain
            L = L * np.exp(-(k + 1) * self.decay_rate)

        # Build Prob
        prob = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
        # options = {}
        options = {
            "verbose": False,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
        }
        solver = ca.nlpsol("solver", "ipopt", prob, options)

        # Solve
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # Get control and state in mpc
        result = (
            sol["x"]
            .full()
            .flatten()
            .reshape(-1, self.model.ol_x_dim + self.model.u_dim)
        )
        result = result.T
        u_opt_seq = result[0 : self.model.u_dim, :]
        x_opt_seq = result[self.model.u_dim :, :]

        # Store the first-step predicted state before update
        self.xhat0 = self.xhatk_seq[:, 0]

        # Update predicted states
        xk_seq = np.hstack([x0.reshape((-1, 1)), x_opt_seq])[:, :-1]
        L = self.L_init
        for ii in range(xk_seq.shape[1]):
            xhatk1 = self.discrete_predfunc(
                self.xhatk_seq[:, ii], xk_seq[:, ii], u_opt_seq[:, ii], L
            )
            self.xhatk_seq[:, ii] = xhatk1.full().squeeze(axis=1)
            L = L * np.exp(-(ii + 1) * self.decay_rate)

        return u_opt_seq, x_opt_seq

    def solve_OC_MultiStepPred(self, x0, xref_seq, uref_seq):
        if self.BOOL_xhat:
            u_opt_seq, x_opt_seq = self.solve_OC_IFxkhat(x0, xref_seq, uref_seq)
            return u_opt_seq, x_opt_seq
        else:
            u_opt_seq, x_opt_seq = self.solve_OC(x0, xref_seq, uref_seq)
            self.xhat0 = x0
            self.xhatk_seq = np.hstack([x0.reshape((-1, 1)), x_opt_seq])[:, :-1]
            self.BOOL_xhat = True
            print("Init Predictor!")
            return u_opt_seq, x_opt_seq

    def refresh(self):
        self.BOOL_xhat = False

    def dx_modelpredict(self, xk, uk):
        dx = self.feedback_NNode(xk, self.xhat0, uk, self.L_init)
        return dx


class AdaptMPC:
    def __init__(self, model, discrete_h, H):
        self.model = model
        self.h = discrete_h
        self.H = H
        # Quadratic Objectives
        self.Q_k = 1e2 * np.diag(
            np.array([1.0] * 3 + [0.5] * 3 + [0.5] * 3 + [0.01] * 3)
        )
        self.R_k = np.diag(np.array([1.0] * 4))
        self.Q_f = self.Q_k
        # State Bound
        self.x_lb = [-ca.inf] * 6 + [-np.pi / 2] * 3 + [-ca.inf] * 3
        self.x_ub = [ca.inf] * 6 + [np.pi / 2] * 3 + [ca.inf] * 3
        # Input Bound
        self.u_lb = list(self.model.u_lb)
        self.u_ub = list(self.model.u_ub)

        # NLP Setup
        self.build_symbolic()

        # Adaptor Status
        self.BOOL_adapt = False

    def loss_kx(self, xk, xk_ref):
        return 0.5 * ca.transpose(xk - xk_ref) @ self.Q_k @ (xk - xk_ref)

    def loss_ku(self, uk, uk_ref):
        return 0.5 * ca.transpose(uk - uk_ref) @ self.R_k @ (uk - uk_ref)

    def loss_k(self, xk, uk, xk_ref, uk_ref):
        return self.loss_kx(xk, xk_ref) + self.loss_ku(uk, uk_ref)

    def loss_N(self, xf, xf_ref):
        return self.loss_kx(xf, xf_ref)

    def set_stateBoxCons(self, lb_list, ub_list):
        self.x_lb = lb_list
        self.x_ub = ub_list

    def set_inputBoxCons(self, lb_list, ub_list):
        self.u_lb = lb_list
        self.u_ub = ub_list

    def discrete_sys(self, xk, uk, FC_vec):
        h = self.h
        k1 = self.model.openloop_forCtrl(xk, uk, FC_vec)
        k2 = self.model.openloop_forCtrl((xk + 0.5 * h * k1), uk, FC_vec)
        k3 = self.model.openloop_forCtrl((xk + 0.5 * h * k2), uk, FC_vec)
        k4 = self.model.openloop_forCtrl((xk + h * k3), uk, FC_vec)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def build_symbolic(self):
        x = ca.SX.sym("x", 12)
        u = ca.SX.sym("u", 4)
        FC_vec = ca.SX.sym("FC_vec", self.model.output_size * self.model.hidden_size)
        rhs = self.discrete_sys(x, u, FC_vec)
        self.discrete_dynfunc = ca.Function("dyn", [x, u, FC_vec], [rhs])
        x_ref = ca.SX.sym("x_ref", 12)
        u_ref = ca.SX.sym("u_ref", 4)
        loss_k = self.loss_k(x, u, x_ref, u_ref)
        loss_N = self.loss_N(x, x_ref)
        self.lk = ca.Function("lkx", [x, u, x_ref, u_ref], [loss_k])
        self.lN = ca.Function("lku", [x, x_ref], [loss_N])

    def solve_OC_adapt(self, x0, xref_seq, uref_seq):
        """
        Solve u with Nominal State

        refk_seq: comming up reference trajectory: state_dim * H

        nonlinear program (NLP):
            min          F(x, p)
            x

            subject to
            LBX <=   x    <= UBX
            LBG <= G(x, p) <= UBG
            p  == P

            nx: number of decision variables
            ng: number of constraints
            np: number of parameters
        """

        w = []  # Solution
        w0 = []  # Init guess
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # Optimize N*states+N*inputs = 16*N parameters
        Xk_ = ca.DM(x0)  # The first states
        # H = xref_seq.shape[1]
        for k in range(self.H):
            # Add control param
            Uk = ca.MX.sym("U_" + str(k), self.model.u_dim)
            w += [Uk]
            w0 += list(uref_seq[:, k])
            lbw += self.u_lb
            ubw += self.u_ub

            # Add state param and constraint (index + 1)
            Xk1 = ca.MX.sym("X_" + str(k + 1), self.model.ol_x_dim)
            w += [Xk1]
            w0 += list(xref_seq[:, k])
            lbw += self.x_lb
            ubw += self.x_ub

            # Add continous constraint
            g += [self.discrete_dynfunc(Xk_, Uk, self.model.FC_vec) - Xk1]
            lbg += [0] * self.model.ol_x_dim
            ubg += [0] * self.model.ol_x_dim

            # Add Obj
            if k == self.H - 1:
                J += self.lk(Xk_, Uk, xref_seq[:, k], uref_seq[:, k])
            else:
                J += self.lN(Xk_, xref_seq[:, k])

            # Refresh Xk_
            Xk_ = Xk1

        # Build Prob
        prob = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
        # options = {}
        options = {
            "verbose": False,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
        }
        solver = ca.nlpsol("solver", "ipopt", prob, options)

        # Solve
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # Get control and state in mpc
        result = (
            sol["x"]
            .full()
            .flatten()
            .reshape(-1, self.model.ol_x_dim + self.model.u_dim)
        )
        result = result.T
        u_opt_seq = result[0 : self.model.u_dim, :]
        x_opt_seq = result[self.model.u_dim :, :]

        # Adapt model update
        if not self.BOOL_adapt:
            self.BOOL_adapt = True
            self.x_pred = x_opt_seq[3:6, 0]
        else:
            # adaptor update
            xk_in = x0[3:9]
            ek = x0[3:6] - self.x_pred
            # print(ek)
            self.model.adaptor_update(xk_in, ek, self.h)
            self.x_pred = x_opt_seq[3:6, 0]

        return u_opt_seq, x_opt_seq

    def refresh(self):
        self.BOOL_adapt = False

    def dx_modelpredict(self, xk, uk):
        dx = self.model.openloop_forCtrl(xk, uk, self.model.FC_vec)
        return dx


class AdaptMPC_MultiStep(AdaptMPC):
    def __init__(self, model, discrete_h, H):
        super(AdaptMPC_MultiStep, self).__init__(model, discrete_h, H)

        self.BOOL_xhat = False

    def set_GainAndDecay(self, L_init, decay_rate):
        self.L_init = L_init  # Initial Feedback Gain
        self.decay_rate = decay_rate

    def feedback_NNode(self, xk, xhatk, uk, L, FC_vec):
        # here xk is used for update the NN series / corrector states
        return self.model.openloop_forCtrl(xk, uk, FC_vec) + L @ (xk - xhatk)

    def feedback_NNode_RK4(self, xk, xhatk, uk, L, FC_vec):
        h = self.h
        k1 = self.feedback_NNode(xk, xhatk, uk, L, FC_vec)
        k2 = self.feedback_NNode((xk + 0.5 * h * k1), xhatk, uk, L, FC_vec)
        k3 = self.feedback_NNode((xk + 0.5 * h * k2), xhatk, uk, L, FC_vec)
        k4 = self.feedback_NNode((xk + h * k3), xhatk, uk, L, FC_vec)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def predictor_RK4(self, xhatk, xk, uk, L, FC_vec):
        h = self.h
        k1 = self.feedback_NNode(xk, xhatk, uk, L, FC_vec)
        k2 = self.feedback_NNode(xk, (xhatk + 0.5 * h * k1), uk, L, FC_vec)
        k3 = self.feedback_NNode(xk, (xhatk + 0.5 * h * k2), uk, L, FC_vec)
        k4 = self.feedback_NNode(xk, (xhatk + h * k3), uk, L, FC_vec)
        xhatk1 = xhatk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xhatk1

    def build_symbolic(self):
        x = ca.SX.sym("x", 12)
        xhat = ca.SX.sym("xhat", 12)
        u = ca.SX.sym("u", 4)
        L = ca.SX.sym("L", 12, 12)
        FC_vec = ca.SX.sym("FC_vec", self.model.output_size * self.model.hidden_size)
        rhs_openloop = self.discrete_sys(x, u, FC_vec)
        self.discrete_dynfunc = ca.Function("dyn", [x, u, FC_vec], [rhs_openloop])
        rhs_feedback = self.feedback_NNode_RK4(x, xhat, u, L, FC_vec)
        self.discrete_FBdynfunc = ca.Function(
            "dyn_feedback", [x, xhat, u, L, FC_vec], [rhs_feedback]
        )
        rhs_predictorRK4 = self.predictor_RK4(xhat, x, u, L, FC_vec)
        self.discrete_predfunc = ca.Function(
            "dyn_predictor", [xhat, x, u, L, FC_vec], [rhs_predictorRK4]
        )
        x_ref = ca.SX.sym("x_ref", 12)
        u_ref = ca.SX.sym("u_ref", 4)
        loss_k = self.loss_k(x, u, x_ref, u_ref)
        loss_N = self.loss_N(x, x_ref)
        self.lk = ca.Function("lkx", [x, u, x_ref, u_ref], [loss_k])
        self.lN = ca.Function("lku", [x, x_ref], [loss_N])

    def solve_OC_IFxkhat(self, x0, xref_seq, uref_seq):
        """
        Solve u with Multi-step Predicted State

        refk_seq: comming up reference trajectory: state_dim * H

        nonlinear program (NLP):
            min          F(x, p)
            x

            subject to
            LBX <=   x    <= UBX
            LBG <= G(x, p) <= UBG
            p  == P

            nx: number of decision variables
            ng: number of constraints
            np: number of parameters
        """

        w = []  # Solution
        w0 = []  # Init guess
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # Optimize N*corrector states+N*predicted states+N*inputs = 16*N parameters
        Xk_ = ca.DM(x0)  # The first corrector / NN input state
        L = self.L_init  # The first Feedback Gain
        # H = xref_seq.shape[1]
        # print(L @ (Xk_ - self.xhatk_seq[:, 0]))
        for k in range(self.H):
            # Add control param
            Uk = ca.MX.sym("U_" + str(k), self.model.u_dim)
            w += [Uk]
            w0 += list(uref_seq[:, k])
            lbw += self.u_lb
            ubw += self.u_ub

            # Add corrector / NN input state param and constraint (index + 1)
            Xk1 = ca.MX.sym("X_" + str(k + 1), self.model.ol_x_dim)
            w += [Xk1]
            w0 += list(xref_seq[:, k])
            lbw += self.x_lb
            ubw += self.x_ub

            # Add continous constraint on Feedback NN Dynamics
            g += [self.discrete_FBdynfunc(Xk_, self.xhatk_seq[:, k], Uk, L, self.model.FC_vec) - Xk1]
            lbg += [0] * self.model.ol_x_dim
            ubg += [0] * self.model.ol_x_dim

            # Add Obj
            if k == self.H - 1:
                J += self.lk(Xk_, Uk, xref_seq[:, k], uref_seq[:, k])
            else:
                J += self.lN(Xk_, xref_seq[:, k])

            # Refresh Xk_ (the corrector state)
            Xk_ = Xk1

            # Update Feedback Gain
            L = L * np.exp(-(k + 1) * self.decay_rate)

        # Build Prob
        prob = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}
        # options = {}
        options = {
            "verbose": False,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
        }
        solver = ca.nlpsol("solver", "ipopt", prob, options)

        # Solve
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # Get control and state in mpc
        result = (
            sol["x"]
            .full()
            .flatten()
            .reshape(-1, self.model.ol_x_dim + self.model.u_dim)
        )
        result = result.T
        u_opt_seq = result[0 : self.model.u_dim, :]
        x_opt_seq = result[self.model.u_dim :, :]

        # Adapt model update
        if not self.BOOL_adapt:
            self.BOOL_adapt = True
            self.x_pred = x_opt_seq[3:6, 0]
        else:
            # adaptor update
            xk_in = x0[3:9]
            ek = x0[3:6] - self.x_pred
            # print(ek)
            self.model.adaptor_update(xk_in, ek, self.h)
            self.x_pred = x_opt_seq[3:6, 0]
        
        # Store the first-step predicted state before update
        self.xhat0 = self.xhatk_seq[:, 0]

        # Update predicted states
        xk_seq = np.hstack([x0.reshape((-1, 1)), x_opt_seq])[:, :-1]
        L = self.L_init
        for ii in range(xk_seq.shape[1]):
            xhatk1 = self.discrete_predfunc(
                self.xhatk_seq[:, ii], xk_seq[:, ii], u_opt_seq[:, ii], L, self.model.FC_vec
            )
            self.xhatk_seq[:, ii] = xhatk1.full().squeeze(axis=1)
            L = L * np.exp(-(ii + 1) * self.decay_rate)

        return u_opt_seq, x_opt_seq

    def solve_OC_MultiStepPred(self, x0, xref_seq, uref_seq):
        if self.BOOL_xhat:
            u_opt_seq, x_opt_seq = self.solve_OC_IFxkhat(x0, xref_seq, uref_seq)
            return u_opt_seq, x_opt_seq
        else:
            u_opt_seq, x_opt_seq = self.solve_OC_adapt(x0, xref_seq, uref_seq)
            self.xhat0 = x0
            self.xhatk_seq = np.hstack([x0.reshape((-1, 1)), x_opt_seq])[:, :-1]
            self.BOOL_xhat = True
            print("Init Predictor!")
            return u_opt_seq, x_opt_seq

    def refresh(self):
        self.BOOL_xhat = False
        self.BOOL_adapt = False

    def dx_modelpredict(self, xk, uk):
        dx = self.feedback_NNode(xk, self.xhat0, uk, self.L_init, self.model.FC_vec)
        return dx
