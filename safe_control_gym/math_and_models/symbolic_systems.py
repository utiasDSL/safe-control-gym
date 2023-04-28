'''Symbolic Models.'''

import casadi as cs


class SymbolicModel():
    '''Implements the dynamics model with symbolic variables.

    x_dot = f(x,u), y = g(x,u), with other pre-defined, symbolic functions
    (e.g. cost, constraints), serve as priors for the controllers.

    Notes:
        * naming convention on symbolic variable and functions.
            * for single-letter symbol, use {}_sym, otherwise use underscore for delimiter.
            * for symbolic functions to be exposed, use {}_func.
    '''

    def __init__(self,
                 dynamics,
                 cost,
                 dt=1e-3,
                 integration_algo='cvodes',
                 funcs=None,
                 params=None,
                 ):
        # Setup for dynamics.
        self.x_sym = dynamics['vars']['X']
        self.u_sym = dynamics['vars']['U']
        self.x_dot = dynamics['dyn_eqn']
        if dynamics['obs_eqn'] is None:
            self.y_sym = self.x_sym
        else:
            self.y_sym = dynamics['obs_eqn']
        # Sampling time.
        self.dt = dt
        # Integration algorithm.
        self.integration_algo = integration_algo
        # Other symbolic functions.
        if funcs is not None:
            for name, func in funcs.items():
                assert name not in self.__dict__
                self.__dict__[name] = func
        # Cache other parameters, for example, X_EQ and U_EQ
        # that would be used in the controller via the symbolic model
        if params is not None:
            for name, param in params.items():
                assert name not in self.__dict__
                self.__dict__[name] = param
        # Variable dimensions.
        self.nx = self.x_sym.shape[0]
        self.nu = self.u_sym.shape[0]
        self.ny = self.y_sym.shape[0]
        # Setup cost function.
        self.cost_func = cost['cost_func']
        # print(self.cost_func)
        self.Q = cost['vars']['Q']
        self.R = cost['vars']['R']
        self.Xr = cost['vars']['Xr']
        self.Ur = cost['vars']['Ur']
        # Setup symbolic model.
        self.setup_model()
        # Setup Jacobian and Hessian of the dynamics and cost functions.
        self.setup_linearization()

    def setup_model(self):
        '''Exposes functions to evaluate the model.'''
        # Continuous time dynamics.
        self.fc_func = cs.Function('fc', [self.x_sym, self.u_sym], [self.x_dot], ['x', 'u'], ['f'])
        # Discrete time dynamics.
        self.fd_func = cs.integrator('fd', self.integration_algo, {'x': self.x_sym,
                                                                   'p': self.u_sym,
                                                                   'ode': self.x_dot}, {'tf': self.dt}
                                     )
        # Observation model.
        self.g_func = cs.Function('g', [self.x_sym, self.u_sym], [self.y_sym], ['x', 'u'], ['g'])

    def setup_linearization(self):
        '''Exposes functions for the linearized model.'''
        # Jacobians w.r.t state & input.
        self.dfdx = cs.jacobian(self.x_dot, self.x_sym)
        self.dfdu = cs.jacobian(self.x_dot, self.u_sym)
        self.df_func = cs.Function('df', [self.x_sym, self.u_sym],
                                   [self.dfdx, self.dfdu], ['x', 'u'],
                                   ['dfdx', 'dfdu'])
        self.dgdx = cs.jacobian(self.y_sym, self.x_sym)
        self.dgdu = cs.jacobian(self.y_sym, self.u_sym)
        self.dg_func = cs.Function('dg', [self.x_sym, self.u_sym],
                                   [self.dgdx, self.dgdu], ['x', 'u'],
                                   ['dgdx', 'dgdu'])
        # Evaluation point for linearization.
        self.x_eval = cs.MX.sym('x_eval', self.nx, 1)
        self.u_eval = cs.MX.sym('u_eval', self.nu, 1)
        # Linearized dynamics model.
        self.x_dot_linear = self.x_dot + self.dfdx @ (
            self.x_eval - self.x_sym) + self.dfdu @ (self.u_eval - self.u_sym)
        self.fc_linear_func = cs.Function(
            'fc', [self.x_eval, self.u_eval, self.x_sym, self.u_sym],
            [self.x_dot_linear], ['x_eval', 'u_eval', 'x', 'u'], ['f_linear'])
        self.fd_linear_func = cs.integrator(
            'fd_linear', self.integration_algo, {
                'x': self.x_eval,
                'p': cs.vertcat(self.u_eval, self.x_sym, self.u_sym),
                'ode': self.x_dot_linear
            }, {'tf': self.dt})
        # Linearized observation model.
        self.y_linear = self.y_sym + self.dgdx @ (
            self.x_eval - self.x_sym) + self.dgdu @ (self.u_eval - self.u_sym)
        self.g_linear_func = cs.Function(
            'g_linear', [self.x_eval, self.u_eval, self.x_sym, self.u_sym],
            [self.y_linear], ['x_eval', 'u_eval', 'x', 'u'], ['g_linear'])
        # Jacobian and Hessian of cost function.
        self.l_x = cs.jacobian(self.cost_func, self.x_sym)
        self.l_xx = cs.jacobian(self.l_x, self.x_sym)
        self.l_u = cs.jacobian(self.cost_func, self.u_sym)
        self.l_uu = cs.jacobian(self.l_u, self.u_sym)
        self.l_xu = cs.jacobian(self.l_x, self.u_sym)
        l_inputs = [self.x_sym, self.u_sym, self.Xr, self.Ur, self.Q, self.R]
        l_inputs_str = ['x', 'u', 'Xr', 'Ur', 'Q', 'R']
        l_outputs = [self.cost_func, self.l_x, self.l_xx, self.l_u, self.l_uu, self.l_xu]
        l_outputs_str = ['l', 'l_x', 'l_xx', 'l_u', 'l_uu', 'l_xu']
        self.loss = cs.Function('loss', l_inputs, l_outputs, l_inputs_str, l_outputs_str)
