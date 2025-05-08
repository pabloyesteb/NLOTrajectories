import casadi as ca

from nlotrajectories.core.dynamics import IRobotDynamics
from nlotrajectories.core.geometry import IRobotGeometry


class RunBenchmark:
    def __init__(
        self,
        dynamics: IRobotDynamics,
        geometry: IRobotGeometry,
        x0: ca.MX,
        x_goal: ca.MX,
        N: int,
        dt: float,
        sdf_func: callable,
        control_bounds: tuple = (-1.0, 1.0),
        use_slack: bool = False,
        slack_penalty: float = 1000,
    ):
        self.dynamics = dynamics
        self.geometry = geometry
        self.x0 = x0
        self.x_goal = x_goal
        self.N = N
        self.dt = dt
        self.sdf_func = sdf_func
        self.control_bounds = control_bounds
        self.use_slack = use_slack
        self.slack_penalty = slack_penalty

    def run(self):
        opti = ca.Opti()
        X = opti.variable(self.dynamics.state_dim(), self.N + 1)
        U = opti.variable(self.dynamics.control_dim(), self.N)

        # Initial condition
        opti.subject_to(X[:, 0] == self.x0)

        # Dynamics constraints
        for k in range(self.N):
            x_k = X[:, k]
            u_k = U[:, k]
            f_k = self.dynamics.dynamics(x_k, u_k)
            x_next = x_k + self.dt * f_k
            opti.subject_to(X[:, k + 1] == x_next)

        # Obstacle avoidance
        if self.use_slack:
            slack = opti.variable(1, self.N + 1)
            opti.subject_to(slack >= 0)
        else:
            slack = None

        for k in range(self.N + 1):
            pose_k = X[:, k]
            self.geometry.sdf_constraints(
                opti, pose_k, self.sdf_func, margin=0.0, slack=slack[:, k] if slack is not None else None
            )

        # Cost: minimize distance to goal
        goal_cost = 0
        for k in range(self.N + 1):
            goal_cost += self.geometry.goal_cost(X[:, k], self.x_goal)

        total_cost = goal_cost
        if self.use_slack:
            total_cost += self.slack_penalty * ca.sumsqr(slack)

        opti.minimize(total_cost)

        # Control bounds
        umin, umax = self.control_bounds
        opti.subject_to(opti.bounded(umin, U, umax))

        # Solver
        opti.solver("ipopt")
        sol = opti.solve()

        X_opt = sol.value(X)
        U_opt = sol.value(U)

        return X_opt, U_opt, opti
