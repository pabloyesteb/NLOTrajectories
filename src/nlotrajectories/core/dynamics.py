from abc import ABC, abstractmethod
from enum import Enum

import casadi as ca


class Dynamics(str, Enum):
    POINT_1ST = "point_1st"
    POINT_2ND = "point_2nd"
    UNICYCLE = "unicycle"


class IRobotDynamics(ABC):
    @abstractmethod
    def state_dim(self) -> int:
        """Returns the dimensionality of the state vector."""
        pass

    @abstractmethod
    def control_dim(self) -> int:
        """Returns the dimensionality of the control vector."""
        pass

    @abstractmethod
    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Returns symbolic expression for f(x, u) = dx/dt"""
        pass


class PointMass1stOrder(IRobotDynamics):
    def state_dim(self) -> int:
        return 2  # x, y

    def control_dim(self) -> int:
        return 2  # vx, vy

    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        return u  # dx/dt = u


class PointMass2ndOrder(IRobotDynamics):
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    def control_dim(self) -> int:
        return 2  # ax, ay

    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        vx = x[2]
        vy = x[3]
        ax = u[0]
        ay = u[1]
        return ca.vertcat(vx, vy, ax, ay)


class Unicycle(IRobotDynamics):
    def state_dim(self) -> int:
        return 3  # x, y, theta

    def control_dim(self) -> int:
        return 2  # v, omega

    def dynamics(self, x: ca.MX, u: ca.MX) -> ca.MX:
        theta = x[2]
        v = u[0]
        omega = u[1]
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = omega
        return ca.vertcat(dx, dy, dtheta)


DYNAMICS_CLASS_MAP = {
    Dynamics.POINT_1ST: PointMass1stOrder,
    Dynamics.POINT_2ND: PointMass2ndOrder,
    Dynamics.UNICYCLE: Unicycle,
}
