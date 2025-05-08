from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
from matplotlib.patches import Circle, Rectangle

from nlotrajectories.core.utils import soft_abs, soft_max, soft_min


class IObstacle(ABC):
    @abstractmethod
    def sdf(self, x: float, y: float) -> float:
        pass

    @abstractmethod
    def draw(self, ax, **kwargs) -> None:
        pass


class CircleObstacle(IObstacle):
    def __init__(self, center: tuple[float, float], radius: float, margin: float = 0.0):
        self.center = np.array(center)
        self.radius = radius
        self.margin = margin

    def sdf(self, x: ca.MX, y: ca.MX) -> float:
        p = np.array([x, y])
        return np.linalg.norm(p - self.center) - (self.radius + self.margin)

    def draw(self, ax, **kwargs) -> None:
        circle = Circle(self.center, self.radius, **kwargs)
        ax.add_patch(circle)


class SquareObstacle(IObstacle):
    def __init__(self, center: tuple[float, float], size: float, margin: float = 0.0):
        self.center = np.array(center)
        self.size = size
        self.margin = margin

    def sdf(self, x: ca.MX, y: ca.MX) -> float:
        p = ca.vertcat(*[x, y]) - self.center
        half = self.size / 2 + self.margin
        d = soft_abs(p) - half
        outside = soft_max([d, 0])
        inside = soft_min([soft_max([d[0], d[1]]), 0])
        return ca.norm_2(outside) + inside

    def draw(self, ax, **kwargs) -> None:
        half = self.size / 2
        lower_left = self.center - half
        rect = Rectangle(lower_left, 2 * half, 2 * half, **kwargs)
        ax.add_patch(rect)


class MultiObstacle(IObstacle):
    def __init__(self, obstacles: list[IObstacle]):
        self.obstacles = obstacles

    def sdf(self, x: ca.MX, y: ca.MX) -> float:
        return soft_min([obs.sdf(x, y) for obs in self.obstacles])

    def draw(self, ax, **kwargs) -> None:
        for obs in self.obstacles:
            obs.draw(ax, **kwargs)
