from abc import ABC, abstractmethod
from enum import Enum

import casadi as ca
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.patches import Polygon

from nlotrajectories.core.utils import soft_min


class GoalMode(str, Enum):
    CENTER = "center"
    ANY_POINT = "any_point"


class Shape(str, Enum):
    DOT = "dot"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"


class IRobotGeometry(ABC):
    @abstractmethod
    def transform(self, pose: ca.MX) -> list[tuple[ca.MX, ca.MX]]:
        """
        Returns a list of (x, y) points in world coordinates based on the current robot pose.

        Parameters:
            pose (ca.MX): A symbolic vector [x, y, theta] representing the robot pose

        Returns:
            list[tuple[ca.MX, ca.MX]]: List of symbolic (x, y) points defining the robot shape
        """
        pass

    @abstractmethod
    def sdf_constraints(
        self, opti: ca.Opti, pose: ca.MX, sdf_func: callable, margin: float, slack: ca.MX | None = None
    ) -> None:
        """
        Adds symbolic obstacle avoidance constraints to the optimization problem.

        Parameters:
            opti (ca.Opti): The CasADi optimization object
            pose (ca.MX): The symbolic robot pose [x, y, theta]
            sdf_func (Callable): A symbolic function sdf(x, y) returning signed distance
            margin (float): Minimum required clearance from obstacles
            slack (ca.MX | None): Optional symbolic relaxation variable for soft constraints. Default: None
        """
        pass

    @abstractmethod
    def goal_cost(self, pose: ca.MX, goal_vec: ca.MX) -> ca.MX:
        """
        Computes the symbolic cost for reaching the goal (can use center, any point, or full shape).

        Parameters:
            pose (ca.MX): The symbolic robot pose [x, y, theta]
            goal_vec (ca.MX): Symbolic vector of goal position [x, y]

        Returns:
            ca.MX: A symbolic expression representing the goal-reaching cost
        """
        pass

    @abstractmethod
    def draw(self, ax: Axes, pose: ca.MX) -> None:
        """Matplotlib drawing using pose (x, y, theta if available)"""
        pass


class DotGeometry(IRobotGeometry):
    def transform(self, pose: ca.MX) -> list[tuple[ca.MX, ca.MX]]:
        return [(pose[0], pose[1])]

    def sdf_constraints(
        self, opti: ca.Opti, pose: ca.MX, sdf_func: callable, margin: float, slack: ca.MX | None = None
    ) -> None:
        x, y = pose[0], pose[1]
        opti.subject_to(sdf_func(x, y) >= margin)

    def goal_cost(self, pose: ca.MX, goal_vec: ca.MX) -> ca.MX:
        return ca.sumsqr(pose[0:2] - goal_vec)

    def draw(self, ax: Axes, pose: ca.MX) -> None:
        ax.plot(pose[0], pose[1], "bo")


class PolygonGeometry(IRobotGeometry):
    def __init__(self, body_points: list[tuple[float, float]], goal_mode: GoalMode = GoalMode.CENTER):
        self.body_points = body_points
        self.goal_mode = goal_mode

    def transform(self, pose: ca.MX) -> list[tuple[ca.MX, ca.MX]]:
        x, y, theta = pose[0], pose[1], pose[2]
        return [
            (x + ca.cos(theta) * px - ca.sin(theta) * py, y + ca.sin(theta) * px + ca.cos(theta) * py)
            for px, py in self.body_points
        ]

    def densify_polygon(
        self, corner_points: list[tuple[ca.MX, ca.MX]], num_points: int = 0
    ) -> list[tuple[ca.MX, ca.MX]]:
        """
        Insert `num_points` evenly spaced points along each edge of the polygon.
        """
        dense_points = []

        n = len(corner_points)
        for i in range(n):
            p0 = np.array(corner_points[i])
            p1 = np.array(corner_points[(i + 1) % n])

            dense_points.append(corner_points[i])

            for j in range(1, num_points + 1):
                t = j / (num_points + 1)
                intermediate = (1 - t) * p0 + t * p1
                dense_points.append(tuple(intermediate))

        return dense_points

    def sdf_constraints(
        self, opti: ca.Opti, pose: ca.MX, sdf_func: callable, margin: float, slack: ca.MX | None = None
    ) -> None:
        corner_points = self.transform(pose)
        dense_points = self.densify_polygon(corner_points)
        if slack is None:
            for px, py in dense_points:
                opti.subject_to(sdf_func(px, py) >= margin)
        else:
            min_dist = soft_min([sdf_func(px, py) for px, py in dense_points])
            opti.subject_to(min_dist + slack >= margin)

    def goal_cost(self, pose: ca.MX, goal_vec: ca.MX) -> ca.MX:
        shape_points = self.transform(pose)
        if self.goal_mode == GoalMode.CENTER:
            return ca.sumsqr(pose[0:2] - goal_vec)
        return ca.mmin(ca.vertcat(*[ca.sumsqr(ca.vertcat(px, py) - goal_vec) for px, py in shape_points]))

    def draw(self, ax: Axes, pose: ca.MX) -> None:
        shape_points = self.transform(pose)
        polygon = Polygon(shape_points, closed=True, edgecolor="black", facecolor="lightblue")
        ax.add_patch(polygon)


class RectangleGeometry(PolygonGeometry):
    def __init__(self, length: float, width: float, goal_mode: GoalMode = GoalMode.CENTER):
        length_bias = length / 2
        width_bias = width / 2
        body_points = [
            (-length_bias, -width_bias),
            (-length_bias, width_bias),
            (length_bias, width_bias),
            (length_bias, -width_bias),
        ]
        super().__init__(body_points, goal_mode)


class TriangleGeometry(PolygonGeometry):
    def __init__(self, length: float, width: float, goal_mode: GoalMode = GoalMode.CENTER):
        tip = (length / 2, 0.0)
        base_left = (-length / 2, -width / 2)
        base_right = (-length / 2, width / 2)
        body_points = [tip, base_right, base_left]
        super().__init__(body_points, goal_mode)
