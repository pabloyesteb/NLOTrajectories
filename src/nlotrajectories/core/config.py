from typing import Literal

from pydantic import BaseModel, Field, RootModel

from nlotrajectories.core.dynamics import DYNAMICS_CLASS_MAP, Dynamics, IRobotDynamics
from nlotrajectories.core.geometry import (
    DotGeometry,
    GoalMode,
    RectangleGeometry,
    Shape,
    TriangleGeometry,
)
from nlotrajectories.core.sdf.casadi import (
    CircleObstacle,
    IObstacle,
    MultiObstacle,
    SquareObstacle,
)


class BodyConfig(BaseModel):
    shape: Shape
    dynamic: Dynamics
    goal_mode: GoalMode = GoalMode.CENTER
    length: float | None = None
    width: float | None = None
    start_state: list[float]
    goal_state: list[float]
    control_bounds: list[float]

    def create_geometry(self):
        if self.shape == Shape.DOT:
            return DotGeometry()
        if self.shape == Shape.RECTANGLE:
            return RectangleGeometry(self.length, self.width, self.goal_mode)
        return TriangleGeometry(self.length, self.width, self.goal_mode)

    def create_dynamics(self) -> IRobotDynamics:
        cls = DYNAMICS_CLASS_MAP.get(self.dynamic)
        if cls is None:
            raise ValueError(f"Unknown dynamics type: {self.dynamic}")
        return cls()


class CircleObstacleConfig(BaseModel):
    type: Literal["circle"]
    center: tuple[float, float]
    radius: float
    margin: float = 0.0

    def to_obstacle(self) -> IObstacle:
        return CircleObstacle(center=self.center, radius=self.radius, margin=self.margin)


class SquareObstacleConfig(BaseModel):
    type: Literal["square"]
    center: tuple[float, float]
    size: float
    margin: float = 0.0

    def to_obstacle(self) -> IObstacle:
        return SquareObstacle(center=self.center, size=self.size, margin=self.margin)


ObstacleConfigs = list[CircleObstacleConfig | SquareObstacleConfig]


class ObstacleConfig(RootModel[ObstacleConfigs]):
    model_config = {"arbitrary_types_allowed": True, "smart_union": True, "discriminator": "type"}

    def to_obstacles(self) -> MultiObstacle:
        return MultiObstacle([ob.to_obstacle() for ob in self.root])


class SolverConfig(BaseModel):
    N: int = Field(default=20, ge=1, description="Number of steps")
    dt: float = 0.1
    use_slack: bool = False
    slack_penalty: float | None = Field(default=1000, ge=1)
    mode: Literal["casadi"]


class Config(BaseModel):
    body: BodyConfig
    obstacles: ObstacleConfig
    solver: SolverConfig

    def get_obstacles(self) -> MultiObstacle:
        return self.obstacles.to_obstacles()
