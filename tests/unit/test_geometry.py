import casadi as ca
import pytest

from nlotrajectories.core.geometry import GoalMode, RectangleGeometry


@pytest.fixture
def pose():
    return ca.MX([1.0, 2.0, ca.pi / 4])


@pytest.fixture
def goal():
    return ca.MX([1.5, 2.5])


@pytest.fixture
def sdf_func():
    def dummy_sdf(x, y):
        return ca.sqrt((x - 0.0) ** 2 + (y - 0.0) ** 2) - 1.0

    return dummy_sdf


class TestRectangleGeometry:
    @pytest.mark.parametrize("goal_mode", [GoalMode.CENTER, GoalMode.ANY_POINT])
    def test_transform_and_goal_cost(self, pose, goal, goal_mode):
        geometry = RectangleGeometry(length=1.0, width=0.5, goal_mode=goal_mode)
        pts = geometry.transform(pose)
        assert len(pts) == 4
        for pt in pts:
            assert isinstance(pt[0], ca.MX) and isinstance(pt[1], ca.MX)

        cost = geometry.goal_cost(pose, goal)
        assert isinstance(cost, ca.MX)
