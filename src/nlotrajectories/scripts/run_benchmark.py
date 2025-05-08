import argparse
from pathlib import Path

import casadi as ca
import yaml

from nlotrajectories.core.config import Config
from nlotrajectories.core.runner import RunBenchmark
from nlotrajectories.core.visualizer import plot_trajectory


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_sdf_circle(center, radius):
    def sdf(x, y):
        dx = x - center[0]
        dy = y - center[1]
        return ca.sqrt(dx**2 + dy**2) - radius

    return sdf


def draw_obstacle(ax, config):
    from matplotlib.patches import Circle

    c = Circle(config.obstacles.root[0].center, config.obstacles.root[0].radius, color="r", alpha=0.5)
    ax.add_patch(c)


def run_benchmark(config_path: Path):
    config = Config(**load_config(config_path))
    obstacles = config.get_obstacles()

    x0 = ca.MX(config.body.start_state)
    x_goal = ca.MX(config.body.goal_state)
    geometry = config.body.create_geometry()

    runner = RunBenchmark(
        dynamics=config.body.create_dynamics(),
        geometry=geometry,
        x0=x0,
        x_goal=x_goal,
        N=config.solver.N,
        dt=config.solver.dt,
        sdf_func=obstacles.sdf,
        control_bounds=tuple(config.body.control_bounds),
        use_slack=config.solver.use_slack,
        slack_penalty=config.solver.slack_penalty,
    )

    X_opt, U_opt, _ = runner.run()
    plot_trajectory(X_opt, geometry, obstacles, title=config_path.stem, goal=config.body.goal_state)

    print("Optimization complete.")
    print("Final state:", X_opt[:, -1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark YAML config")
    args = parser.parse_args()

    run_benchmark(Path(args.config))


if __name__ == "__main__":
    main()
