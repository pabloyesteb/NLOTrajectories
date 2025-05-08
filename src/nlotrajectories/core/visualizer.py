from pathlib import Path

import matplotlib.pyplot as plt


def plot_trajectory(X_opt, geometry, obstacles, title="Trajectory", goal=None):
    path = Path("result")
    path.mkdir(parents=True, exist_ok=True)

    _, ax = plt.subplots()
    N = X_opt.shape[1]

    for k in range(N):
        pose_k = X_opt[:, k]
        geometry.draw(ax, pose_k)

    obstacles.draw(ax, alpha=0.7, color="r")

    if goal is not None:
        ax.plot(goal[0], goal[1], "ro", label="Goal")

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)
    plt.legend()
    plt.savefig(path / f"{title}.png", bbox_inches="tight")
