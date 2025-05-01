# ---------- Problem Description (Multiple Shooting) ----------
# Goal:
#   Plan an optimal trajectory for a point mass in 2D space moving from a start to goal position.
#   The trajectory must avoid multiple circular obstacles and obey acceleration constraints.
#
# State Variables (X):
#   Each state X_k = [x, y, vx, vy] at timestep k
# Control Variables (U):
#   Each control U_k = [ax, ay] represents acceleration
# Dynamics (discretized using forward Euler):
#   X[k+1] = X[k] + dt * f(X[k], U[k])
#
# Multiple Shooting:
#   In this method, all state variables X[:, 0...N] are decision variables.
#   Dynamics are not simulated forward from x0. Instead, we impose the dynamics as equality constraints:
#     X[:, k+1] == X[:, k] + dt * f(X[:, k], U[:, k])
#   This allows direct access to all intermediate states for constraint or cost formulation.
#
# Objective:
#   Minimize total path length over the trajectory
# Constraints:
#   - Initial and final state match specified values
#   - Dynamics enforced explicitly as equality constraints
#   - Velocity and control limits
#   - Obstacle avoidance using circle exclusion constraints

import casadi as ca
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# ---------- Problem Setup ----------
N = 100  # Number of time steps
dt = 0.1  # Time step size (s)

x0 = np.array([0.0, 0.0, 0.0, 0.0])  # Initial state: [x, y, vx, vy]
xT = np.array([1.0, 1.0, 0.0, 0.0])  # Target state: [x, y, vx, vy]

# Define circular obstacles (center, radius)
obs_centers = [np.array([0.5, 0.5]), np.array([0.45, 0.9]), np.array([0.9, 0.6])]
obs_radii = [0.2, 0.1, 0.12]

# ---------- CasADi Optimization ----------
opti = ca.Opti()

X = opti.variable(4, N + 1)  # States: [x, y, vx, vy] at each time step
U = opti.variable(2, N)  # Controls: [ax, ay] at each time step

# Initial and terminal constraints
opti.subject_to(X[:, 0] == x0)  # Enforce initial state
opti.subject_to(X[:, -1] == xT)  # Enforce terminal state

# ---------- Dynamics constraints (Euler integration) ----------
# Enforce physics step-by-step between state and control variables
for k in range(N):
    x_k = X[:, k]  # state at step k
    u_k = U[:, k]  # control at step k
    x_next = X[:, k + 1]  # state at step k+1

    # Define dynamics function f = [vx, vy, ax, ay]
    f = ca.vertcat(x_k[2], x_k[3], u_k[0], u_k[1])

    # Enforce the Euler integration rule
    opti.subject_to(x_next == x_k + dt * f)

# ---------- Obstacle avoidance constraints ----------
# Ensure (x, y) stays outside each circular obstacle
for cx, r in zip(obs_centers, obs_radii):
    for k in range(N + 1):
        pos = X[0:2, k]  # extract position [x, y] at time k
        dist_sq = (pos[0] - cx[0]) ** 2 + (pos[1] - cx[1]) ** 2
        opti.subject_to(dist_sq >= r**2)  # stay outside circle

# ---------- Control input bounds ----------
# Limit acceleration in both x and y directions
opti.subject_to(opti.bounded(-0.3, U, 0.3))

# ---------- Objective Function: Minimize Path Length ----------
# Approximate trajectory length by summing segment distances
epsilon = 1e-6  # small term to avoid sqrt(0)
cost = 0
for k in range(N):
    dx = X[0, k + 1] - X[0, k]
    dy = X[1, k + 1] - X[1, k]
    cost += ca.sqrt(dx**2 + dy**2 + epsilon)
opti.minimize(cost)

# ---------- Solve Optimization Problem ----------
opti.solver("ipopt")
sol = opti.solve()

# ---------- Extract Results ----------
x_opt = sol.value(X)
u_opt = sol.value(U)

x_vals = x_opt[0, :]
y_vals = x_opt[1, :]
vx_vals = x_opt[2, :]
vy_vals = x_opt[3, :]

# ---------- Animation ----------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(-0.1, 1.2)
ax.set_aspect("equal")
ax.set_title("2D Trajectory with Obstacles and Velocity Vectors")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Draw Obstacles
for cx, r in zip(obs_centers, obs_radii):
    circle = plt.Circle(cx, r, color="red", alpha=0.3, label="Obstacle")
    ax.add_patch(circle)

# Draw Start and Goal
ax.plot(x0[0], x0[1], "go", label="Start")
ax.plot(xT[0], xT[1], "ro", label="Goal")

# Animation Elements
(point,) = ax.plot([], [], "bo", label="Current Position")
(path,) = ax.plot([], [], "b--", alpha=0.5)
velocity_quiver = ax.quiver([], [], [], [], color="green", scale=1, label="Velocity")
speed_text = ax.text(
    0.02,
    1.05,
    "",
    transform=ax.transAxes,
    fontsize=12,
    color="purple",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)


# ---------- Initialize Animation Elements ----------
def init():
    point.set_data([], [])
    path.set_data([], [])
    velocity_quiver.set_UVC([], [])
    speed_text.set_text("")
    return point, path, velocity_quiver, speed_text


# ---------- Update Function per Frame ----------
def update(frame):
    if frame < len(x_vals):
        x = x_vals[frame]
        y = y_vals[frame]
        vx = vx_vals[frame]
        vy = vy_vals[frame]

        point.set_data([x], [y])
        path.set_data(x_vals[: frame + 1], y_vals[: frame + 1])
        velocity_quiver.set_offsets([[x, y]])
        velocity_quiver.set_UVC([vx], [vy])

        speed = np.sqrt(vx**2 + vy**2)
        speed_text.set_text(f"Speed: {speed:.2f} m/s")

    return point, path, velocity_quiver, speed_text


# ---------- Generate and Save Animation ----------
ani = animation.FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=True, interval=100)
ani.save("Multiple_Shooting_Trajectory_with_Velocity.gif", writer="pillow", fps=10)

plt.legend()
plt.show()
