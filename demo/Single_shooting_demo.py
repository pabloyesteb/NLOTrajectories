# ---------- Problem Description ----------
# We aim to compute an optimal trajectory for a point mass moving in 2D space.
# The system follows Newtonian mechanics and moves under the influence of control accelerations.
#
# ----------------------------- System Dynamics -----------------------------
# Forward Euler integration is used:
#   x[k+1]  = x[k]  + dt * vx[k]
#   y[k+1]  = y[k]  + dt * vy[k]
#   vx[k+1] = vx[k] + dt * ax[k]
#   vy[k+1] = vy[k] + dt * ay[k]
# where the control inputs at each timestep are accelerations (ax[k], ay[k]).
#
# ------------------------------- Objective --------------------------------
# Minimize the total trajectory length over the time horizon:
#   J = sum_{k=0}^{N-1} sqrt((x[k+1] - x[k])^2 + (y[k+1] - y[k])^2)
#
# ------------------------------- Constraints -------------------------------
# - System dynamics via forward Euler integration
# - Control constraints: ||[ax[k], ay[k]]||_2 <= a_max
# - Obstacle avoidance: Circular obstacle at (cx, cy) with radius r:
#     (x[k] - cx)^2 + (y[k] - cy)^2 >= r^2 for all k
# - Initial and final state must match given values
#
# ------------------------- Optimization Strategy ---------------------------
# We use the **Single Shooting** method to solve this trajectory optimization:
#
# - Only the control sequence U = [ax, ay] for all timesteps is defined as a decision variable.
# - The full trajectory X = [x, y, vx, vy] is not explicitly optimized.
# - Instead, X is generated **implicitly** by simulating forward the system dynamics
#   starting from the known initial state, using the chosen U.
#
# This makes the optimization smaller (fewer variables) and conceptually simpler.
# However, it can be more sensitive to the initial guess and less robust for complex dynamics.
#
# ----------------------------------------------------------------------------

import casadi as ca
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# ---------- Problem Setup ----------
N = 100  # Number of time steps
dt = 0.1  # Time step size (s)

x0_val = np.array([0.0, 0.0, 0.0, 0.0])  # Initial state: [x, y, vx, vy]
xT_val = np.array([1.0, 1.0, 0.0, 0.0])  # Target state: [x, y, vx, vy]

# Define circular obstacles (center, radius)
obs_centers = [np.array([0.5, 0.5]), np.array([0.45, 0.9]), np.array([0.9, 0.6])]
obs_radii = [0.2, 0.1, 0.12]

# ---------- CasADi Optimization Problem ----------
opti = ca.Opti()

U = opti.variable(2, N)  # Control variables [ax, ay] for each time step
x0 = ca.MX(x0_val)  # Fixed initial state


# ---------- Trajectory Simulation Function (Single Shooting) ----------
def simulate_trajectory(x0, U, dt, N):
    X = [x0]
    x = x0
    for k in range(N):
        u = U[:, k]
        f = ca.vertcat(x[2], x[3], u[0], u[1])  # [dx, dy, dvx, dvy]
        x = x + dt * f
        for cx, r in zip(obs_centers, obs_radii):
            dist_sq = (x[0] - cx[0]) ** 2 + (x[1] - cx[1]) ** 2
            opti.subject_to(dist_sq >= r**2)  # Obstacle avoidance constraint
        X.append(x)
    return ca.horzcat(*X)


# ---------- Simulate Full Trajectory ----------
X_sim = simulate_trajectory(x0, U, dt, N)

# ---------- Terminal Constraint ----------
opti.subject_to(X_sim[:, -1] == xT_val)

# ---------- Control input bounds ----------
# Limit acceleration in both x and y directions
opti.subject_to(opti.bounded(-0.3, U, 0.3))

# ---------- Objective: Minimize Path Length ----------
epsilon = 1e-6
cost = 0
for k in range(N):
    dx = X_sim[0, k + 1] - X_sim[0, k]
    dy = X_sim[1, k + 1] - X_sim[1, k]
    cost += ca.sqrt(dx**2 + dy**2 + epsilon)
opti.minimize(cost)

# ---------- Solve Optimization Problem ----------
opti.solver("ipopt")
sol = opti.solve()

# ---------- Extract Solution ----------
x_opt = sol.value(X_sim)
u_opt = sol.value(U)

x_vals = x_opt[0, :]
y_vals = x_opt[1, :]
vx_vals = x_opt[2, :]
vy_vals = x_opt[3, :]

# ---------- Visualization using Animation ----------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(-0.1, 1.2)
ax.set_aspect("equal")
ax.set_title("2D Trajectory with Obstacle and Velocity Vectors")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Draw Obstacle
for cx, r in zip(obs_centers, obs_radii):
    circle = plt.Circle(cx, r, color="red", alpha=0.3, label="Obstacle")
    ax.add_patch(circle)


# Start and Goal Points
ax.plot(x0_val[0], x0_val[1], "go", label="Start")
ax.plot(xT_val[0], xT_val[1], "ro", label="Goal")

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


# Initialize Animation Elements
def init():
    point.set_data([], [])
    path.set_data([], [])
    velocity_quiver.set_UVC([], [])
    speed_text.set_text("")
    return point, path, velocity_quiver, speed_text


# Update Function per Frame
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


# Create and Save Animation
ani = animation.FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=True, interval=100)
ani.save("Single_Shooting_Trajectory_with_Velocity.gif", writer="pillow", fps=10)

plt.legend()
plt.show()
