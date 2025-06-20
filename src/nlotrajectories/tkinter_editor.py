import datetime
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Rectangle
from scripts.run_benchmark import run_benchmark


class SimplePlannerUI:
    MAX_OBSTACLES = 5

    def __init__(self, root):
        self.root = root
        self.root.title("2D Trajectory Planner")

        self.start = [0.1, 0.1]
        self.goal = [0.9, 0.9]
        self.obstacles = []
        self.trajectory = []
        self.change_config = True
        self.optimized_trajectory = None

        self.setup_controls()

        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, pady=5)
        self.save_button = tk.Button(button_frame, text="Save Yaml", command=self.save_yaml)
        self.config_button = tk.Button(button_frame, text="Run Optimizer", command=self.run_optimizer)
        self.clear_button = tk.Button(button_frame, text="Clear Trajectories", command=self.clear_trajectories)
        self.config_button.pack(side=tk.LEFT, padx=5)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.setup_plot()

    def setup_controls(self):
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, fill=tk.Y)

        # Start and goal sliders
        tk.Label(frame, text="Start X").pack()
        self.start_x = tk.Scale(frame, from_=0, to=1, resolution=0.01,
                                orient=tk.HORIZONTAL, command=self.update_start)
        self.start_x.set(self.start[0])
        self.start_x.pack()

        tk.Label(frame, text="Start Y").pack()
        self.start_y = tk.Scale(frame, from_=0, to=1, resolution=0.01,
                                orient=tk.HORIZONTAL, command=self.update_start)
        self.start_y.set(self.start[1])
        self.start_y.pack()

        tk.Label(frame, text="Goal X").pack()
        self.goal_x = tk.Scale(frame, from_=0, to=1, resolution=0.01,
                               orient=tk.HORIZONTAL, command=self.update_goal)
        self.goal_x.set(self.goal[0])
        self.goal_x.pack()

        tk.Label(frame, text="Goal Y").pack()
        self.goal_y = tk.Scale(frame, from_=0, to=1, resolution=0.01,
                               orient=tk.HORIZONTAL, command=self.update_goal)
        self.goal_y.set(self.goal[1])
        self.goal_y.pack()

        # Obstacle inputs
        tk.Label(frame, text="Obstacle X").pack()
        self.ob_x = tk.Entry(frame)
        self.ob_x.insert(0, "0.5")
        self.ob_x.pack()

        tk.Label(frame, text="Obstacle Y").pack()
        self.ob_y = tk.Entry(frame)
        self.ob_y.insert(0, "0.5")
        self.ob_y.pack()

        tk.Label(frame, text="Size").pack()
        self.ob_size = tk.Entry(frame)
        self.ob_size.insert(0, "0.1")
        self.ob_size.pack()

        tk.Label(frame, text="Type").pack()
        self.ob_type = tk.StringVar()
        self.ob_type.set("circle")
        # Only circle and square allowed now
        tk.OptionMenu(frame, self.ob_type, "circle", "square").pack()

        tk.Button(frame, text="Add Obstacle", command=self.add_obstacle).pack()

        # Obstacle removal
        tk.Label(frame, text="Remove Obstacle Index").pack()
        self.remove_index = tk.Entry(frame)
        self.remove_index.insert(0, "0")
        self.remove_index.pack()
        tk.Button(frame, text="Remove Obstacle", command=self.remove_obstacle).pack()

        # Cost evaluation button
        tk.Button(frame, text="Evaluate Cost", command=self.evaluate_cost).pack()

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT)
        self.canvas.mpl_connect("button_press_event", self.start_draw)
        self.canvas.mpl_connect("motion_notify_event", self.draw_path)
        self.canvas.mpl_connect("button_release_event", self.end_draw)
        self.drawing = False
        self.update_plot(preserve_optimized=True)

    def update_start(self, event=None):
        self.start = [self.start_x.get(), self.start_y.get()]
        self.update_plot(preserve_optimized=True)

    def update_goal(self, event=None):
        self.goal = [self.goal_x.get(), self.goal_y.get()]
        self.update_plot(preserve_optimized=True)

    def add_obstacle(self):
        if len(self.obstacles) >= self.MAX_OBSTACLES:
            messagebox.showerror(
                "Limit reached",
                f"Cannot add more than {self.MAX_OBSTACLES} obstacles."
            )
            return

        try:
            x = float(self.ob_x.get())
            y = float(self.ob_y.get())
            size = float(self.ob_size.get())
            otype = self.ob_type.get()
            self.obstacles.append((x, y, size, otype))
            self.change_config = True
            self.update_plot(preserve_optimized=True)
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numbers for x, y, and size.")

    def remove_obstacle(self):
        try:
            index = int(self.remove_index.get())
            if 0 <= index < len(self.obstacles):
                self.obstacles.pop(index)
                self.change_config = True
                self.update_plot(preserve_optimized=True)
            else:
                messagebox.showerror("Invalid index", "Index out of range.")
        except ValueError:
            messagebox.showerror("Input error", "Please enter a valid integer index.")

    def point_in_obstacle(self, x, y):
        for ox, oy, size, otype in self.obstacles:
            if otype == "circle":
                if (x - ox) ** 2 + (y - oy) ** 2 <= size**2:
                    return True
            elif otype == "square":
                if ox - size/2 <= x <= ox + size/2 and oy - size/2 <= y <= oy + size/2:
                    return True
        return False

    def start_draw(self, event):
        if event.inaxes != self.ax:
            return
        self.drawing = True
        self.trajectory = [(event.xdata, event.ydata)]
        self.update_plot(preserve_optimized=True)

    def draw_path(self, event):
        if self.drawing and event.inaxes == self.ax \
           and event.xdata is not None and event.ydata is not None:
            self.trajectory.append((event.xdata, event.ydata))
            self.update_plot(preserve_optimized=True)

    def end_draw(self, event):
        self.drawing = False
        if self.trajectory:
            for i in range(len(self.trajectory) - 1):
                x0, y0 = self.trajectory[i]
                x1, y1 = self.trajectory[i + 1]
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                color = "r" if self.point_in_obstacle(mid_x, mid_y) else "k"
                self.ax.plot([x0, x1], [y0, y1], color=color)
        self.ax.legend()
        self.canvas.draw()

    def save_yaml(self):
        self.change_config = False
        config = {
            "body": {
                "shape": "dot",
                "dynamic": "point_2nd",
                "goal_mode": "center",
                "start_state": [self.start[0], self.start[1], 0.0, 0.0],
                "goal_state": [self.goal[0], self.goal[1], 0.0, 0.0],
                "control_bounds": [-1.0, 1.0],
            },
            "obstacles": [],
            "solver": {"N": 40, "dt": 0.1, "mode": "casadi"},
        }

        for x, y, size, otype in self.obstacles:
            if otype == "circle":
                config["obstacles"].append({
                    "type": "circle",
                    "center": [x, y],
                    "radius": size,
                    "margin": 0.05
                })
            elif otype == "square":
                config["obstacles"].append({
                    "type": "square",
                    "center": [x, y],
                    "size": size,
                    "margin": 0.05
                })

        folder = Path("configs")
        folder.mkdir(exist_ok=True)
        self.config = folder / Path(
            f"config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        with open(self.config, "w") as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=None)

        print(f"Configuration saved to {self.config}")

    def run_optimizer(self):
        if self.change_config:
            self.save_yaml()
        try:
            X_opt = run_benchmark(self.config, verbose=False)
            if X_opt is not None and X_opt.shape[0] >= 2:
                self.optimized_trajectory = X_opt
                self.update_plot(preserve_optimized=True)
        except ImportError as e:
            messagebox.showerror("Import Error", f"Could not import run_benchmark: {e}")
        except Exception as e:
            messagebox.showerror("Execution Error", f"Error running benchmark: {e}")

    def update_plot(self, preserve_optimized=False):
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect("equal")
        self.ax.set_xticks(np.linspace(0, 1, 11))
        self.ax.set_yticks(np.linspace(0, 1, 11))
        self.ax.grid(True)

        self.ax.plot(self.start[0], self.start[1], "go", label="Start")
        self.ax.plot(self.goal[0], self.goal[1], "ro", label="Goal")

        if preserve_optimized and self.optimized_trajectory is not None:
            x_vals = self.optimized_trajectory[0, :]
            y_vals = self.optimized_trajectory[1, :]
            self.ax.plot(x_vals, y_vals, "b-", label="Optimized")

        for idx, (x, y, size, otype) in enumerate(self.obstacles):
            if otype == "circle":
                circ = Circle((x, y), size, color="gray", alpha=0.5)
                self.ax.add_patch(circ)
            elif otype == "square":
                sq = Rectangle((x - size/2, y - size/2), size, size, color="blue", alpha=0.5)
                self.ax.add_patch(sq)
            self.ax.text(x, y, str(idx), fontsize=8, ha="center", va="center")

        if self.trajectory:
            x_vals, y_vals = zip(*self.trajectory)
            self.ax.plot(x_vals, y_vals, "k-", label="Trajectory")

        self.ax.legend()
        self.canvas.draw()

    def evaluate_cost(self):
        def compute_cost(path):
            if path is None or len(path) < 2:
                return 0
            cost = 0
            eps = 1e-8
            for i in range(len(path) - 1):
                dx = path[i + 1][0] - path[i][0]
                dy = path[i + 1][1] - path[i][1]
                cost += np.sqrt(dx**2 + dy**2 + eps)
            return cost

        hand_cost = compute_cost(self.trajectory)
        opt_cost = 0
        if self.optimized_trajectory is not None and self.optimized_trajectory.shape[1] >= 2:
            pts = list(zip(self.optimized_trajectory[0, :], self.optimized_trajectory[1, :]))
            opt_cost = compute_cost(pts)

        # Choose title based on comparison
        title = "You won!" if hand_cost <= opt_cost else "The optimization had lower cost"

        # Build message text
        msg = (
            f"Hand-drawn trajectory cost: {hand_cost:.4f}\n"
            f"Optimized trajectory cost: {opt_cost:.4f}"
        )

        # Show popup
        messagebox.showinfo(title, msg)

    def clear_trajectories(self):
        self.trajectory = []
        self.optimized_trajectory = None
        self.update_plot(preserve_optimized=False)


if __name__ == "__main__":
    root = tk.Tk()
    app = SimplePlannerUI(root)
    root.mainloop()
