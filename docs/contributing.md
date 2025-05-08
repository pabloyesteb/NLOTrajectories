# 🛠️ How to Extend the Project

This guide explains how to add new components to the trajectory optimization project (bodies, dynamics, obstacles, etc.) and how to keep your code clean and consistent.

---


## ➕ Adding a New Body Shape

> Shapes like dot, rectangle, triangle...

1. Open `core/geometry.py`
2. Create a new class that inherits from `IRobotGeometry` (or `PolygonGeometry` for polygons)
3. Implement the required methods:
    - `transform()`
    - `sdf_constraints()`
    - `goal_cost()`
    - `draw()`
4. Register your class in `Shape` enum and modify `create_geometry` in `BodyConfig`

---

## ➕ Adding a New Dynamics Model

> For example: holonomic car, 3D drone, etc.

1. Open `core/dynamics.py`
2. Create a class that inherits from `IRobotDynamics`
3. Implement:
   - `state_dim()`
   - `control_dim()`
   - `dynamics(x, u)`
4. Register your class in `Dynamics` enum and modify `create_dynamics` in `BodyConfig`

---

## ➕ Adding a New Obstacle Type

1. Open `core/sdf/casadi.py`
2. Create a class that inherits from `IObstacle`
3. Implement:
   - `sdf()`
   - `draw()`
4. Write config for your class in `config.py`, update `ObstacleConfigs`

---

## 🧪 Adding a New Benchmark

1. Create a YAML file under `benchmarks/` (e.g., `benchmark_4_new_task.yaml`)
2. Define:
   - `body` section (shape, dynamic, etc.)
   - `obstacles` section
   - `solver` settings (`N`, `dt`, etc.)
3. Run it using:

```bash
poetry run run-benchmark --config <BENCHMARK_PATH>
```

---

## ✅ Final Checklist Before You Commit

- [ ] Code runs successfully with `run-benchmark`
- [ ] Added tests (optional but recommended)
- [ ] Ran formatting and linting:

```bash
poetry run make pretty lint
```

- [ ] Your new class is registered in the appropriate configs

---

Thank you for contributing 🚀
