# MuJoCo Rover Simulation with LQR Control

This project simulates a 4-wheeled rover navigating over Mars-like terrain using [MuJoCo](https://mujoco.org/). The rover uses a Linear Quadratic Regulator (LQR) for balance and speed control. A GUI built with PySide6 allows users to control and visualize the rover in real-time.

---

## 🧠 Features

- ✅ 6-DOF rover model with suspension and articulated wheels
- 🛞 Independent motor control for each wheel
- 🕹️ Real-time user control via keyboard or sliders
- 🧮 LQR-based velocity and orientation control
- 🪨 Simulated Mars-like terrain with rocks and elevation
- 🌄 Camera control and lighting via MuJoCo + Qt OpenGL

---

## 📁 Files

| File              | Description |
|-------------------|-------------|
| `robot-02.xml`    | Defines the MuJoCo model of the rover including suspension, wheels, body, sensors, etc. |
| `scene.xml`       | Adds terrain, rocks, and visual elements, and includes the robot model. |
| `simulate_robot.py` | Main launcher: sets up GUI, OpenGL rendering, and interactive control. |
| `robot_lqr.py`    | Contains the LQR-based control logic for speed and steering adjustments. |

---

## 🚀 Installation

### 1. Install MuJoCo
- Download and install MuJoCo from [https://mujoco.org/](https://mujoco.org/)
- Set environment variable `MUJOCO_PY_MUJOCO_PATH` or use pip-installed version

  ## 🎮 Controls

- `↑` / `↓`: Increase/decrease speed  
- `←` / `→`: Steer left/right  
- `Space`: Stop movement  
- GUI Sliders: Fine-tune speed and steering


### 2. Install Python Requirements

```bash
pip install mujoco PySide6 numpy scipy
