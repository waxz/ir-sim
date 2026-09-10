<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/hanruihua/ir-sim/main/docs/source/_static/branding/ir-sim-logo-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/hanruihua/ir-sim/main/docs/source/_static/branding/ir-sim-logo-light.png">
  <img src="https://raw.githubusercontent.com/hanruihua/ir-sim/main/docs/source/_static/branding/ir-sim-logo-light.png" alt="IR-SIM — Intelligent Robot Simulator" width="560">
</picture>

*A lightweight, YAML-driven robot simulator for navigation, control, and learning*

<a href="https://arxiv.org/pdf/2606.08729"><img src="https://img.shields.io/badge/arXiv-2606.08729-b31b1b?style=for-the-badge" alt="arXiv Paper"></a>
<a href="#citation" title="View the IR-SIM BibTeX citation"><img src="https://img.shields.io/badge/Cite-IR--SIM-2ea44f?style=for-the-badge" alt="Cite IR-SIM"></a>
<a href="https://pypi.org/project/ir-sim/"><img src="https://img.shields.io/pypi/v/ir-sim?color=orange&style=for-the-badge" alt="PyPI Version"></a>
<a href="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?style=for-the-badge" alt="Python Version"></a>
<a href="https://github.com/hanruihua/ir-sim/actions/workflows/python-version-test.yml"><img src="https://img.shields.io/github/actions/workflow/status/hanruihua/ir-sim/python-version-test.yml?branch=main&style=for-the-badge&label=CI" alt="CI"></a>
<a href="https://codecov.io/gh/hanruihua/ir-sim"><img src="https://img.shields.io/codecov/c/github/hanruihua/ir-sim?style=for-the-badge&color=yellow" alt="Coverage"></a>
<a href="https://ir-sim.readthedocs.io/en/stable/"><img src="https://img.shields.io/badge/docs-online-blue?style=for-the-badge" alt="Docs"></a>
<a href="https://github.com/hanruihua/ir-sim?tab=MIT-1-ov-file"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"></a>
<a href="https://pepy.tech/project/ir-sim"><img src="https://img.shields.io/pepy/dt/ir-sim?style=for-the-badge" alt="Downloads"></a>
<a href="https://github.com/knmcguire/best-of-robot-simulators?tab=readme-ov-file#robotic-simulators-in-2d"><img src="https://img.shields.io/badge/2D%20Robotics%20Simulators-%F0%9F%A5%87%20Rank%20%231-success?style=for-the-badge" alt="Ranked #1 among 2D Robotics Simulators in best-of-robot-simulators"></a>

</div>

## Contents

- [Overview](#overview)
- [Demonstrations](#demonstrations)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Integrations](#advanced-integrations)
  - [Foxglove Studio](#foxglove-studio)
  - [shmbridge — Shared-Memory Robot Control](#shmbridge--shared-memory-robot-control)
- [Support](#support)
- [Projects Using IR-SIM](#projects-using-ir-sim)
- [Citation](#citation)

## Overview

**IR-SIM** is an open-source, Python-based, lightweight robot simulator designed for navigation, control, and learning. It provides a simple, user-friendly framework with built-in collision detection for modeling robots, sensors, and environments. Ideal for academic and educational use, IR-SIM enables rapid prototyping of robotics and learning algorithms in custom scenarios with minimal coding and hardware requirements.

## Key Features

- Simulate robot platforms with diverse kinematics, sensors, and behaviors  ([support](#support)). 
- Quickly configure and customize scenarios using straightforward YAML files. No complex coding required.
- Visualize simulation results with a lightweight Matplotlib-based renderer for rapid debugging.
- Support collision detection and customizable behavior policies for each object.
- Suitable for multi-agent and robot-learning research ([Projects](#projects-using-ir-sim)).

## Demonstrations

<table>
<tr>
<td align="center" width="33%">
<img src="https://github.com/user-attachments/assets/5930b088-d400-4943-8ded-853c22eae75b" width="240"/><br/>
<b>Multi-Robot RVO Collision Avoidance</b><br/>
<a href="https://github.com/hanruihua/ir-sim/blob/main/usage/11collision_avoidance/collision_avoidance.py">Source</a>
</td>
<td align="center" width="33%">
<img src="https://github.com/user-attachments/assets/3257abc1-8bed-40d8-9b51-e5d90b06ee06" width="240"/><br/>
<b>Ackermann Robot with 2D LiDAR</b><br/>
<a href="https://github.com/hanruihua/ir-sim/blob/main/usage/10grid_map/grid_map.py">Source</a>
</td>
<td align="center" width="33%">
<img src="https://github.com/user-attachments/assets/0fac81e7-60c0-46b2-91f0-efe4762bb758" width="240"/><br/>
<b>HM3D / MatterPort3D Grid Map</b><br/>
<a href="https://github.com/hanruihua/ir-sim/blob/main/usage/10grid_map/grid_map_hm3d.py">Source</a>
</td>
</tr>
<tr>
<td align="center" width="33%">
<img src="https://github.com/user-attachments/assets/7aa809c2-3a44-4377-a22d-728b9dbdf8bc" width="240"/><br/>
<b>Field-of-View Detection</b><br/>
<a href="https://github.com/hanruihua/ir-sim/blob/main/usage/15fov_world/fov_world.py">Source</a>
</td>
<td align="center" width="33%">
<img src="https://github.com/user-attachments/assets/1cc8a4a6-2f41-4bc9-bc59-a7faff443223" width="240"/><br/>
<b>Dynamic Random Obstacles</b><br/>
<a href="https://github.com/hanruihua/ir-sim/blob/main/usage/08random_obstacle/dynamic_random.py">Source</a>
</td>
<td align="center" width="33%">
<img src="https://github.com/user-attachments/assets/162cf52e-070d-4588-b9b2-bf21c487fbc8" width="240"/><br/>
<b>200-Agent ORCA via <a href="https://github.com/hanruihua/pyrvo">pyrvo</a></b><br/>
<a href="https://github.com/hanruihua/ir-sim/blob/main/usage/19orca_world/orca_behavior_world.py">Source</a>
</td>
</tr>
</table>

## Installation

> **Requires Python >= 3.10**

### pip

```bash
pip install ir-sim

# Optional: keyboard control and all extras
pip install ir-sim[all]
```

### From source

```bash
git clone https://github.com/hanruihua/ir-sim.git
cd ir-sim
pip install -e .
```

### uv

```bash
git clone https://github.com/hanruihua/ir-sim.git
cd ir-sim
uv sync
```

## Quick Start

A minimal example: a differential-drive robot navigates toward a goal using the built-in `dash` behavior.

```python
import irsim

env = irsim.make(
    "robot_world.yaml"
)  # initialize the environment with the configuration file

for i in range(300):  # run the simulation for 300 steps
    env.step()  # update the environment
    env.render()  # render the environment

    if env.done():
        break  # check if the simulation is done

env.end()  # close the environment
```

YAML Configuration: `robot_world.yaml`

```yaml
world:
  height: 10  # the height of the world
  width: 10   # the width of the world
  step_time: 0.1  # 10Hz calculate each step
  sample_time: 0.1  # 10 Hz for render and data extraction
  offset: [0, 0] # the offset of the world on x and y

robot:
  kinematics: {name: 'diff'}  # omni, omni_angular, diff, acker
  shape: {name: 'circle', radius: 0.2}  # radius
  state: [1, 1, 0]  # x, y, theta
  goal: [9, 9, 0]  # x, y, theta
  behavior: {name: 'dash'} # move toward to the goal directly
  color: 'g' # green
```

For more examples, see the [usage directory](https://github.com/hanruihua/ir-sim/tree/main/usage) and the [documentation](https://ir-sim.readthedocs.io/en).

## Advanced Integrations

### Foxglove Studio

Visualize and tele-operate a live IR-SIM environment from [Foxglove Studio](https://foxglove.dev) over a WebSocket connection.

**Install the extra dependency:**

```bash
pip install ir-sim[foxglove]
# or: pip install ir-sim foxglove-websocket
```

**Run the demo:**

```bash
python usage/foxglove_demo.py
```

Open Foxglove Studio → **Add connection** → **Foxglove WebSocket** → `ws://localhost:8765`.

Channels streamed by the demo:

| Channel | Schema | Description |
|---|---|---|
| `/irsim/pose` | `foxglove.PoseInFrame` | Robot pose |
| `/irsim/lidar2d` | `foxglove.LaserScan` | 2D LiDAR scan |
| `/irsim/imu` | `foxglove.Imu` | Gyro + accelerometer |
| `/irsim/scene` | `foxglove.SceneUpdate` | 3D bodies for robot and obstacles |
| `/irsim/map` | `foxglove.Grid` | 2D occupancy map |

Send commands back to the sim from Studio's **Publish** panel:

```json
// /irsim/cmd_vel  — velocity override
{"linear": {"x": 0.5}, "angular": {"z": 0.3}}

// /irsim/control  — pause / resume / reset
{"command": "pause"}
```

---

### shmbridge — Shared-Memory Robot Control

**shmbridge** provides a POSIX shared-memory seqlock transport so Python
(IR-SIM) and C++ controllers can exchange robot state and velocity commands at
low latency without a network stack.

```
IR-SIM (Python publisher)  ──write_state──►  C++ or Python controller
                           ◄──write_cmd──    (shmbridge subscriber)
```

#### Building wheels

**IR-SIM wheel** (pure Python):

```bash
pip install build
python -m build          # produces dist/ir_sim-*.whl
```

**shmbridge wheel** (Python + C++ pybind11 extension):

```bash
cd shmbridge
pip install scikit-build-core pybind11
python -m build          # produces dist/shmbridge-*.whl
# or install directly:
pip install -e .
```

The C++ extension (`_core.so`) is compiled automatically by scikit-build-core.
When it is not available the package falls back to a pure-Python ctypes
implementation transparently.

#### Building the C++ controller demo

The [`usage/27shmbridge_control/`](usage/27shmbridge_control/) directory
contains a standalone C++ controller (`controller_cpp.cpp`) that attaches to
the IR-SIM shared-memory segment with no Python dependency at runtime.

```bash
# From the repo root
cmake -S usage/27shmbridge_control \
      -B /tmp/ctrl_build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ctrl_build
```

This produces `/tmp/ctrl_build/ctrl_cpp`.  The `CMakeLists.txt` finds the
shmbridge headers from an installed package first, then falls back to
`shmbridge/include/` in the repo.

#### Running the integration demo

The demo requires IR-SIM and shmbridge to be installed (or on `PYTHONPATH`).
Open **two terminals**:

```bash
# Terminal 1 — simulator (IR-SIM publisher, 20 Hz)
python usage/27shmbridge_control/sim.py

# Terminal 2 — Python controller (proportional-heading, 40 Hz)
python usage/27shmbridge_control/controller_py.py

# — or — C++ controller (same algorithm, no Python runtime needed)
/tmp/ctrl_build/ctrl_cpp
```

Add `--render` to `sim.py` to open a Matplotlib window.

The simulator publishes robot pose, velocity, goal distance, and flags after
each `env.step()`.  The controller reads the state, computes
`[linear, angular]` via a proportional-heading law, and writes the command
back.  The sim picks it up with `read_best_cmd()` and feeds it to the next
`env.step()`.

**Run the integration tests** (no subprocesses — everything in-process):

```bash
pytest usage/27shmbridge_control/test_shmbridge_irsim.py -v
```

Five tests cover the state round-trip, command round-trip, multiple writes,
and a 200-step closed-loop run that verifies the robot closes ≥ 30 % of
its goal distance.

---

## Support

| **Category**     | **Features**                                                                                                                                                                            |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Kinematics**   | Differential Drive mobile Robot · Omnidirectional mobile Robot · Omnidirectional with Angular control · Ackermann Steering mobile Robot                                                 |
| **Sensors**      | 2D LiDAR · 2D FMCW LiDAR · FOV Detector                                                                                                                                                 |
| **Geometries**   | Circle · Rectangle · Polygon · LineString · Binary Grid Map · Fog of Map                                                                                                                |
| **Behaviors**    | dash (move directly toward goal) · RVO (Reciprocal Velocity Obstacle) · ORCA (Optimal Reciprocal Collision Avoidance) · SFM (Social Force Model)                                        |

## Documentation

- **English:** [https://ir-sim.readthedocs.io/en](https://ir-sim.readthedocs.io/en)
- **Chinese (中文):** [https://ir-sim.readthedocs.io/zh-cn](https://ir-sim.readthedocs.io/zh-cn)

## Projects Using IR-SIM

### Academic Publications

- **[RAL & ICRA 2023]** [rl-rvo-nav](https://github.com/hanruihua/rl_rvo_nav) -- Reinforcement learning-based RVO behavior for multi-robot navigation.
- **[RAL & IROS 2023]** [RDA_planner](https://github.com/hanruihua/RDA_planner) -- Accelerated collision-free motion planner for cluttered environments.
- **[T-RO 2025]** [NeuPAN](https://github.com/hanruihua/NeuPAN) -- Direct point robot navigation with end-to-end model-based learning.
- **[ROBIO 2025]** [MfNeuPAN](https://doi.org/10.1109/ROBIO66223.2025.11377233) -- Proactive end-to-end navigation in dynamic environments using direct multi-frame point constraints.
- **[WCL 2025]** [Robotic Sensor Network: Achieving Mutual Communication Control Assistance With Fast Cross-Layer Optimization](https://doi.org/10.1109/LWC.2024.3502757) -- Cross-layer communication and control optimization evaluated on IR-SIM.
- **[IROS 2026]** [SDLW](https://github.com/williamleong/sdlw) -- Decentralized scalable exploration using sensor-driven Lévy walks for minimal-sensing robot teams.
- **[Sensors 2026]** [PPO-GAT-Follow](https://doi.org/10.3390/s26154711) -- Graph-attention reinforcement learning for robot person following in dense crowds.
- **[TWC 2026]** [Energy-Efficient Federated Edge Learning for Small-Scale Datasets in Large IoT Networks](https://doi.org/10.1109/TWC.2026.3683911) -- Energy-efficient federated edge learning evaluated through IR-SIM autonomous-navigation experiments and CARLA validation.

### Community Projects

- [DRL-robot-navigation-IR-SIM](https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM) -- Deep reinforcement learning for robot navigation.
- [AutoNavRL](https://github.com/harshmahesheka/AutoNavRL) -- Autonomous navigation using reinforcement learning.
- [IRSIM-3DGS-Bridge](https://github.com/Wayneyujie/IRSIM-3DGS-Bridge) -- A closed-loop bridge from 3D Gaussian Splatting scenes to IR-SIM planning/following and back to Habitat-GS trajectory playback.
- [EdgeVox](https://github.com/nrl-ai/edgevox) -- Offline voice-agent framework with an IR-SIM mobile-navigation backend.

### Courses Using IR-SIM

- **The University of Hong Kong (HKU):** [COMP3356 Robotics](https://www.cs.hku.hk/index.php/programmes/course-offered?infile=2026/comp3356.html)
- **Southern University of Science and Technology (SUSTech):** [CS401 Intelligent Robotics](https://github.com/Intelligent-Robot-Course)

*If your publication, project, or course uses IR-SIM, we welcome proposals for inclusion via an issue or pull request.*

## Citation

If you find IR-SIM useful, please consider starring ⭐ this project and citing our paper:

```bibtex
@article{han2026ir,
  title={IR-SIM: A Lightweight Skill-Native Simulator for Navigation, Learning, and Benchmarking},
  author={Han, Ruihua and Wang, Shuai and Li, Chengyang and Gao, Rui and Wang, Xinyi and Liu, Zhe and Li, Guoliang and Lu, Yupu and Hao, Qi and Pan, Jia and Zhao, Hengshuang},
  journal={arXiv preprint arXiv:2606.08729},
  year={2026},
  doi={10.48550/arXiv.2606.08729},
  url={https://arxiv.org/abs/2606.08729},
  eprint={2606.08729},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](https://github.com/hanruihua/ir-sim/blob/main/CONTRIBUTING.md) for guidelines.

## Acknowledgement

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)

## License

IR-SIM is released under the [MIT License](https://github.com/hanruihua/ir-sim?tab=MIT-1-ov-file).
