# Project Capture

Multi-UAV cooperative target capture and tracking simulation using distributed estimation, Voronoi-based formation control, Delaunay communication networks, and full 3D quadrotor dynamics.

Developed by:
- Filipe Cavalheiro
- Bruno Guerreiro

# Overview

This project simulates a swarm of autonomous aerial vehicles (UAVs) that cooperatively track and surround a moving target. Each drone only has access to local information and communicates with neighboring agents through a dynamically generated communication network based on Delaunay triangulation.

The simulation combines:

- Distributed target state estimation (POS and VEL or just POS (param.sim))
- Voronoi-based formation generation OR Fibonacci-based formation (param.sim)
- Dynamic communication topology
- Full nonlinear quadrotor dynamics
- Agent failure (drone dropout)
- Collision avoidance via Horizon
- Visualization and post-processing tools

The objective is to maintain a cooperative formation around the target while remaining robust to the loss of communication agents.

# Simulation Pipeline

At every simulation step:

1. Update the target trajectory.
2. Simulate target motion.
3. Compute the Delaunay communication graph.
4. Remove failed drones (if scheduled).
5. Estimate the target state using neighboring information.
6. Compute desired formation positions.
7. Compute horizon to desired position (including vel,acc restrictions and collision avoidance)
8. Compute control inputs.
9. Propagate each UAV using the nonlinear dynamic model.
10. Store simulation data.
11. Display results after completion.

# Communication Network

The communication topology is generated online using **Delaunay triangulation** of the drone positions.

Advantages include:

- Local communication only
- Sparse connectivity
- Adaptive topology
- Neighbor relationships based on geometry

The resulting adjacency matrix and Laplacian are used by the distributed estimator.

# Formation Control

Desired UAV positions are generated using a Voronoi-based or Fibonacci-based strategy.

Each drone computes its desired state using:

- Current formation geometry
- Neighbor information
- Target estimate

The objective is to surround the moving target while maintaining a balanced spatial distribution.

# Distributed Target Estimation

Each UAV independently estimates:

- Target position
- Target velocity

Information is exchanged only with neighboring drones through the communication graph, allowing the swarm to operate without centralized coordination.

# Drone Dynamics

Each UAV is modeled using a full nonlinear 3D quadrotor dynamic model including:

- Position
- Velocity
- Attitude
- Angular velocity

The controller computes the required inputs, which are integrated through the dynamic model at every simulation step.

# Drone Failure Simulation

The simulator supports agent failures by removing drones during execution.

When a drone is dropped:

- It is removed from the communication network.
- It no longer contributes to target estimation.
- It follows passive dynamics until reaching the ground.
- Remaining agents automatically reconfigure the communication graph and continue the mission.

# Output Data

The simulation stores:

- UAV position history
- Desired positions
- Estimated target states
- Ground truth target trajectory
- Active drone information
- Number of active agents

These logs are used for visualization and performance evaluation.

# Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

# Running the Simulation

Run the main script:

```bash
python main.py
```

# Main Components

| Module | Description |
||-|
| `Structs.py` | Simulation parameters and data structures |
| `target_motion.py` | Target dynamic model |
| `delaunay_triangulation_network.py` | Communication graph generation |
| `estimate_target_variables.py` | Distributed target estimator |
| `compute_desired_state.py` | Formation generation |
| `actuate_control.py` | UAV controller |
| `drone_3dfull_dyn.py` | Nonlinear quadrotor dynamics |
| `Animation_display.py` | Visualization and animation |
| `helper_funcs.py` | Mathematical helper functions |

# Future Improvements

Potential extensions include:

- Obstacle avoidance
- Collision avoidance
- Sensor noise models
- Communication delays
- Packet loss simulation
- Multiple moving targets
- Formation switching
- Hardware-in-the-loop support
- ROS2 integration
- Real UAV experiments