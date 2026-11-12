# Simulation of the swarm geometric Fibonacci flight

One script, `main.py`, for two runs, as in `../swarm_take_off_hover_land/` and `../8traj/`:
- **no flags**: the thesis simulation alone (described below), every parameter from `Structs.py`;
- **`--IRL_log`**: the `Crazy_Flie/Optitrack/swarm_geometric_Fibonacci_sim_target` flight replayed from the initial values of a real flight, which it then compares with.

Both save `simulation_results.pkl` and draw the `Animation_display.py` figures. The replay of the real flight:

| Phase | Setpoint (as in the flight code) |
|---|---|
| Take off | ramp to (home x, home y, `default_height` 1.3 m) in 3 s |
| Main Controller | every `dTi` (0.1 s), for `Tend` (30 s): geodesic guidance of `q_ref` to the Fibonacci point of the drone, APF on the references, adaptive radius `r -> sphere_radius` (0.8 m); setpoint = target + r q_ref. The virtual target (circle of 1.5 m at 0.5 rad/s) starts with the controller |
| Return | ramp to (home x, home y, 1.3 m) in 3 s |
| Land | ramp to (home x, home y, 0) in 4 s, motors off at the end of the log |

The Main Controller setpoints do not depend on the measured positions, so the sim can rebuild them exactly from the guidance initial state. What it uses depends on the log:

| Log | Constants | Main Controller setpoints | Controller period | Plant position gains |
|---|---|---|---|---|
| with `_meta.json` / `_setpoints.csv` (from 12 Nov) | those of the flight (`_meta.json`) | the logged ones (`_setpoints.csv`): exact times and values | measured in the flight | `ctrlMel.kp_*` / `kd_*` read back from the firmware |
| without (older) | read from the flight code now (`Structs.py`, `main.py`) | rebuilt from the old initial state: `q_ref = (1, 0, 0)` for every drone, radius 0 | `dTi` | those of `Structs.py` |

With the setpoint log, the real setpoints of the comparison are the logged ones too. The ramps (take off, return, land) are always replayed from the simulated positions, with the setpoint period measured in the log.

| File | Content |
|---|---|
| `main.py` | `simulate_isolated` (no flags) and `simulate_flight` (`--IRL_log`), then the `.pkl` and the figures |
| `Structs.py`, `drone_motion.py`, `helper_funcs.py`, `fiboancci_funcs.py`, `Animation_display.py` | the thesis simulation: `Param` / `DroneState` / `SimulationData`, the inner loop (Mellinger + full nonlinear dynamics, RK4 on SO(3); `passive_motion` and `ground_contact` for the motors off / on the floor), the guidance (`linear_motion_ref`, `apf_safe_guard`, `update_radius`, integrated with the flight's `dTi` in the replay), the Hungarian assignment, the Fibonacci points, the figures |
| `flight_code_funcs.py` | the flight code: its constants (read from its `Structs.py` / `main.py`), target trajectory, log and logged flight setup, setpoint schedule, ramps, initial guidance state |
| `compare_real.py` | real vs sim: metrics and figures |
| `drone_models.toml` | drone parameters (`cf2x_P250`) |

```bash
# from the repository root, with the root .venv active
python Python/flight_sims/swarm_geometric_Fibonacci_sim_target/main.py                    # the simulation alone (Structs.py)
python Python/flight_sims/swarm_geometric_Fibonacci_sim_target/main.py --IRL_log          # newest real run in Results/
python Python/flight_sims/swarm_geometric_Fibonacci_sim_target/main.py --IRL_log Results/swarm_geometric_Fibonacci_sim_target_trajectory_<date>/trajectory_<date>.csv --no-show
```

| Mode | Initial values | Figures |
|---|---|---|
| no flags | the thesis simulation, isolated: `param.sim.nD` drones (odd: the Fibonacci points need it; an even `nD` stops with a message) at random positions around the target, `sphere_radius`, gains, target trajectory, `Tend` and `dTi` of `Structs.py`; nothing from the flight code or a log | simulation only |
| `--IRL_log [LOG]` | those of the real flight `LOG` (default: the newest `swarm_geometric_Fibonacci_sim_target` run, or an older `swarm_Fibonacci_sim_target` one): logged positions and yaw, logged phase times and setpoint period | simulation **and** comparison real vs sim |

| Option | Effect |
|---|---|
| `--ctrl-period S` | with `--IRL_log`: period of the controller loop (default: measured in the flight if logged, else `dTi` of the flight code) |
| `--no-show` | save the figures without opening them |

Outputs:
- **Both**: `Results/simulation_results.pkl` in this folder (`python Animation_display.py` redraws the figures / animations from it).
- **No flags**: `Results/swarm_geometric_Fibonacci_sim_target_<date>/`: `sim.csv` (positions, setpoints and target of every drone) and `simulation/` (figures and a copy of the `.pkl`).
- **`--IRL_log`**, in the run folder of the log:
  - `trajectory_<date>_sim.csv`: same columns as the log, plus the setpoints `sp_*` and the virtual target `target_*`.
  - `simulation/`: the `Animation_display.py` figures (the `GraphicalDisplay` flags in `Structs.py`) and a copy of `simulation_results.pkl`.
  - `comparison/`: `position`, `top_view`, `encirclement`, `difference` and `summary` figures (PDF + PNG), and `metrics.json`: position RMS / max per phase, setpoint tracking RMS (real and sim), distance to the target over the last 10 s of the Main Controller, minimum distance between drones.

`real_flights/trajectory_20261015_101358.csv` is a 3-drone flight from 15 Oct (with `--IRL_log` on it, the outputs go into `real_flights/`).

## The thesis simulation (`main.py` without flags)

The rest of this README describes the thesis simulation (V8), what `main.py` runs without `--IRL_log`.

### Hierarchical Nonlinear Control for Swarm Encirclement Around a Moving Target UAV

A Python simulation framework for autonomous multi-UAV swarm encirclement of a moving target using a hierarchical nonlinear control architecture.

The project combines:

- Hierarchical inner–outer loop swarm coordination
- Geometric (reduced-attitude) control on the two-sphere Γ²
- Fibonacci lattice point separation for uniform spatial coverage
- Hungarian (Munkres) assignment of agents to encirclement points
- Artificial Potential Fields (APF) for collision avoidance
- Adaptive radial control toward the desired encirclement radius
- Full nonlinear quadrotor dynamics with a Mellinger tracking controller

## Overview

Multiple quadrotors cooperatively surround and follow a moving aerial target. The encirclement problem is formulated on a sphere of radius `r*` centered at the target: each UAV `i` is driven toward a desired position

```
pᵢ* = p_T + r* qᵢ*
```

where `p_T` is the target position and `qᵢ* ∈ S²` is a desired direction generated by Fibonacci lattice sampling. The objective is decomposed into two guidance sub-problems solved by the outer loop:

- **angular:** drive the direction `qᵢ → qᵢ*` along geodesics of the unit sphere,
- **radial:** drive the distance `rᵢ → r*` with an adaptive law gated by the angular error.

The inner loop (Mellinger controller + full quadrotor dynamics) tracks the position references produced by the outer loop.

## Repository Structure

```
.
├── main.py                          # Main simulation script
├── Structs.py                       # All parameters and data classes
├── helper_funcs.py                  # Guidance laws: geodesic reference,
│                                    #   APF safeguard, radial update, SO(3) utils
├── fiboancci_funcs.py               # Fibonacci points, geodesic distances,
│                                    #   safety distance, sphere projection
├── target_motion.py                 # Target dynamics propagation
├── actuate_control.py               # Chaser inner-loop step (control + dynamics)
├── drone_mellinger_ctrl.py          # Mellinger position/attitude controller
├── drone_3dfull_dyn.py              # Quadrotor nonlinear dynamics
├── Animation_display.py             # Figures and 3D animation (matplotlib/plotly)
├── Results/                         # Generated figures (PDF)
└── ...                              # Auxiliary/experimental modules
```

## Control Architecture

Per simulation step (`dTi = 0.01 s`), for each UAV:

```
Moving Target  ──────────────►  target_motion()
      │
      ▼
Fibonacci reference points  (once, at t = 0)
      │
      ▼
Hungarian assignment        (once, at t = 0, geodesic cost)
      │
      ▼
Geodesic guidance state     q_ref ∈ S²    (eq. 11, integrated by the outer loop)
      │
      ▼
APF safeguard               deflects q_ref around neighboring references (eq. 14)
      │
      ▼
Radial guidance state       r_ref → r*    (eq. 15, gated by e^(−‖e_q‖²))
      │
      ▼
Position reference          p_d = p_T + r_ref · q_ref
      │
      ▼
Mellinger controller  ──►  thrust, torques  ──►  quadrotor dynamics (Euler step)
```

Two implementation points matter for correctness:

1. **The guidance references are states.** Both `q_ref` (on S²) and `r_ref` are
   initialized from each drone's initial pose and integrated by the outer loop —
   they are *not* re-anchored to measured positions each step. Re-anchoring feeds
   inner-loop tracking lag back into the reference and prevents convergence when
   the target moves.
2. **The APF acts on the references.** Repulsion is computed between the guidance
   references of all drones and integrated into them, so commanded paths deflect
   around each other and vanish once separations exceed the safety threshold
   (half of the average geodesic distance between neighboring Fibonacci points).

## Simulation Parameters

All parameters live in `Structs.py`:

| Parameter | Location | Description |
|-----------|----------|-------------|
| `param.sim.nD` | `SimulationConfig` | Number of drones (odd, ≥ 3) |
| `param.sim.Tend` / `dTi` | `SimulationConfig` | Duration / timestep |
| `param.sim.sphere_radius` | `SimulationConfig` | Encirclement radius `r*` |
| `param.traj` | `TrajectoryConfig` | Target trajectory (default: half-circle, then stationary) |
| `param.k_geometric` | `Param` | Geodesic guidance gain `k_geo` |
| `param.k_rad` | `Param` | Radial guidance gain `k_rad` |
| `param.k_apf` | `Param` | APF repulsion gain |
| `param.controller.Mellinger` | `ControllerConfig` | Inner-loop gains `kp`, `kv`, `kR`, `kom` |
| `param.display` | `GraphicalDisplay` | Figure/animation switches |

Note on gains: the paper's Table I lists `Kp = diag(150)`, but combined with
`Kv = diag(15)` and no thrust saturation the inner loop is underdamped enough to
diverge for some initial conditions. The code ships with `Kp = diag(100)`, which
converges reliably.

## Running the Simulation

```bash
source <repo>/.venv/bin/activate     # the repository's root environment
python main.py
```

To redo the figures / animations of the last run without re-simulating (as in `V17_Lisbon_aeroport`): `main.py` saves the simulation data to `Results/simulation_results.pkl` when `param.display.store_data` is `True` (default), and

```bash
python Animation_display.py     # loads Results/simulation_results.pkl, draws what the GraphicalDisplay flags select
```

Set the flags in `Structs.py` first (e.g. `trajectory_animate = True`). The pkl stores the `param.sim` values of its run (`nD`, `Tend`, `dTi`, `Nsim`, `sphere_radius`), so it is redrawn as it was simulated. `save_to_mp4` is forced off.

Initial chaser positions are random (no fixed seed), so each run differs. The
simulation prints its iteration counter, then opens the configured figures. With
figures on an interactive backend, press `q` in a figure window to exit; on a
headless/Agg backend the wait is skipped automatically.

## Visualization

Configured through the flags in `GraphicalDisplay` (`Structs.py`):

- `composed` — 3D snapshots of the swarm and target over time (saved to `Results/composed_<nD>_agents.pdf`)
- `radial_hist` — radial distance of every agent vs. time
- `consensus` — inter-agent angular separations vs. time
- `animate` — interactive 3D animation (plotly)

## Requirements

- Python ≥ 3.10
- `numpy`, `scipy`, `scikit-learn`, `matplotlib` (in the repository's root `requirements.txt` / `.venv`)

## Citation

If you use this repository in your research, please cite the associated publication (not yet available).

```bibtex
@article{NOTYETAVAILABLE,
  title={Hierarchical Nonlinear Control for Swarm Encirclement Around a Moving Target UAV},
  author={Filipe Cavalheiro, Bruno Guerreiro},
  year={2026}
}
```

---

## License

This work uses the AGPL license; more information can be viewed at `LICENSE`.

## Future Work

Potential extensions include:

- Decentralized estimation and assignment
- Obstacle avoidance (Control Barrier Functions on Γ² or in ℝ³)
- Time-varying communication networks
- Hardware-in-the-loop simulation
- ROS 2 integration
- PX4 implementation
- Real-world flight experiments
