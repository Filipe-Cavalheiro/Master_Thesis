# Simulation of a figure-8 flight

One script, `main.py`, for two runs, as in `../swarm_take_off_hover_land/` and `../swarm_geometric_Fibonacci_sim_target/`. Both save `Results/simulation_results.pkl` and draw the `Animation_display.py` figures.

| Mode | Initial values | Figures |
|---|---|---|
| no flags | the simulation alone, **Main Controller only** (the figure 8): every value from `Structs.py` (`param.sim`, the trajectory of `TrajectoryConfig`, the Mellinger gains of `ControllerConfig`), nothing from `8traj.py` or a real run. The drone starts at rest at the start of the trajectory and tracks its position and velocity (`time_alignment.run_simulation`) | simulation only (`trajectory_plot`) |
| `--IRL_log [LOG]` | those of the real flight `LOG` (default: the newest `8traj` run): logged position and yaw, logged phase times and setpoint period. **All flight stages** | simulation **and** comparison real vs sim |

## The replay of a real flight (`--IRL_log`)

Replays the setpoint stream of `8traj.py` on the simulated Crazyflie. The Mellinger gains are the ones `8traj.py` writes to the firmware (kp = 1.1448, kv = 0.34344). The flight constants (height, speed, figure-8 size, times) are read from `8traj.py` itself, so edits to the flight script are followed.

| Phase | Setpoint |
|---|---|
| Take off | (0, 0, H (k+1)/n) |
| Main Controller | (A sin φ, B sin 2φ, H), with the rate eased in and out over `TRAJ_EASE_TIME` |
| Return to origin | smoothstep ramp to (0, 0, H). Flights before 10 Nov used a step instead |
| Land | (0, 0, H (1 − (k+1)/n)) |

```bash
# from the repository root, with the root .venv active
python Python/flight_sims/8traj/main.py                     # the simulation alone (Main Controller, Structs.py)
python Python/flight_sims/8traj/main.py --IRL_log           # newest real 8traj run in Results/
python Python/flight_sims/8traj/main.py --IRL_log Results/8traj_trajectory_<date>/trajectory_<date>.csv --no-show
```

| Option | Effect |
|---|---|
| `--IRL_log [LOG]` | start from the initial values of a real flight and compare with it (default `LOG`: newest `8traj` run) |
| `--no-show` | save the figures without opening them |
| `--return auto\|ramp\|step` | with `--IRL_log`: return phase of the flight, `ramp` (from 10 Nov) or the old `step`; `auto` (default): detected from the log |
| `--ease S` | with `--IRL_log`: `TRAJ_EASE_TIME` of the flight (default: detected from the log; use 0 for flights before 10 Nov) |

Outputs:
- **Both**: `Results/simulation_results.pkl` in this folder (`python Animation_display.py` redraws the figures from it). It holds the Main Controller section and the `param.sim` values of the run (flight log, steps).
- **No flags**: `Results/8traj_<date>/`: `sim.csv` and `simulation/` (`trajectory_plot`; the real-vs-sim figures are off: nothing real to compare with).
- **`--IRL_log`**, in the run folder of the log:
  - `trajectory_<date>_sim.csv`: same columns as the log, plus the setpoints `sp_x`, `sp_y`, `sp_z`. This is what `video_overlay/flight_overlay.py` draws over the phone video.
  - `simulation/`: the `Animation_display.py` figures (the `GraphicalDisplay` flags in `Structs.py`: `real_sim_trajectory`, `real_sim_difference`, the real vs sim animation of the Main Controller) and a copy of `simulation_results.pkl`.
  - `comparison/`: `position`, `top_view` and `difference` figures, and `metrics.json` (RMS per phase, max, tracking error).

| File | Content |
|---|---|
| `main.py` | the simulation of the flight (laid out as `legacy_main.py`) |
| `Structs.py`, `agent_motion.py`, `helper_funcs.py`, `drone_3dfull_dyn.py`, `drone_mellinger_ctrl.py`, `Animation_display.py`, `plot_real_data.py` | the thesis simulation, shared with `legacy_main.py`: `Param` / `DroneState` / `SimulationData`, the inner loop (Mellinger + full nonlinear dynamics, RK4 on SO(3); `passive_motion` and `ground_contact` for the motors off / on the floor), `get_initial_real_offset`, the figures |
| `flight_code_funcs.py` | `8traj.py`: its constants (read from it), log, and the setpoint stream it sends |
| `compare_real.py` | real vs sim over the whole flight: metrics and figures |
| `drone_models.toml` | drone parameters (`cf2x_P250`) |

## `real_flights/`

`real_flights/` holds the October flight logs (`Param()` takes the newest one as `param.sim.real_trajectory_csv`; `main.py` replaces it with its `--IRL_log`).

Python environment: the repository's root `.venv` (`source .venv/bin/activate` from the repository root; see the root README).
