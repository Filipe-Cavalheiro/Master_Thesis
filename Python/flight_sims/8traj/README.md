# Simulation of a figure-8 flight

Simulation of the one-drone figure-8 flight `Crazy_Flie/Optitrack/8traj.py`. Same layout as `../swarm_take_off_hover_land/`: one script, `main.py`, for both modes.

| Mode | Initial values | Figures |
|---|---|---|
| no flags | simulated: drone at rest at the origin, nominal setpoint period and phase times | simulation only |
| `--IRL_log [LOG]` | those of the real flight `LOG` (default: the newest `8traj` run): logged position and yaw, logged phase times and setpoint period | simulation **and** comparison real vs sim |

## The simulation

Replays the setpoint stream of `8traj.py` on the simulated Crazyflie. The Mellinger gains are the ones `8traj.py` writes to the firmware (kp = 1.1448, kv = 0.34344). The flight constants (height, speed, figure-8 size, times) are read from `8traj.py` itself, so edits to the flight script are followed.

| Phase | Setpoint |
|---|---|
| Take off | (0, 0, H (k+1)/n) |
| Main Controller | (A sin φ, B sin 2φ, H), with the rate eased in and out over `TRAJ_EASE_TIME` |
| Return to origin | smoothstep ramp to (0, 0, H). Flights before 10 Nov used a step instead |
| Land | (0, 0, H (1 − (k+1)/n)) |

```bash
# from the repository root, with the root .venv active
python Python/flight_sims/8traj/main.py                     # simulation only
python Python/flight_sims/8traj/main.py --IRL_log           # newest real 8traj run in Results/
python Python/flight_sims/8traj/main.py --IRL_log Results/8traj_trajectory_<date>/trajectory_<date>.csv --no-show
```

| Option | Effect |
|---|---|
| `--IRL_log [LOG]` | start from the initial values of a real flight and compare with it (default `LOG`: newest `8traj` run) |
| `--no-show` | save the figures without opening them |
| `--return auto\|ramp\|step` | return phase of the flight: `ramp` (from 10 Nov) or the old `step`; `auto` (default): detected from the log with `--IRL_log`, `ramp` without |
| `--ease S` | `TRAJ_EASE_TIME` of the flight (default: detected from the log with `--IRL_log`, from `8traj.py` without; use 0 for flights before 10 Nov) |

Outputs:
- **No flags**: `Results/8traj_<date>/`: `sim.csv` and `simulation/`.
- **`--IRL_log`**, in the run folder of the log:
  - `trajectory_<date>_sim.csv`: same columns as the log, plus the setpoints `sp_x`, `sp_y`, `sp_z`. This is what `video_overlay/flight_overlay.py` draws over the phone video.
  - `simulation/`: `position` and `top_view` figures of the simulated flight (PDF + PNG).
  - `comparison/`: `position`, `top_view` and `difference` figures, and `metrics.json` (RMS per phase, max, tracking error).

Self-contained: `main.py` only uses `cf_plant.py` (Crazyflie plant of the thesis simulation, V8). That file is the same as in `../swarm_take_off_hover_land/`; keep both copies equal. It also uses `drone_models.toml` (drone `cf2x_P250`).

## `legacy_main.py`: the older simulation

The first sim-vs-real comparison of the figure 8 (October flights), with its own modules: `Structs.py`, `helper_funcs.py`, `agent_motion.py`, `drone_3dfull_dyn.py`, `drone_mellinger_ctrl.py`, `time_alignment.py`, `Animation_display.py`, `plot_real_data.py`. It starts from the real drone state and automatically finds the time offset between the reference trajectory and the flight (`time_alignment.py`). Results go to `simulation_results.pkl` and the figures to `8traj/Results/` (`real_sim_trajectory.pdf`, `real_sim_difference.pdf`, ...).

```bash
cd Python/flight_sims && python 8traj/legacy_main.py      # paths are relative to Python/flight_sims (8traj/drone_models.toml, 8traj/real_flights/)
```

`real_flights/` holds the October flight logs it uses. `main.py` does not use any of these files.

Python environment: the repository's root `.venv` (`source .venv/bin/activate` from the repository root; see the root README).
