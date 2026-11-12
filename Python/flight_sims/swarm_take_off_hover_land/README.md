# Simulation of a swarm take off / hover / land flight

Two runs:
- **no flags**: the simulation alone, **Main Controller only** (the hover): every value from `Structs.py` (`param.sim`: `nD`, `Tend`, `dTi`; `param.hover`: height and spacing of the homes; the Mellinger gains), nothing from the flight code or a real run. The drones start at rest at their hover points and hold them.
- **`--IRL_log`**: the `Crazy_Flie/Optitrack/swarm_take_off_hover_land` flight replayed, **all flight stages**, which it then compares with. Each drone starts from its logged position on the floor and gets the setpoints of the flight code (smoothstep ramps); each phase starts at its logged time with the setpoint period measured in the log:

| Phase | Setpoint |
|---|---|
| Take off | ramp to (home x, home y, 0.6 m) in 3 s |
| Hover | hold above home, 8 s |
| Return | ramp to the assigned take-off point (Hungarian assignment) |
| Land | ramp down to 5 cm in 4 s, motors off at the end of the log |

One script, `main.py`, for both modes. Same structure as `../swarm_geometric_Fibonacci_sim_target/` and `../8traj/`: the thesis simulation (V8) split in modules, the flight code in `flight_code_funcs.py`, the comparison in `compare_real.py`, and a `Results/simulation_results.pkl` that `Animation_display.py` redraws from.

| File | Content |
|---|---|
| `main.py` | `simulate_isolated` (no flags) and `simulate_flight` (`--IRL_log`), then the `.pkl` and the figures |
| `Structs.py` | `Param` (drone model `cf2x_P250`, Mellinger gains, `HoverConfig` of the simulation alone, `GraphicalDisplay` flags), `DroneState`, `SimulationData` |
| `drone_motion.py` | inner loop: Mellinger controller (`drone_mellinger_ctrl.py`) + full nonlinear dynamics (`drone_3dfull_dyn.py`), RK4 on SO(3); `passive_motion` (motors off), `ground_contact` |
| `helper_funcs.py` | rotations, RK4, `get_initial_real_offset` (initial state from the log) |
| `Animation_display.py` | figures (`position`, `top_view`, `trajectory_<n>_agents`) and the 3D animation; `python Animation_display.py` redraws them from `Results/simulation_results.pkl` |
| `flight_code_funcs.py` | the flight code: its constants (read from its `Structs.py` / `main.py`), log, setpoint schedule and ramps (with the Hungarian return assignment) |
| `compare_real.py` | real vs sim: metrics and figures. Also runs on its own: `python compare_real.py [--log LOG] [--sim SIM]` |
| `drone_models.toml` | drone parameters (`cf2x_P250`) |

Python environment: the repository's root `.venv` (`source .venv/bin/activate` from the repository root; see the root README).

```bash
# from the repository root, with the root .venv active
python Python/flight_sims/swarm_take_off_hover_land/main.py                      # the simulation alone (hover, Structs.py)
python Python/flight_sims/swarm_take_off_hover_land/main.py --IRL_log            # newest real run in Results/
python Python/flight_sims/swarm_take_off_hover_land/main.py --IRL_log Results/swarm_take_off_hover_land_trajectory_<date>/trajectory_<date>.csv --no-show
```

| Mode | Initial values | Figures |
|---|---|---|
| no flags | `Structs.py` only: `param.sim.nD` drones (default 3) at rest at their hover points, a row along x `param.hover.spacing` apart (0.5 m) at `param.hover.height` (0.6 m), for `param.sim.Tend` (10 s) | simulation only |
| `--IRL_log [LOG]` | those of the real flight `LOG` (default: the newest `swarm_take_off_hover_land` run): logged positions and yaw, logged phase times and setpoint period | simulation **and** comparison real vs sim |

`--no-show` saves the figures without opening them.

Outputs:
- **Both**: `Results/simulation_results.pkl` in this folder (`python Animation_display.py` redraws the figures / animations from it; it stores the `param.sim` values of its run).
- **No flags**: `Results/swarm_take_off_hover_land_<date>/`: `sim.csv` and `simulation/`.
- **`--IRL_log`**, in the run folder of the log:
  - `trajectory_<date>_sim.csv`: same columns as the log, plus the setpoints.
  - `simulation/`: the `Animation_display.py` figures of the simulated flight (the `GraphicalDisplay` flags in `Structs.py`; PDF + PNG) and a copy of `simulation_results.pkl`.
  - `comparison/`: `position`, `difference`, `takeoff_landing` and `summary` figures (PDF + PNG), and `metrics.json`. The metrics are position RMS / max per phase, lift-off and touchdown times, hover bias and jitter, and setpoint tracking error.

The `_sim.csv` is what `video_overlay/flight_overlay.py` draws over the phone video (see "How to run" in the repository README).
