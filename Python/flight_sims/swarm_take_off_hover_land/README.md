# Simulation of a swarm take off / hover / land flight

Simulates the `Crazy_Flie/Optitrack/swarm_take_off_hover_land` flight on the simulated Crazyflie, either from simulated initial values or, with `--IRL_log`, from those of a real flight, which it then compares with. Each drone starts at rest on the floor and gets the setpoints of the flight code (smoothstep ramps). With `--IRL_log`, each drone starts from its logged position, and each phase starts at its logged time with the setpoint period measured in the log:

| Phase | Setpoint |
|---|---|
| Take off | ramp to (home x, home y, 0.6 m) in 3 s |
| Hover | hold above home, 8 s |
| Return | ramp to the assigned take-off point (Hungarian assignment) |
| Land | ramp down to 5 cm in 4 s, motors off at the end of the log |

Self-contained. Same layout as `../8traj/`: one script, `main.py`, for both modes.

| File | Content |
|---|---|
| `main.py` | the simulation, the simulation figures and the comparison with a real flight |
| `cf_plant.py` | Crazyflie plant of the thesis simulation (V8): drone model, Mellinger controller, full nonlinear dynamics, RK4 on SO(3), initial state from the log. Same file as in `../8traj/`; keep both copies equal |
| `drone_models.toml` | drone parameters (`cf2x_P250`) |

Python environment: the repository's root `.venv` (`source .venv/bin/activate` from the repository root; see the root README).

```bash
# from the repository root, with the root .venv active
python Python/flight_sims/swarm_take_off_hover_land/main.py                      # simulation only
python Python/flight_sims/swarm_take_off_hover_land/main.py --drones 5 --spacing 0.6
python Python/flight_sims/swarm_take_off_hover_land/main.py --IRL_log            # newest real run in Results/
python Python/flight_sims/swarm_take_off_hover_land/main.py --IRL_log Results/swarm_take_off_hover_land_trajectory_<date>/trajectory_<date>.csv --no-show
```

| Mode | Initial values | Figures |
|---|---|---|
| no flags | simulated: `--drones` drones at rest on the floor in a row, `--spacing` m apart (default 3, 0.5 m), nominal phase times | simulation only |
| `--IRL_log [LOG]` | those of the real flight `LOG` (default: the newest `swarm_take_off_hover_land` run): logged positions and yaw, logged phase times and setpoint period | simulation **and** comparison real vs sim |

`--no-show` saves the figures without opening them.

Outputs:
- **No flags**: `Results/swarm_take_off_hover_land_<date>/`: `sim.csv` and `simulation/`.
- **`--IRL_log`**, in the run folder of the log:
  - `trajectory_<date>_sim.csv`: same columns as the log, plus the setpoints.
  - `simulation/`: `position` and `top_view` figures of the simulated flight (PDF + PNG).
  - `comparison/`: `position`, `difference`, `takeoff_landing` and `summary` figures (PDF + PNG), and `metrics.json`. The metrics are position RMS / max per phase, lift-off and touchdown times, hover bias and jitter, and setpoint tracking error.

The `_sim.csv` is what `video_overlay/flight_overlay.py` draws over the phone video (see "How to run" in the repository README).
