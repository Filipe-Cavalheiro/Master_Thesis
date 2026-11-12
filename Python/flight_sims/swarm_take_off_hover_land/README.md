# Simulation of a swarm take off / hover / land flight

Replays a real `Crazy_Flie/Optitrack/swarm_take_off_hover_land` flight on the simulated Crazyflie and compares the two. Each drone starts from its logged position, at rest on the floor, and gets the setpoints the flight code sent (smoothstep ramps, at the setpoint period measured in the log). Each phase starts at its logged time:

| Phase | Setpoint |
|---|---|
| Take off | ramp to (home x, home y, 0.6 m) in 3 s |
| Hover | hold above home, 8 s |
| Return | ramp to the assigned take-off point (Hungarian assignment) |
| Land | ramp down to 5 cm in 4 s, motors off at the end of the log |

Self-contained:

| File | Content |
|---|---|
| `main.py` | the simulation (setpoint schedule rebuilt from the log, simulation loop, floor contact) |
| `compare.py` | metrics and figures real vs sim |
| `cf_plant.py` | Crazyflie plant of the thesis simulation (V8): drone model, Mellinger controller, full nonlinear dynamics, RK4 on SO(3), initial state from the log. Same file as in `../8traj/`; keep both copies equal |
| `drone_models.toml` | drone parameters (`cf2x_P250`) |

Python environment: the repository's root `.venv` (`source .venv/bin/activate` from the repository root; see the root README).

```bash
# from the repository root, with the root .venv active
python Python/flight_sims/swarm_take_off_hover_land/main.py                    # newest run in Results/
python Python/flight_sims/swarm_take_off_hover_land/main.py --log Results/swarm_take_off_hover_land_trajectory_<date>/trajectory_<date>.csv --no-show
python Python/flight_sims/swarm_take_off_hover_land/main.py --sim --drones 3   # no log: nominal flight only
python Python/flight_sims/swarm_take_off_hover_land/compare.py LOG.csv [SIM.csv] --no-show   # comparison only
```

| Option | Effect |
|---|---|
| `--log` | flight log (default: newest `swarm_take_off_hover_land` run) |
| `--no-show` | save the figures without opening them |
| `--sim` | no log, no comparison: nominal flight of `--drones` drones in a row, `--spacing` m apart (default 3, 0.5 m) |

Outputs:
- **With a log**, next to it:
  - `trajectory_<date>_sim.csv`: same columns as the log, plus the setpoints.
  - `comparison/`: `position`, `difference`, `takeoff_landing` and `summary` figures (PDF + PNG), and `metrics.json`. The metrics are position RMS / max per phase, lift-off and touchdown times, hover bias and jitter, and setpoint tracking error.
- **`--sim`**: `Results/swarm_take_off_hover_land_<date>/sim.csv`.

The `_sim.csv` is what `video_overlay/flight_overlay.py` draws over the phone video (see "How to run" in the repository README).
