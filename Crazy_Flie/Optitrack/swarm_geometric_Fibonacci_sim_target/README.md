# Swarm geometric Fibonacci encirclement (virtual target)

The real-flight version of the thesis Fibonacci-sphere controller (`Python/flight_sims/swarm_geometric_Fibonacci_sim_target`). A virtual target moves on a horizontal circle. Each chaser is assigned a Fibonacci-lattice point on a sphere around the target and reaches it with the reduced-attitude (geometric) controller on the two-sphere, with APF collision avoidance and an adaptive radius. The setpoints are computed on the PC and sent with `send_position_setpoint`.

Flight: take off to `default_height` above each home → controller for `Tend` (30 s) → return above home → land → disarm.

Onboard controller: Mellinger (`stabilizer.controller = 2`), with the position gains of `8traj.py` (`kp = 1.1448`, `kv = 0.34344`, `MELLINGER_PARAMS` in `Structs.py`), the ones of the simulated Crazyflie plant.

Guidance initial state (`cf_funcs.init_guidance`, after the homes are captured): `q_ref` is the direction from the target's start position to the drone's takeoff point (home x, home y, `default_height`), and the radius state its distance. So the first controller setpoint is where the takeoff left the drone, and the radius then converges to `sphere_radius`. Before 12 Nov `q_ref` was computed before the OptiTrack stream started (all drones got the fallback (1, 0, 0)) and the radius started at 0, so every drone was first sent to the target itself.

Self-contained: everything it needs is in this folder.

| File | Content |
|---|---|
| `main.py` | the flight |
| `Structs.py` | URIs, Motive IPs, timings, safety limits; `Param` (sphere radius 0.8 m, height 1.3 m, gains `k_rad`, `k_geometric`, `k_apf`, target trajectories) |
| `cf_funcs.py` | per-drone commands (take off, `controller`, return, land, disarm), tracking checks, external-position threads |
| `fiboancci_funcs.py` | Fibonacci points, minimum safe distance, Delaunay network |
| `helper_funcs.py` | guidance reference, APF safeguard, radius update |
| `natnet_funcs.py`, `trajectory_logger.py` | OptiTrack frames, flight log |
| `ctrl_c.py` | two-stage Ctrl+C (see below) |

Python environment: the repository's root `.venv` (`source .venv/bin/activate` from the repository root; see the root README).

```bash
python main.py
```

The target trajectory is chosen in `main.py` (`Param(number_of_drones=..., traj=2)`):
- 2: the circle of radius `target_traj_radius` (1.5 m) at `default_height`. This is the only one sized for the arena.
- 1, 3, 4: a still point at 5 m, and 5 m figure 8s. These are simulation-scale, so don't fly them here.

Safety: the flight aborts (return home and land) if OptiTrack tracking is lost or a drone goes above `MAX_HEIGHT` (2.5 m).
- **1st Ctrl+C**: return home and land.
- **2nd Ctrl+C**: motors off at once; the drones fall.

Unlike the other swarm folders, this one has no battery check and does not log the phone pose, so it can't be used with the video overlay.

Logs go to `Results/swarm_geometric_Fibonacci_sim_target_trajectory_<date>/`:
- `trajectory_<date>.csv`: OptiTrack and Kalman state of every drone at 50 Hz.
- `trajectory_<date>_setpoints.csv`: every position setpoint sent (time on the same base, phase, drone), with the virtual target, the radius and `q_ref` of the controller.
- `trajectory_<date>_meta.json`: what the simulation needs: the firmware parameters read back after the setup (controller, all `ctrlMel.*` gains), timings and the nominal controller / target periods, the guidance gains and `min_safe_dist`, the target trajectory, and per drone its URI, guidance index (Fibonacci point), home, home yaw, `q_ref0` and `radius0`. At the end it also gets the measured setpoint period per phase and drone.

 To simulate a flight and compare it with the real one: `python Python/flight_sims/swarm_geometric_Fibonacci_sim_target/main.py --IRL_log` (see its README). Runs from before the folder was renamed are in `Results/swarm_Fibonacci_sim_target_trajectory_*`.
