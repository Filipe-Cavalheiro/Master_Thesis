# Crazyflie flights with OptiTrack

Flight code for the Crazyflie 2.x with OptiTrack (Motive, NatNet unicast). The drones run the onboard Kalman filter. The PC streams their OptiTrack position to the filter (`extpos`, 100 Hz) and sends position setpoints (20 Hz).

## Setup

Python environment: the repository's root `.venv` (`source .venv/bin/activate` from the repository root; see the root README).

Before flying:
- Set `MOTIVE_IP` / `LOCAL_IP`, the radio URIs and the Motive rigid-body IDs. In the swarm folders these are in `Structs.py`; in the standalone scripts, at the top of each file. Drone `…E7E7E7E70N` is rigid body `N`.
- Place every drone pointing along +X of the tracking frame: the Kalman filter assumes yaw ≈ 0 at reset.

Run the swarm folders from inside their folder (`python main.py`) and the standalone scripts from this folder. Both use a `./cache` folder for the Crazyflie TOC cache.

## Swarm folders (self-contained, one README each)

| Folder | Flight | Logs the phone (overlay) |
|---|---|---|
| [`swarm_take_off_hover_land/`](swarm_take_off_hover_land/README.md) | take off, hover, return, land. Run it first with new drones, batteries or gains | yes |
| [`swarm_geometric_Fibonacci_sim_target/`](swarm_geometric_Fibonacci_sim_target/README.md) | Fibonacci-sphere encirclement of a virtual target (geometric controller) | no |
| [`swarm_voronoi_sim_target/`](swarm_voronoi_sim_target/README.md) | Voronoi formation around a virtual target | no |
| [`swarm_MPC_voronoi/`](swarm_MPC_voronoi/README.md) | Voronoi formation with the distributed MPC (`--target sim` or `--target real`) | no |

## Standalone scripts

| Script | Flight | Flight log |
|---|---|---|
| `takeoff_hover_land.py` | one drone: take off to 0.5 m, hover 5 s, land | — |
| `8traj.py` | one drone: figure 8 (1.5 × 1.0 m, 30 s) at 1 m, then return and land. Logs the phone pose (rigid body 0) for the overlay | `Results/8traj_trajectory_<date>/` |
| `box.py` | one drone: back and forth inside a ±1 m box for 30 s (velocity setpoints) | — |
| `swarm_8traj.py` | swarm: figure 8 around each home, each drone on its own height layer | — |
| `swarm_fibonacci.py` | swarm: Fibonacci sphere (radius 0.5 m) whose centre moves on a circle for 30 s | — |

For the single-drone scripts, the URI can also be set with the `CFLIB_URI` environment variable.

## Safety (all flight scripts)

| | Effect |
|---|---|
| tracking lost; also, depending on the script, a drone outside the arena or above `MAX_HEIGHT` (the single-drone scripts check tracking only) | abort: return home, land, disarm |
| **1st Ctrl+C** (in flight) | the same abort |
| **2nd Ctrl+C** | motors off at once: **the drones fall** |

Before arming and after landing, Ctrl+C just quits. The handler is `ctrl_c.py`, one copy here and one in each swarm folder; keep them equal.

## Analysis tools

| Script | Use |
|---|---|
| `plot_data.py` | plots / animation of a flight log: `python plot_data.py [log.csv] [--animate] [--source est] [--save-figs DIR]` (default: newest log in `Results/`) |
| `flight_sim.py` | simulates a swarm flight from its log and compares, or a dry run before flying: `python flight_sim.py swarm_MPC_voronoi`, `python flight_sim.py swarm_voronoi_sim_target --dry-run --drones 3` (the code is in V17, `Python/flight_sims/V17_Lisbon_aeroport/Python/flight_replay.py`; for `swarm_MPC_voronoi` also `python main.py --IRL_log` / `--arena` there) |
| `camera_logger.py` | logs the phone rigid body from the flight's NatNet frames (used by `8traj.py`) |
| `natnet_test.py` | checks the connection to Motive |

The simulation of `swarm_take_off_hover_land` and `8traj.py` flights, and the overlay on the phone video, are in `Python/flight_sims/` and `Python/demos_and_helpers/video_overlay/`. See "How to run" in the repository README.
