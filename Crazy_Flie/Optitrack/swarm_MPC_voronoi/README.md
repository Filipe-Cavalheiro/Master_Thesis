# Swarm Voronoi formation + MPC around a simulated or real target (OptiTrack)

The target is chosen per flight (`python main.py --target sim | real`, default `sim`):

- `sim`: virtual target, the formation follows the target path itself.
- `real`: the Crazyflie `REAL_TARGET_DRONE_ID` (Structs.py) flies the target path with full-state setpoints; the formation follows its OptiTrack position and velocity and the chasers' MPC avoids it (envelope `TARGET_RADIUS_XY`).

(This folder replaces `swarm_MPC_voronoi_sim_target`, the virtual-target-only copy.)

`swarm_voronoi_sim_target` (virtual target on a circle, formation mesh + CVT) where the drones reach their Voronoi goals through the MPC of the Python simulation (`Python/obstacle_avoidance_funcs.py`, `AdmmMPC`) instead of a rate-limited setpoint.

**MPC** (`mpc.py`, `SwarmMPC`): every drone plans a 1 s Bernstein polynomial horizon (degree 5, 5 samples) to its formation goal, with smoothness + goal + goal velocity (target velocity) costs and an exponential approach to the goal. The other drones' horizons, the arena geofence and the velocity/acceleration limits (`MPC_V_MAX`, `MPC_A_MAX`) are projection constraints solved with ADMM. The KKT matrix is factorized once. A planner thread replans all the drones every `MPC_PERIOD` (0.1 s, ~5–12 ms for 2–5 drones). Each drone's `controller` sends its plan sampled at the current time every `SETPOINT_PERIOD` as a full-state setpoint (position, velocity, acceleration; needs Mellinger, `SETPOINT_MODE`).

Changes from the simulation's MPC for real drones:

- **Replanning start:** the plan starts from the previous plan at the current time while the drone is within `MPC_REPLAN_RESET_DIST` of it, else from the measured state (OptiTrack finite-difference velocity, clipped to 0.95 `v_max`).
- **ADMM:** scaled form `b = Proj(A x + lambda / rho)`, and the multipliers reset every replanning (`MPC_WARM_MULTIPLIERS`).
- **First horizon sample:** left out of the obstacle constraints, since the equality constraint already fixes it.
- **Feedforward:** clipped to 1.2 `v_max` / 1.5 `a_max`.
- **Downwash:** a **cylinder** (radius `MPC_RADIUS_XY` 0.3 m, half-height `MPC_RADIUS_Z` 1.2 m) around the other drones instead of an ellipsoid, and `MPC_RHO_OBS` = 5000 instead of 100. The 0.4 m formation stacks the goals most of the time (3–5 drones). The ellipsoid pushed stacked drones vertically, which the formation cannot give, and with `rho_obs` = 100 the goal cost kept them stacked. Offline (below), the drones are never closer than 0.2 m horizontally with less than 1.2 m vertically, at the cost of ~0.15 m median goal error.
- **Slots:** the approach slots are where the MPC holds the drones around the still formation (`SwarmMPC.settle`, before arming), not the stacked Voronoi goals. The flight is refused if two slots are stacked (`SLOT_MIN_DH`).

Flight sequence (`main.py`):

1. Discovery, OptiTrack tracking, external position at 100 Hz, Kalman + Mellinger, Kalman reset, home capture (|yaw| ≤ 10°), battery check. The formation is converged at the initial target position, the slots are assigned from the takeoff points and settled by the MPC, then checked for stacking.
2. Arm, take off to `TAKEOFF_HEIGHT`, fly to the slots on straight synchronized paths (`APPROACH_SPEED`).
3. Target, Voronoi and MPC threads start; the controllers send the plans for `FORMATION_TIME`.
4. Return above the takeoff points, land, disarm. This also runs after an abort: lost tracking, outside the arena, two drones closer than `MIN_SEPARATION`, an MPC plan older than `PLAN_TIMEOUT`, an error, or `Ctrl+C`.

Offline check (formation + MPC + drones as double integrators tracking the full-state setpoint, 2–5 drones, 30 s): stable, median goal error 0.06–0.18 m, max 0.57 m when the formation morphs (target crossing x = 0), max speed 0.84–0.95 m/s, min distance 0.39 m, inside x, y ∈ [−2.07, 2.06] m and z ∈ [0.89, 1.73] m. Not flown yet.

Files: `Structs.py` (all settings, including the `MPC_*` ones), `mpc.py`, `cf_funcs.py` (per-drone commands, controller, safety), `formation.py` (virtual target and Voronoi threads, goal velocity), `voronoi_mesh_funcs.py`, `natnet_funcs.py` (pose and velocity), `trajectory_logger.py` (CSV with setpoint `sp_*` and target `target_*`). Logs go to the repository `Results/` folder, one folder per run (`Results/swarm_MPC_voronoi_<sim|real>_target_trajectory_<date>/`), the TOC cache to `cache/`.

Requires `cflib`, `natnet`, `numpy` and `scipy`. Run `swarm_take_off_hover_land/` first with new drones, batteries or controller settings.

```bash
python main.py                  # virtual target
python main.py --target real    # target Crazyflie
```
