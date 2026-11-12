# Swarm Voronoi formation around a virtual target (OptiTrack)

The `lighthouse_swarm_voronoi_sim_target` mission with OptiTrack. A virtual target moves on a horizontal circle. The formation mesh (circle → cone → sphere around the target, morphed by the target's distance to the plane x = 0) is updated every `VORONOI_PERIOD` with a CVT every 5 updates, and every drone follows its Voronoi site for `FORMATION_TIME`.

Flight sequence (`main.py`):

1. Discovery, OptiTrack tracking, external position at 100 Hz, Kalman + Mellinger, Kalman reset, home capture (|yaw| ≤ 10°), battery check. The formation at the initial target position is converged (CVT) before arming.
2. Arm, take off to `TAKEOFF_HEIGHT`.
3. Each drone gets the formation slot closest to it (minimum sum of squared distances) and flies there on a straight synchronized path (`APPROACH_SPEED`).
4. Target and Voronoi threads start; each drone's `controller` follows its goal for `FORMATION_TIME`, the setpoint moving towards it at `MAX_SETPOINT_SPEED` at most.
5. Return above the takeoff points (reassigned, minimum sum of squared distances), land, disarm. This also runs after an abort.

Changes from the lighthouse version for the 5 × 5 × 2.5 m arena: target circle radius 1.6 m (was 2.2), 0.3 rad/s (was 0.5); trajectories 3 and 4 (z = 5 m) removed; goals clipped 0.3 m inside the arena; setpoints rate limited instead of jumping to the goal; abort if two drones are closer than `MIN_SEPARATION` (0.25 m), on lost tracking, or outside the arena.

Self-contained (the logs go to `Results/` two levels up from `Crazy_Flie/Optitrack`). Settings are in `Structs.py` (URIs, Motive IPs, arena, heights, target, formation), `cf_funcs.py` (per-drone commands, controller, safety), `formation.py` (virtual target and Voronoi threads), `voronoi_mesh_funcs.py` (mesh and CVT, same as the lighthouse version), `natnet_funcs.py` and `trajectory_logger.py` (CSV with setpoint `sp_*` and target `target_*`). Logs go to the repository `Results/` folder, one folder per run (`Results/<this folder>_trajectory_<date>/`) and the Crazyflie TOC cache to `cache/`.

Requires `cflib`, `natnet`, `numpy` and `scipy`. Run `swarm_take_off_hover_land/` first with new drones, batteries or controller settings.

```bash
python main.py
```
