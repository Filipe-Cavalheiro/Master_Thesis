# Swarm Voronoi formation + MPC around a simulated or real target (OptiTrack)

The target is chosen per flight (`python main.py --target sim | real`, default `sim`):

- `sim`: virtual target, the formation follows the target path itself.
- `real`: the Crazyflie `REAL_TARGET_DRONE_ID` (Structs.py) flies the target path with full-state setpoints; the formation follows its OptiTrack position and velocity and the chasers' MPC avoids it (envelope `TARGET_RADIUS_XY`).

(This folder replaces `swarm_MPC_voronoi_sim_target`, the virtual-target-only copy.)

`swarm_voronoi_sim_target` (virtual target on a circle, formation mesh + CVT) where the drones reach their Voronoi goals through the MPC of the thesis instead of a rate-limited setpoint.

**MPC** (`mpc.py`, `SwarmMPC`): the distributed Bernstein-polynomial MPC of the thesis (Chapter 6), **the same code as the Python simulation V17** (`Python/flight_sims/V17_Lisbon_aeroport/Python/obstacle_avoidance_funcs.py`: `init_obstacle_avoidance`, `set_goal_cost`, `AdmmMPC`). `mpc.py` only feeds it the OptiTrack states and the formation goals and runs it in a thread. Settings (`MPC_*`, `Structs.py`), as in the thesis unless noted:

- every control step (`MPC_PERIOD` = `SETPOINT_PERIOD`, 0.05 s; ~6 ms for 3 chasers, ~12 ms for 4 with a target drone) each chaser plans a 1 s horizon (degree 5, 5 samples) to its formation goal;
- cost: tracking of the exponential approach to the goal moving with the goal velocity (w_g 1000, w_v 300, τ_r0 0.5 s) + smoothness (w_s 0.1);
- constraints as projections, scaled ADMM (ρ_obs 100, ρ_ineq 20): ‖v‖ ≤ `MPC_V_MAX`, ‖a‖ ≤ `MPC_A_MAX`, a **sphere** of radius `MPC_SAFETY_RADIUS` (0.3 m, arena scale; 1 m in Lisbon) around the other chasers' horizons and the constant-velocity prediction of a target drone, all horizon samples. No downwash term. No airspace / geofence block (free space); the formation goals are still kept in the arena (`clip_goals`);
- multipliers warm-started, up to 15 iterations with the 1e-2 residual stop, chasers updated in turn;
- setpoint: the plan at the second horizon sample t₁ (`MPC_SETPOINT = "t1"`), position, velocity, acceleration, as a full-state setpoint.

One departure from the thesis, for the real drones: **`MPC_START = "plan"`**. The thesis starts every plan from the measured state. On the Crazyflie that diverges in the dry run (the drones fall): the onboard controller holds the drone ~3 cm below its setpoint (`z_offset` of the V17 Crazyflie plant, fitted on the 26 Oct flights), so a plan from the measured position puts the next setpoint just above where the drone already is, and it sinks step after step. As V17's `mpc_tight_tracking`, the plan starts from the previous plan one control period later while the drone is within `MPC_REPLAN_RESET_DIST` (0.2 m) of it, else from the measured state (OptiTrack velocity clipped to 0.95 v_max). `MPC_START = "measured"` and `MPC_SETPOINT = "next_step"` (the plan one period ahead, V17 `mpc_setpoint`) are there to test the alternatives.

Note: with ρ_obs = 100 the separation pushes with at most ~ρ_obs·r (~30) against the goal weight (1000), so two goals inside each other's safety sphere stay there (offline: two goals 5 cm apart end 5.5 cm apart). The arena formation (radius 0.8 m) keeps its goals 0.98–1.17 m apart for 2–4 chasers, so this does not arise in the dry runs below.

**Slots:** the approach slots are where the MPC holds the drones around the still formation (`SwarmMPC.settle`, before arming).

**Log:** besides the trajectory (`trajectory_<date>.csv`, with the setpoint `sp_*` and target `target_*`), `<log>_meta.json` saves the flight setup: target mode and trajectory, formation and MPC settings, per drone its role, home, goal, slot and the firmware parameters read back after the setup (estimator, controller, all `ctrlMel.*` gains). The flight is not armed unless every drone reads back the Kalman estimator and the Mellinger controller.

Flight sequence (`main.py`):

1. Discovery, OptiTrack tracking, external position at 100 Hz, Kalman + Mellinger, Kalman reset, home capture (|yaw| ≤ 10°), battery check. The formation is converged at the initial target position, the slots are assigned from the takeoff points and settled by the MPC, then the firmware parameters are read back and checked, and the flight setup is saved to `<log>_meta.json`.
2. Arm, take off to `TAKEOFF_HEIGHT`, fly to the slots on straight synchronized paths (`APPROACH_SPEED`).
3. Target, Voronoi and MPC threads start; the controllers send the plans for `FORMATION_TIME`.
4. Return above the takeoff points, land, disarm. This also runs after an abort: lost tracking, outside the arena, two drones closer than `MIN_SEPARATION`, an MPC plan older than `PLAN_TIMEOUT`, an error, or `Ctrl+C`.

Offline check (`python main.py --arena` in V17, below: this flight code, the V17 Crazyflie plant, 30 s, trajectory 2): 3 / 4 chasers and 3 chasers + target drone: stable, distance to the target 0.9–1.0 m on average (sphere 0.8 m; 1.7 m while the target is behind the wall at x = 0), min distance between chasers 0.87–1.09 m, to the target drone 0.61 m, setpoint speed ≤ 0.88 m/s. The simulated drones overshoot to 1.5–2 m/s for under 0.7 s, at the start and when the formation closes as the target comes back (~22 s). Not flown yet.

Simulation of this flight (V17, `Python/flight_sims/V17_Lisbon_aeroport/Python`):

```bash
python main.py --arena [--target real] [--traj 5] [--drones 4]   # dry run with the settings of Structs.py
python main.py --IRL_log [LOG]                                   # replay of the newest run (or LOG) and comparison
```

Files: `Structs.py` (all settings, including the `MPC_*` ones), `mpc.py`, `cf_funcs.py` (per-drone commands, controller, safety), `formation.py` (virtual target and Voronoi threads, goal velocity), `voronoi_mesh_funcs.py`, `natnet_funcs.py` (pose and velocity), `trajectory_logger.py` (CSV with setpoint `sp_*` and target `target_*`). Logs go to the repository `Results/` folder, one folder per run (`Results/swarm_MPC_voronoi_<sim|real>_target_trajectory_<date>/`), the TOC cache to `cache/`.

The pose of the filming phone (Motive rigid body `CAMERA_RIGID_BODY_ID`, `Structs.py`; `None` to disable) is logged from the same NatNet frames to `<log>_camera.csv/.json` next to the trajectory log (`camera_logger.py`), from before the take off, for the video overlay (`Python/demos_and_helpers/video_overlay/flight_overlay.py`).

Requires `cflib`, `natnet`, `numpy` and `scipy`. Run `swarm_take_off_hover_land/` first with new drones, batteries or controller settings.

```bash
python main.py                  # virtual target
python main.py --target real    # target Crazyflie
```
