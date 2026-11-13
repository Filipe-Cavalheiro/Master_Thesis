# V17: Lisbon-airport simulation (spherical envelope, scaled ADMM)

V17 is the Python simulation used for the Lisbon-airport results of the TAES paper
("Non-Destructive Deterrence of Drone Intrusions into Restricted Airspace with Morphing Swarm
Formations"), copied from the `air-space-deterrent` repository (`Python/`, 8 Nov 2026). It replaces V16 (removed on 12 Nov, in the git history), whose trajectories 1-4 it flies in the Lisbon and arena scenes (below). It differs from V16 in:

- **Morphing formation and restricted airspace**: wall at 30 m outside the red zone of the LPPT
  KMZ. The mesh blends disc -> cone -> sphere -> disc on the red border. A geodesic CVT (Dijkstra) assigns the goals.
- **Distributed Bernstein MPC solved with ADMM.** Velocity, acceleration, separation, airspace and obstacle
  constraints are all projections.
- **Spherical safety envelope only.** Each chaser keeps a ball of radius `obstacle_avoidance_dim`
  (1 m in Lisbon, 0.3 m in the Crazyflie arena) from the other chasers and from the target
  when it is avoided. The old downwash cylinder and ellipsoid were removed on 12 Nov (the arena numbers below are unchanged).
- **Fixed ADMM multiplier update.** The obstacle/airspace projection is the standard scaled form
  `b_o = Pi(A_o z + lambda_o / rho_o)` (`obstacle_avoidance_funcs.py`, `AdmmMPC.solve`). Before, it was `Pi(A_o z)`, which
  froze the multipliers and left the chasers up to 2.35 m from the red border instead of on its 1 m margin. The multipliers are
  warm-started between control steps in Lisbon and reset every step in the arena (`admm_warm_multipliers`).

## Layout

```
V17/
  Python/                    simulation code (run from here)
    Results/python/          results used in the paper (seed 1)
      simulation_results_traj_5_12drop4.pkl   trajectory 5: 12 chasers, 4 lost at 25 s
      simulation_results_traj_6.pkl           trajectory 6 (34.5 deg notch): 5 chasers
      arena5_3.pkl, arena5_4.pkl              arena trajectory 5, Crazyflie model, 3 / 4 chasers
  Assets/Aeroporto de Lisboa.kmz              ANAC airspace (read as ../Assets from Python/)
  figures/make_lisbon_figures.py              figures + Table II metrics (paper's make_figures.py, V17 paths)
  figures/*.pdf                               the generated figures (same as in the paper)
  review_scripts/                             re-run and number-extraction scripts of the paper
    run_sims_sphere.py        re-run a scenario: python run_sims_sphere.py traj5|traj6|arena3|arena4 SEED [OUT]
    lisbon_claims_sphere.py   numbers of Sec. VI-A (trajectories 5 and 6)
    lisbon_extras_sphere.py   extra numbers + repetitions summary
    arena_sim_sphere.py       arena simulation numbers
    sphere_pass_outputs.txt   output of all of the above for the paper's data (incl. seeds 2-5)
```

## Running

```bash
source <repo>/.venv/bin/activate          # the repository's root environment (<repo> = repository root)
cd review_scripts
python run_sims_sphere.py traj5 1         # ~5 min; writes ../Python/Results/python/simulation_results_traj_5_12drop4.pkl
python run_sims_sphere.py traj6 1         # ~5 min
python run_sims_sphere.py arena3 1        # < 1 min
cd ../figures && python make_lisbon_figures.py
```

Or run `python main.py` in `Python/` with the settings in `Structs_sim.py`.

### Trajectories and scenes

`Structs_sim.py`: `sim.traj` (1-6, or `main.py --traj K`) and `sim.scene` (`"lisbon"` or `"arena"`, or `main.py --scene S`). Every trajectory flies in both scenes.

| Trajectory | Lisbon (`RestrictedAirspace`, X500, formation radius 5 m) | Arena (`experiment_arena.py`, Crazyflie, radius 0.8 m, 6 x 6 x 3 m room) |
|---|---|---|
| 1 hover | 10 m past the wall, in the grey band | 1 m from the centre, past the wall |
| 2 circle | 15 m around (-5, 0, 5) m in the wall frame, 0.12 rad/s (1.8 m/s; V16: 0.25 rad/s, faster than v_max) | 1.2 m around the centre, 0.3 rad/s |
| 3, 4 Lissajous (4: 3D) | 5 m, centred on the wall, 0.2 rad/s (up to 2.2 m/s) | 1.2 m, centred on the centre, 0.15 rad/s |
| 5 intrusion, straight border | KMZ red edge 1 | a slice of the Lisbon flight (scaled, the wall at the Lisbon angle) |
| 6 intrusion, border joint | the 34.6 deg notch at red vertex 8 | a slice of the Lisbon flight (scaled, the corner of the two walls at the Lisbon angle) |

Lisbon, trajectories 1-4: the wall frame has its origin on the drones' wall at the trajectory 5 crossing (red edge 1, a straight 6 km edge), x into the grey band (the red border at x = `wall_offset`, 30 m), y along the wall. The target crosses the wall and the formation morphs wall <-> sphere; it never gets closer than 20 m to the red zone.

Arena, trajectories 1-4: the drones' wall is x = `arena_wall_x` (0 m, facing +x; the scaled red border is 4.8 m past it, outside the room), so the target crosses it and the formation morphs wall <-> sphere inside the room. Every arena flight keeps the formation inside the geofence (0.3 m from the room walls and ceiling).

### Real-world simulation of the Crazyflie flights (`main.py --arena` / `--IRL_log`)

The OptiTrack flight `Crazy_Flie/Optitrack/swarm_MPC_voronoi` uses **this simulation's MPC** (`obstacle_avoidance_funcs.py`, the thesis MPC; see its README for the settings and the one departure, the plan start for the real drones). `main.py` runs that flight code on the simulated Crazyflies (`plant.py`, `CrazyfliePlant`, onboard Mellinger) with `flight_replay.py`:

```bash
cd Python
python main.py [--scene arena] [--traj K]       # the simulation, Structs_sim.py
python main.py --arena [--target real] [--traj 5] [--drones 4] [--no-show]
                                                 # only the Formation phase of the flight code with its current
                                                 # settings, no log: the drones start on its formation slots
                                                 # (Results/<code>_dry_run_..._<date>/)
python main.py --IRL_log [LOG] [--no-show]       # a real flight (default: the newest swarm_MPC_voronoi run in
                                                 # Results/), the whole flight from the logged takeoff points,
                                                 # then compared with it
```

As in the geometric Fibonacci simulation, the plain simulation runs only the formation, and `--IRL_log` the whole flight: the drones start at rest on the floor at their logged takeoff points and get the setpoints the flight sent, at the times it sent them, until the motors stop (`flight_replay.simulate_logged`; from `<log>_setpoints.csv`, logged by the flight from 13 Nov, else from the setpoints sampled in the trajectory log). The decisions of the flight code are the flown ones, so the difference measures the drone model (13 Nov flight: 4–6 cm RMS per drone). `--IRL_log --rerun` re-runs the flight code instead (take off, approach to the formation slots, the Formation phase with its formation, MPC and controllers, return to the takeoff points, landing, motors off; `flight_replay.simulate_flight`), each phase starting at its logged time: the formation can then settle in another valid arrangement and the drones return to other takeoff points than in the flight. Both write `<run>/simulation/simulation_results.pkl` (rewatch with `python animation_display.py`) and the comparison (`compare_real.py`: figures with the phases as backgrounds, `metrics.json`). It uses the flight setup of `<log>_meta.json` (scene, always `arena`; drone, always the Crazyflie `cf2x_L250`; target mode and trajectory; formation goals and approach slots; formation and MPC settings; Mellinger position gains read back from the firmware) and the setpoint / formation periods measured in the log. The comparison covers the whole flight (position error per phase). It writes `<log>_sim.csv` next to the log and the comparison figures (`position`, `top_view`, `formation`) to `<run folder>/comparison/`. The same runs, for any flight folder: `Crazy_Flie/Optitrack/flight_sim.py`. The default is trajectory 6 with 5 chasers
and the animation on. The paper runs use `numpy.random.seed(1)`. Seeds 2-5 are the repetitions quoted in the paper
(their `.pkl` files are not copied here. Re-run them with `run_sims_sphere.py traj5 2 Results/python/traj5_s2.pkl`).

## Figures and numbers for a Lisbon-airport results section

| Figure | Content |
|---|---|
| `figures/fig_airspace.pdf` | LPPT red/orange zones, wall 30 m outside the border, margin |
| `figures/fig_mesh.pdf` | morphing mesh: disc, cone, cone->sphere, sphere |
| `figures/fig_lisbon.pdf` | top view, (a) trajectory 5 (12 chasers, 4 lost), (b) trajectory 6 (notch) |
| `figures/fig_traj5_metrics.pdf` | traj. 5: distance to target, min inter-agent distance, speed, goal error |

Key numbers (seed 1, from `sphere_pass_outputs.txt`):

- **Trajectory 5 (12 -> 8 chasers):**
  - Phase times: the target crosses the wall at 13.9 s and the red border at 34.1 s. The chasers are lost at 25 s.
  - Barrier on the wall until 12.2 s (all chasers at s = 30.0 m).
  - Goal-error peak 3.08 m while the sphere closes. Median goal error 0.22 m.
  - Distance to the target: median 4.92 m (p5-p95 4.74-5.07) at 18-25 s, and 4.96 m (4.74-5.03) at 25-31 s.
  - Falling chasers stay at least 1.60 m from active chasers. All goal errors are below 0.5 m 0.61 s after the loss.
  - Minimum inter-agent distance 1.43 m (at 35.6 s, red-border barrier). Envelope 1 m.
  - Peak speed 3.09 m/s (0.2 % of samples above 3 m/s). Median estimate error 6.6 cm.
  - At 40 s, all 8 chasers are on the 1 m margin (1.00-1.01 m from the border).
  - After 30 s, no chaser comes closer than 1.76 m to the target.
  - Repetitions (seeds 2-5): minimum separation after 2 s 1.96-2.31 m. Closest chaser to the target after 30 s 1.38-2.06 m.
- **Trajectory 6 (5 chasers):**
  - Distance to the target from 15 s: median 4.90 m (4.46-5.07).
  - Minimum inter-agent distance 2.28 m, and 4.26 m after 2 s. Median goal error 0.09 m.
  - The target stays in the band (13.3 m from the border at 40 s).
  - Repetitions: minimum separation after 2 s 3.55-5.00 m. In 1 of 4, a goal near the cone apex brought a chaser within 1.89 m of the target.
- **Arena sim. (Crazyflie model, 0.3 m sphere, target avoided), 3 / 4 chasers:**
  - Minimum inter-agent distance 0.89 / 0.85 m. Median goal error 0.12 / 0.13 m.
  - Peak speed 0.40 / 0.46 m/s (v_max 0.48). No envelope violations.
  - But chasers stack vertically (0.83-1.50 m apart, within 0.3 m horizontally) and fly under the target. This is a downwash concern for real Crazyflies.

The real Crazyflie flight of the paper (26 Oct 2026) was flown with the old downwash cylinder. Its figure
(`fig_arena.pdf`) and its numbers are in the paper and the `air-space-deterrent` repository, not here.
