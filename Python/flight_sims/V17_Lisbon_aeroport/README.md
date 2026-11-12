# V17: Lisbon-airport simulation (spherical envelope, scaled ADMM)

V17 is the Python simulation used for the Lisbon-airport results of the TAES paper
("Non-Destructive Deterrence of Drone Intrusions into Restricted Airspace with Morphing Swarm
Formations"), copied from the `air-space-deterrent` repository (`Python/`, 8 Nov 2026). It differs from V16 in:

- **Morphing formation and restricted airspace**: wall at 30 m outside the red zone of the LPPT
  KMZ. The mesh blends disc -> cone -> sphere -> disc on the red border. A geodesic CVT (Dijkstra) assigns the goals.
- **Distributed Bernstein MPC solved with ADMM.** Velocity, acceleration, separation, airspace and obstacle
  constraints are all projections.
- **Spherical safety envelope only.** Each chaser keeps a ball of radius `obstacle_avoidance_dim`
  (1 m in Lisbon, 0.3 m in the Crazyflie arena) from the other chasers and from the target
  when it is avoided. The old downwash cylinder and ellipsoid are still available as legacy options in `Structs_sim.py`, but nothing uses them.
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

Or run `python main.py` in `Python/` with the settings in `Structs_sim.py`. The default is trajectory 6 with 5 chasers
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
