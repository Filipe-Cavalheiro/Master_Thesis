"""
Straight-boundary figures of the thesis (Chapter 6, "Simulation Results:
Straight Boundary"), with the current simulation code.

Scenario: the Lisbon scene, trajectory 2 (the V16 circle, in the frame of a
straight 6 km red edge: the target crosses the wall back and forth and never
reaches the red zone), 12 chasers, 4 of them lost at 10 s, 30 s, random seed 1.
The target flies at TARGET_HEIGHT = 10 m (instead of 5 m) so that no part of the
mesh (wall disc of radius ~1.41 R = 7.1 m, sphere of radius R = 5 m) comes
closer to the ground than the clearance env_clearance (1.5 m): the goals are then
never raised to the clearance height.
Every other setting is the default of Structs_sim.py (the Lisbon settings).

    cd Python/flight_sims/swarm_MPC_voronoi/figures
    python make_straight_wall_figures.py            # simulate (if needed) and draw
    python make_straight_wall_figures.py --rerun    # simulate again

The run is stored in ../Python/Results/python/simulation_results_traj_2_12drop4.pkl
and the figures are written to ../Python/Results/python/ (the file names of
the thesis, Thesis_report/5-Figures/Chapter6/Results/).
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
PY = HERE.parent / "Python"
sys.path.insert(0, str(PY))
os.chdir(PY)   # the simulation modules resolve their paths from here

from Structs_sim import Param, SimulationConfig, param_from_config, simulation_data  # noqa: E402

RESULTS_FILE = "Results/python/simulation_results_traj_2_12drop4.pkl"   # relative to the Python folder
SEED = 1
TARGET_HEIGHT = 10.0    # [m]
CLOSE_UP_STEPS = [100, 300, 700, 1200, 2350, 2800]   # 1, 3, 7, 12, 23.5, 28 s


def scenario_config():
    sim = SimulationConfig()
    sim.scene = "lisbon"
    sim.traj = 2
    sim.lisbon_traj_height = TARGET_HEIGHT
    sim.Tend = 30
    sim.Nsim = round(sim.Tend / sim.dTi) + 1
    sim.t = np.arange(0, sim.Tend + sim.dTi, sim.dTi)
    sim.nD = 12
    sim.sphere_radius = 5.0
    sim.obstacle_avoidance_dim = 1.0
    sim.nD_drop = 4
    sim.nD_drop_time = round(10.0 / sim.dTi)     # 10 s
    return sim


def simulate():
    import main as sim_main
    import sim_meta
    sim_meta.seed_rng(SEED)
    param = Param(scenario_config())
    param.display.verbose = False
    param.display.store_data = True
    param.display.results_file = RESULTS_FILE
    sim_main.main(param)


def draw():
    import animation_display as ad
    with open(PY / RESULTS_FILE, "rb") as f:
        d = pickle.load(f)
    param = param_from_config(d["sim"])
    # one instant per formation: wall, cone, sphere, sphere after the loss, cone, wall
    param.display.close_up_steps = CLOSE_UP_STEPS
    data = simulation_data(d)

    from Structs_sim import Voronoi
    from voronoi_mesh_funcs import pre_compute_desired_pos
    voronoi = Voronoi(param)
    _, _, voronoi = pre_compute_desired_pos(voronoi, param)

    ad.detailed_close_up(data, voronoi, param)
    ad.composed_plot(data, voronoi, param)
    ad.radial_hist_plot(data, param)
    ad.inter_agent_dist_plot(data, param)
    ad.velocity_plot(data, param)
    ad.low_level_tracking_error_plot(data, param)
    ad.trajectory_plot(data, param)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", action="store_true", help="simulate again even if the run is stored")
    args = ap.parse_args()
    if args.rerun or not (PY / RESULTS_FILE).exists():
        simulate()
    draw()
