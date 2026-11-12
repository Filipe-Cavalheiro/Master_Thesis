"""
Re-run the simulations of the paper with the spherical safety envelope and
the standard scaled ADMM projection b_o = Pi(A_o z + lambda_o / rho_o)
(sphere/ADMM pass, Nov 2026).

usage: python run_sims_sphere.py SCENARIO SEED [OUT] [--reset]
  SCENARIO: traj5   (Lisbon trajectory 5, 12 chasers, 4 lost at 25 s)
            traj6   (Lisbon trajectory 6, 5 chasers)
            arena3 / arena4  (arena trajectory 5, Crazyflie model, 3 / 4 chasers, target drone avoided)
  SEED    : numpy seed (initial sites, lost chasers, measurement noise)
  OUT     : results file relative to Python/ (default Results/python/<paper name>)
  --reset : reset the ADMM multipliers every control step (default in Lisbon: carried over)
"""
import os, sys
import numpy as np
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Python")
REPO = os.path.abspath(REPO)
sys.path.insert(0, REPO); os.chdir(REPO)
from Structs_sim import SimulationConfig, Param
import main as M

args = [a for a in sys.argv[1:] if not a.startswith("--")]
scen, seed = args[0], int(args[1])
default_out = {"traj5": "Results/python/simulation_results_traj_5_12drop4.pkl",
               "traj6": "Results/python/simulation_results_traj_6.pkl",
               "arena3": "Results/python/arena5_3.pkl",
               "arena4": "Results/python/arena5_4.pkl"}[scen]
out = args[2] if len(args) > 2 else default_out
np.random.seed(seed)

sim = SimulationConfig()
if scen == "traj5":
    sim.traj = 5; sim.nD = 12; sim.nD_drop = 4; sim.nD_drop_time = 2500
elif scen == "traj6":
    sim.traj = 6; sim.nD = 5
else:
    sim.arena_experiment = True; sim.arena_traj = 5; sim.arena_target_drone = True
    sim.nD = 3 if scen == "arena3" else 4
if "--reset" in sys.argv:
    sim.admm_warm_multipliers = False
param = Param(sim)
param.display.verbose = False
param.display.store_data = True
param.display.results_file = out
print(f"{scen} seed={seed} out={out} envelope r={param.sim.obstacle_avoidance_dim} "
      f"warm={param.sim.admm_warm_multipliers}", flush=True)
M.main(param)
