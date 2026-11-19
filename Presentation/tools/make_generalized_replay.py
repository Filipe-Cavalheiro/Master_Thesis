"""
Data of the Results replay (assets/data/generalized_sim_12agents.npz): the thesis Ch. 6
straight-boundary scenario, simulated with Python/flight_sims/swarm_MPC_voronoi (the thesis mesh
controller: cone rim on the wall, no mesh vertex or goal behind it).

Lisbon scene, trajectory 2 (15 m circle around (-5, 0, 5) in the wall frame) at V16's 0.25 rad/s,
so the phases fall at the times of the thesis (disc 1 s, cone 2.5 s, sphere 4 s, disc 13 s);
12 chasers, 4 lost at 10 s; formation radius 5 m and 1 m envelope (the Lisbon values).
Stored in the wall frame (x past the wall, y along it, z up), every 5th step (0.05 s).

usage (repository environment, ~5 min):
  .venv/bin/python Presentation/tools/make_generalized_replay.py [SEED]      (default seed 1)
"""
import os
import pickle
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", "..", "Python", "flight_sims", "swarm_MPC_voronoi", "Python"))
OUT = os.path.abspath(os.path.join(HERE, "..", "assets", "data", "generalized_sim_12agents.npz"))
sys.path.insert(0, SIM)
os.chdir(SIM)
import Structs_sim as SS  # noqa: E402


def local_traj(traj, height=5.0, _orig=SS.lisbon_local_traj):
    if traj != 2:
        return _orig(traj, height)
    w, r = 0.25, 15.0
    return {"pos": lambda t: np.array([-5 + r * np.sin(w * t), r * np.cos(w * t), height]),
            "vel": lambda t: np.array([r * w * np.cos(w * t), -r * w * np.sin(w * t), 0.0])}


SS.lisbon_local_traj = local_traj
import main as M  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
np.random.seed(seed)
sim = SS.SimulationConfig()
sim.scene, sim.traj = "lisbon", 2
sim.nD, sim.nD_drop, sim.nD_drop_time = 12, 4, 1000       # 4 lost at step 1000 (10 s)
sim.sphere_radius, sim.obstacle_avoidance_dim = 5.0, 1.0
param = SS.Param(sim)
param.display.verbose = False
param.display.store_data = True
pkl = os.path.join(tempfile.mkdtemp(), "simulation_results.pkl")
param.display.results_file = pkl
M.main(param)

d = pickle.load(open(pkl, "rb"))
o2, n, tan = param.traj.wall_frame_origin()
Rm = np.array([[n[0], tan[0], 0.0], [n[1], tan[1], 0.0], [0.0, 0.0, 1.0]])
o = np.array([o2[0], o2[1], 0.0]).reshape(3, 1)


def wall(X):
    """World (..., 3, T) -> wall frame: R^T (X - o)."""
    return np.einsum("ji,...jk->...ik", Rm, X - o)


idx = np.arange(0, d["position_history"].shape[2], 5)
np.savez_compressed(
    OUT,
    t=idx * sim.dTi,
    P=wall(d["position_history"])[:, :, idx],
    pd=wall(d["p_d_hist"])[:, :, idx],
    T=wall(d["target_true_hist"][:3])[:, idx],
    active=np.asarray(d["active_ids"]),
    R=5.0,
    drop_time=10.0,
    source=f"Python/flight_sims/swarm_MPC_voronoi, Lisbon trajectory 2 at 0.25 rad/s, 12 chasers, "
           f"4 lost at 10 s, seed {seed}, every 5th step (Presentation/tools/make_generalized_replay.py)",
)
print("wrote", OUT)
