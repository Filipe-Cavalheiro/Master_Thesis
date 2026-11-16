"""Export the Lisbon-airport simulation (trajectory 5, 12 chasers, 4 lost at 25 s) for the slides.

Run with the repository's root environment, from anywhere:
    python Presentation/assets/data/make_lisbon_data.py

Reads  Python/flight_sims/swarm_MPC_voronoi/Python/Results/python/simulation_results_traj_5_12drop4.pkl
Writes Presentation/assets/data/lisbon_traj5.npz, in the wall frame of the crossed border edge:
    x along the wall normal (into the prohibited zone, wall at x = 0), y along the wall, z up.
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = HERE.parents[2] / "Python" / "flight_sims" / "swarm_MPC_voronoi" / "Python"
sys.path.insert(0, str(PY))
os.chdir(PY)

from Structs_sim import SimulationConfig  # noqa: E402
from restricted_airspace import RestrictedAirspace  # noqa: E402

with open(PY / "Results" / "python" / "simulation_results_traj_5_12drop4.pkl", "rb") as f:
    d = pickle.load(f)

STEP = 5                                             # 100 Hz -> 20 Hz
P = d["position_history"]                            # (n, 3, N)
T = d["target_true_hist"][0:3]                       # (3, N)
G = d["p_d_hist"]                                    # (n, 3, N), NaN once a chaser is lost
N = T.shape[1]
t = np.arange(N) * 0.01

airspace = RestrictedAirspace(SimulationConfig())
c = np.array([1696.5, 6725.0])                       # wall crossing of trajectory 5 (ENU, m)
n = np.asarray(airspace.wall_normal(c), float)[:2]
n /= np.linalg.norm(n)
tg = np.array([-n[1], n[0]])
R = np.array([n, tg])                                # ENU xy -> wall frame


def to_wall(X):                                      # (..., 3, N) -> same shape, wall frame
    Y = X.copy()
    xy = np.moveaxis(X[..., 0:2, :], -2, -1) - c     # (..., N, 2)
    Y[..., 0:2, :] = np.moveaxis(xy @ R.T, -1, -2)
    return Y


# signed distance to the red border along the normal through c: wall at wall_offset, border at 0
xs = np.linspace(-60.0, 60.0, 241)
sd = airspace.red.signed_distance(c + xs[:, None] * n)
x_red = float(np.interp(0.0, -sd, xs))               # sd decreases into the zone
alive = ~np.isnan(G).any(axis=1)

np.savez_compressed(
    HERE / "lisbon_traj5.npz",
    t=t[::STEP], P=to_wall(P)[..., ::STEP], T=to_wall(T)[..., ::STEP], G=to_wall(G)[..., ::STEP],
    alive=alive[:, ::STEP], x_red=x_red, wall_offset=float(airspace.wall_offset),
)
print("samples", t[::STEP].size, "x_red", x_red, "wall_offset", airspace.wall_offset,
      "lost", [i for i in range(P.shape[0]) if not alive[i].all()])
Tw = to_wall(T)
print("target x: start %.1f end %.1f | y: %.1f -> %.1f | z %.1f" % (Tw[0, 0], Tw[0, -1], Tw[1, 0], Tw[1, -1], Tw[2, 0]))
print("crosses wall at t=%.1f, red border at t=%.1f" % (t[np.argmax(Tw[0] > 0)], t[np.argmax(Tw[0] > x_red)]))
Pw = to_wall(P)
print("chaser ranges x %.1f..%.1f y %.1f..%.1f z %.1f..%.1f" % (np.nanmin(Pw[:, 0]), np.nanmax(Pw[:, 0]), np.nanmin(Pw[:, 1]), np.nanmax(Pw[:, 1]), np.nanmin(Pw[:, 2]), np.nanmax(Pw[:, 2])))
