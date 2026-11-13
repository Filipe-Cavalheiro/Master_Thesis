"""Extra numbers for Sec. VI-A (sphere/ADMM pass) and a summary over repetitions (seeds).
usage: python lisbon_extras_sphere.py traj5|traj6 FILE [FILE ...]   (files relative to Python/)"""
import os, sys, pickle, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Python"))
sys.path.insert(0, REPO); os.chdir(REPO)
from Structs_sim import SimulationConfig
from restricted_airspace import RestrictedAirspace
sim = SimulationConfig(); air = RestrictedAirspace(sim)
scen, files = sys.argv[1], sys.argv[2:]
for f in files:
    d = pickle.load(open(f, "rb"))
    P, V, Pd = d["position_history"], d["velocity_history"], d["p_d_hist"]; T = d["target_true_hist"][0:3]
    alive = ~np.isnan(Pd).any(axis=1); n, N = P.shape[0], P.shape[2]; t = np.arange(N) / 100
    D = np.linalg.norm(P[:, None] - P[None], axis=2); D[np.eye(n, dtype=bool)] = np.inf
    D[~(alive[:, None] & alive[None])] = np.inf; ms = D.min(axis=(0, 1))
    r = np.where(alive, np.linalg.norm(P - T[None], axis=1), np.nan)
    e = np.linalg.norm(P - Pd, axis=1); sp = np.where(alive, np.linalg.norm(V, axis=1), np.nan)
    i, k = np.unravel_index(np.nanargmin(r[:, 200:]), r[:, 200:].shape); k += 200
    print(f"{f}: min d_ij {ms.min():.2f} (t={t[ms.argmin()]:.2f}) after 2 s {ms[200:].min():.2f} | min d_iT(t>2 s) {r[i, k]:.2f} at {t[k]:.2f} s (chaser {i}) "
          f"| median e_g {np.nanmedian(e):.3f} max {np.nanmax(e):.2f} | peak speed {np.nanmax(sp):.2f}")
    if scen == "traj5":
        sdT = air.red_signed_distance(T.T); kh = np.argmax(sdT < sim.wall_offset + sim.sphere_radius / 2)
        s_ch = np.array([air.red_signed_distance(P[j].T) for j in range(n)])
        act = np.flatnonzero(alive[:, -1])
        print(f"   target R/2 from wall at {t[kh]:.2f} s; chaser s before then {s_ch[:, 200:kh].min():.4f}-{s_ch[:, 200:kh].max():.4f} (t>2 s); "
              f"min s between {t[kh]:.1f} s and wall crossing {s_ch[:, kh:np.argmax(sdT < sim.wall_offset)].min():.2f}")
        print(f"   final s of active chasers {s_ch[act, -1].min():.3f}-{s_ch[act, -1].max():.3f}; "
              f"min d_iT after 30 s {np.nanmin(r[:, 3000:]):.2f}; time any chaser <1 m from target {np.sum(np.nanmin(r, axis=0) < 1) / 100:.2f} s; "
              f"min sep 25-31 s {ms[2500:3100].min():.2f}; min sep after 30 s {ms[3000:].min():.2f} at {t[3000 + ms[3000:].argmin()]:.2f}")
        k0 = 2800; k1 = 3300
        print(f"   goal error 28-33 s median {np.nanmedian(e[act][:, k0:k1]):.3f}, p95 {np.nanpercentile(e[act][:, k0:k1], 95):.3f}; "
              f"max goal error after 30 s {np.nanmax(e[act][:, 3000:]):.2f} at {t[3000 + np.nanargmax(np.nanmax(e[act][:, 3000:], axis=0))]:.2f} s")
        print(f"   goal error 36-40 s median {np.nanmedian(e[act][:, 3600:]):.3f}; max speed after 2 s before wall {np.nanmax(sp[:, 200:1380]):.2f}; "
              f"peak speed time {t[np.nanargmax(np.nanmax(sp, axis=0))]:.2f}; frac>vmax {100*np.nanmean(sp[alive] > 3):.2f}%")
