"""Recompute the numbers quoted in the Lisbon results paragraphs (trajectories 5 and 6).
Sphere/ADMM pass: paths relative to this folder; optional args TRAJ5_FILE TRAJ6_FILE (relative to Python/)."""
import os, sys, pickle
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Python"))
F5 = sys.argv[1] if len(sys.argv) > 1 else "Results/python/simulation_results_traj_5_12drop4.pkl"
F6 = sys.argv[2] if len(sys.argv) > 2 else "Results/python/simulation_results_traj_6.pkl"
sys.path.insert(0, REPO); os.chdir(REPO)
from Structs_sim import SimulationConfig
from restricted_airspace import RestrictedAirspace
sim = SimulationConfig(); air = RestrictedAirspace(sim)
pct = lambda x: "median %.2f (p5-p95 %.2f-%.2f)" % (np.nanmedian(x), np.nanpercentile(x, 5), np.nanpercentile(x, 95))

def load(name):
    d = pickle.load(open(name, "rb"))
    P, V, Pd = d["position_history"], d["velocity_history"], d["p_d_hist"]
    T, VT = d["target_true_hist"][0:3], d["target_true_hist"][3:6]
    alive = ~np.isnan(Pd).any(axis=1)
    return d, P, V, Pd, T, VT, alive

def minsep(P, alive):
    n = P.shape[0]; D = np.linalg.norm(P[:, None] - P[None], axis=2)
    D[np.eye(n, dtype=bool)] = np.inf; D[~(alive[:, None] & alive[None])] = np.inf
    return D.min(axis=(0, 1))

print("===== trajectory 5")
d, P, V, Pd, T, VT, alive = load(F5)
n, N = P.shape[0], P.shape[2]; t = np.arange(N) / 100
r = np.where(alive, np.linalg.norm(P - T[None], axis=1), np.nan)
e = np.linalg.norm(P - Pd, axis=1)
sdT = air.red_signed_distance(T.T)
kw = np.argmax(sdT < sim.wall_offset); kr = np.argmax(sdT < 0)
print("wall crossing %.2f s, red crossing %.2f s" % (t[kw], t[kr]))
s_ch = np.array([air.red_signed_distance(P[i].T) for i in range(n)])
print("before wall (t<%.1f): chaser s range %.2f-%.2f; chaser-target distance at t=0: %.1f-%.1f, at wall crossing median %.2f" %
      (t[kw], s_ch[:, :kw].min(), s_ch[:, :kw].max(), np.nanmin(r[:, 0]), np.nanmax(r[:, 0]), np.nanmedian(r[:, kw])))
sp = np.linalg.norm(V, axis=1)
print("speed before wall: median %.2f m/s (after 2 s)" % np.median(sp[:, 200:kw]))
print("max goal error %.2f at t=%.2f" % (np.nanmax(e), t[np.nanargmax(np.nanmax(e, axis=0))]))
w = (t >= 18) & (t < 25); print("18-25 s distance", pct(r[:, w]), "| speed", pct(sp[:, w]), "| target speed %.2f" % np.linalg.norm(VT[:, 2000]))
kd = 2500
lost = np.flatnonzero(~alive[:, -1]); act = np.flatnonzero(alive[:, -1])
# falling chasers: time to ground and min distance to active chasers
for i in lost:
    zg = np.flatnonzero(P[i, 2, kd:] <= 0.05)
    dmin = min(np.linalg.norm(P[i, :, kd:] - P[j, :, kd:], axis=0).min() for j in act)
    print(f" lost {i}: reaches z<=0.05 m after {zg[0]/100 if len(zg) else np.nan:.2f} s, min dist to active {dmin:.2f}")
dj = np.linalg.norm(Pd[act][:, :, kd + 1] - Pd[act][:, :, kd - 1], axis=1); print("goal jump at loss: max %.2f" % dj.max())
after = np.flatnonzero(np.nanmax(e[act][:, kd:], axis=0) < 0.5)
print("all goal errors < 0.5 m from %.2f s after the loss (first instant)" % (after[0] / 100))
w = (t >= 28) & (t < 33); print("28-33 s median goal error %.3f" % np.nanmedian(e[act][:, w]))
w = (t >= 25) & (t < 31); print("25-31 s distance", pct(r[:, w]))
# closest approach to the target near the red border
k0 = 3000; i, k = np.unravel_index(np.nanargmin(r[:, k0:]), r[:, k0:].shape); k += k0
print("closest approach after 30 s: chaser %d, %.2f m at %.2f s; below 1 m for %.2f s" % (i, r[i, k], t[k], np.sum(r[i, k0:] < 1) / 100))
# goal distance to the target path (line through the target positions, horizontal)
path = T[:2, -1] - T[:2, 0]; path /= np.linalg.norm(path)
for kk in [3300, 3340, 3400]:
    g = Pd[act][:, :, kk]; rel = g[:, :2] - T[:2, 0]
    lat = np.abs(rel[:, 0] * path[1] - rel[:, 1] * path[0]); dzp = np.abs(g[:, 2] - T[2, 0])
    print(f" t={kk/100}: min goal distance to target path (3D, path at fixed height) {np.min(np.hypot(lat, dzp)):.2f}")
print("at 40 s: target sd %.2f; active chasers s:" % sdT[-1], np.round(np.sort(s_ch[act, -1]), 2), "goal s:", np.round(air.red_signed_distance(Pd[act][:, :, -1]), 2))
ms = minsep(P, alive)
print("min sep: start %.2f; 5-13.9 s (barrier) median %.2f; 18-25 s range %.2f-%.2f; 25-31 s min %.2f; overall %.2f at %.2f s; after 2 s %.2f" %
      (ms[0], np.median(ms[500:kw]), ms[1800:2500].min(), ms[1800:2500].max(), ms[2500:3100].min(), ms.min(), t[ms.argmin()], ms[200:].min()))
ka = np.argmin(ms[3000:]) + 3000; print("min sep after 30 s %.2f at %.2f s" % (ms[ka], t[ka]))
spa = sp[alive]; print("peak speed %.3f; fraction of active samples above v_max %.2f%%" % (spa.max(), 100 * np.mean(spa > sim.v_max)))
est = d["perceived_target_hist"][:, 0:3]; print("median estimate error %.4f" % np.median(np.linalg.norm(est - T[None], axis=1)[alive]))

print("===== trajectory 6")
d, P, V, Pd, T, VT, alive = load(F6)
t = np.arange(P.shape[2]) / 100
r = np.linalg.norm(P - T[None], axis=1)
w = t >= 15; print("from 15 s distance", pct(r[:, w]))
i, k = np.unravel_index(np.argmin(r), r.shape); print("closest approach: chaser %d %.2f m at %.2f s" % (i, r[i, k], t[k]))
gT = np.linalg.norm(Pd - T[None], axis=1); print(" its goal distance to target around then: min %.2f in [12,16] s" % gT[i, 1200:1600].min())
ms = minsep(P, alive); print("min sep %.2f, after 2 s %.2f; median goal error %.3f" % (ms.min(), ms[200:].min(), np.median(np.linalg.norm(P - Pd, axis=1))))
sdT = air.red_signed_distance(T.T); print("target speed %.2f; wall crossing %.2f s; red crossing %.2f s" % (np.linalg.norm(VT[:, 100]), t[np.argmax(sdT < 30)], t[np.argmax(sdT < 0)]))
