"""Arena simulations (Python, Crazyflie model, 3/4 chasers, spherical 0.3 m envelope, target drone avoided).
usage: python arena_sim_sphere.py [file ...]  (default: Results/python/arena5_3.pkl arena5_4.pkl)"""
import os, sys, pickle, numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Python")) + "/"
files = sys.argv[1:] or ["Results/python/arena5_3.pkl", "Results/python/arena5_4.pkl"]
r_env, vmax, dt = 0.3, 0.48, 0.01
for f in files:
    d = pickle.load(open(R + f, "rb")); P = d["position_history"]; T = d["target_true_hist"][0:3]; V = d["velocity_history"]
    Pd = d["p_d_hist"]; est = d["perceived_target_hist"][:, 0:3]
    n, N = P.shape[0], P.shape[2]
    D = np.linalg.norm(P[:, None] - P[None], axis=2); D[np.eye(n, dtype=bool)] = np.inf
    dmin = D.min(axis=(0, 1))
    r = np.linalg.norm(P - T[None], axis=1); sp = np.linalg.norm(V, axis=1); e = np.linalg.norm(P - Pd, axis=1)
    # vertical stacking (downwash exposure): pairs within 0.3 m horizontally
    dh = np.linalg.norm(P[:, None, :2] - P[None, :, :2], axis=2); dh[np.eye(n, dtype=bool)] = np.inf
    dhT = np.linalg.norm(P[:, :2] - T[None, :2], axis=1); dzT = P[:, 2] - T[None, 2]
    print(f, "n=%d dur=%.2f s" % (n, (N - 1) * dt))
    print("  min d_ij %.3f (t=%.2f) | after 2 s %.3f | time below 0.3 m: %.2f s" % (dmin.min(), dmin.argmin() * dt, dmin[200:].min(), np.sum(dmin < r_env) * dt))
    print("  d_iT min %.3f (t=%.2f) median %.3f p5-p95 %.3f-%.3f | time any chaser < 0.3 m from target %.2f s" %
          (r.min(), np.unravel_index(r.argmin(), r.shape)[1] * dt, np.median(r), np.percentile(r, 5), np.percentile(r, 95), np.sum(r.min(axis=0) < r_env) * dt))
    print("  per-chaser median d_iT", np.round(np.median(r, axis=1), 3))
    print("  goal error median %.3f max %.3f | peak speed %.3f, %.2f%% samples above v_max" % (np.median(e), e.max(), sp.max(), 100 * np.mean(sp > vmax)))
    print("  est error median %.4f" % np.median(np.linalg.norm(est - T[None], axis=1)))
    print("  min horizontal chaser-chaser distance %.3f; time with a pair < 0.3 m horizontally: %.2f s" % (dh.min(), np.sum(dh.min(axis=(0, 1)) < 0.3) * dt))
    dz = np.abs(P[:, None, 2] - P[None, :, 2]); st = dh < 0.3
    if st.any():
        iu = np.triu_indices(n, 1); pairs = [(a, b, st[a, b].sum() * dt, dz[a, b][st[a, b]].min(), dz[a, b][st[a, b]].max()) for a, b in zip(*iu) if st[a, b].any()]
        for a, b, dur, lo, hi in pairs:
            print("  pair %d-%d within 0.3 m horizontally for %.2f s, vertical separation %.2f-%.2f m" % (a + 1, b + 1, dur, lo, hi))
    under = (dhT < 0.3) & (dzT < 0)
    print("  chaser under the target within 0.3 m horizontally: %.2f s (chaser-time), min dz below %.3f" % (under.sum() * dt, (-dzT[under]).min() if under.any() else np.nan))
    print("  min chaser height %.3f" % P[:, 2].min())
