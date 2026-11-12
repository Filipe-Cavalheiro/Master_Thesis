"""
Dashboard video of a flight vs its simulation, in the style of the V17
animation_display.py animations (trajectory_animate / velocity_animate:
trails that grow with time, the current position marked, curves that grow
with time), composed next to the overlay video of flight_overlay.py.

Layout (1920x1080 by default):
    +-------------------------------+-------------+
    |                               |  3D scene:  |
    |  overlay video                |  arena,     |
    |  (real video + sim drone)     |  camera     |
    |                               |  path and   |
    |                               |  frustum,   |
    +---------------+---------------+  real / sim |
    | z(t) real,    | real - sim    +-------------+
    | sim, setpoint | error [cm]    |  top view   |
    |               | x, y, z, |e|  |  [cm]       |
    +---------------+---------------+-------------+

Used by flight_overlay.py (--dashboard, on by default); it only needs the
tracks as read there ({id: {"t", "p", ("rpy")}}) and the camera track.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

# same colours as the overlay (BGR there, RGB here)
C_REAL = (60 / 255, 210 / 255, 80 / 255)
C_SIM = (255 / 255, 140 / 255, 30 / 255)
C_SP = (0.45, 0.45, 0.45)
C_CAM = (0.35, 0.45, 0.85)
PHASE_PALETTE = ("#dbe9f6", "#e5f3e0", "#efe3f6", "#f6e3db", "#f6f2d8", "#e0f1f1")
SHORT = {"Main Controller": "Figure 8", "Return to origin": "Return", "Return to takeoff points": "Return"}


def phase_names(ph):
    """Phases of the log in time order (without EOF)."""
    return [n for n, _ in sorted(ph.items(), key=lambda kv: kv[1]) if n and n != "EOF"]
AXES = ("x", "y", "z")


def _interp(tr, t):
    """Positions of a track at the times t (NaN outside of it)."""
    return np.column_stack([np.interp(t, tr["t"], tr["p"][:, a], left=np.nan, right=np.nan) for a in range(3)])


def _fig_to_bgr(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def _frustum(T_wc, K, w, h, depth):
    """Camera centre and the 4 image corners at `depth` [m], world frame."""
    Kinv = np.linalg.inv(K)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ Kinv.T * depth
    return T_wc[:3, 3], corners @ T_wc[:3, :3].T + T_wc[:3, 3]


class Dashboard:
    def __init__(self, real, sim, ph, cam, K, video_size, t_range, size=(1920, 1080), trail=None,
                 sim_setpoint=None, dpi=100, position_panel="auto"):
        """
        real, sim   {id: {"t", "p"}} tracks in flight-log time (the sim on the same time base)
        ph          phase start times {name: t, ..., "EOF": t}, any phases (from the log)
        cam         CameraTrack (phone rigid body), for the camera path in the 3D view
        K           intrinsics of the video (for the frustum)
        t_range     (t_a, t_b) flight times shown on the time axes
        trail       trail length [s] in the 3D / top views (None = whole history, as V17)
        sim_setpoint {id: {"t", "p"}} position setpoint of the sim (dashed), optional
        position_panel "z", "xyz" or "auto" (xyz when the drones move > 0.3 m horizontally)
        """
        self.real, self.sim, self.ph, self.K = real, sim, ph, K
        self.vw, self.vh = video_size
        self.W, self.H = size
        self.trail = trail
        self.t_a, self.t_b = t_range
        self.ids = [i for i in real if i in sim]
        self.names = phase_names(ph)
        self.colors = {n: PHASE_PALETTE[k % len(PHASE_PALETTE)] for k, n in enumerate(self.names)}
        if position_panel == "auto":
            span = max([np.ptp(real[i]["p"][:, :2], axis=0).max() for i in self.ids] + [0.0])
            position_panel = "xyz" if span > 0.3 else "z"
        self.position_panel = position_panel

        # video panel: as wide as possible in 2/3 of the width, at most 2/3 of the height
        s = min(2 * self.W / 3 / self.vw, 2 * self.H / 3 / self.vh)
        self.pw, self.ph_px = int(round(self.vw * s)) // 2 * 2, int(round(self.vh * s)) // 2 * 2
        self.right_w = self.W - self.pw

        # real-vs-sim error on the real samples
        self.err = {}
        for i in self.ids:
            t = real[i]["t"]
            e = real[i]["p"] - _interp(sim[i], t)
            self.err[i] = {"t": t, "e": e, "n": np.linalg.norm(e, axis=1)}

        # origin of the top view: the take-off spot of the first drone
        p0 = real[self.ids[0]]["p"][0] if self.ids else np.zeros(3)
        self.origin = p0

        self._build_bottom(dpi, sim_setpoint)
        self._build_right(dpi, cam)

    # ------------------------------------------------------------
    # figures
    # ------------------------------------------------------------
    def _span(self, k):
        n = self.names[k]
        t1 = self.ph[self.names[k + 1]] if k + 1 < len(self.names) else self.ph.get("EOF", self.t_b)
        return self.ph[n], t1

    def _shade(self, ax):
        for k, n in enumerate(self.names):
            ax.axvspan(*self._span(k), color=self.colors[n], zorder=0, lw=0)

    def _build_bottom(self, dpi, sim_setpoint):
        w, h = self.pw, self.H - self.ph_px
        self.fig_b = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        axes = self.position_panel == "xyz" and (0, 1, 2) or (2,)
        gs = self.fig_b.add_gridspec(len(axes), 2, left=0.065, right=0.99, bottom=0.17, top=0.86,
                                     wspace=0.2, hspace=0.12)
        apos = [self.fig_b.add_subplot(gs[r, 0]) for r in range(len(axes))]
        ae = self.fig_b.add_subplot(gs[:, 1])
        for ax in (*apos, ae):
            self._shade(ax)
            ax.set_xlim(self.t_a, self.t_b)
            ax.grid(True, alpha=0.4)
        for ax in apos[:-1]:
            ax.tick_params(labelbottom=False)
        for ax in (apos[-1], ae):
            ax.set_xlabel("Time [s]")
        narrow = 0
        for k, n in enumerate(self.names):
            t0, t1 = self._span(k)
            # short phases (8traj: return, land): staggered so that their labels do not overlap
            small_phase = (t1 - t0) < 0.12 * (self.t_b - self.t_a)
            y = 1.02 + (0.16 if small_phase and narrow % 2 else 0.0)
            narrow += small_phase
            apos[0].text(0.5 * (t0 + t1), y, SHORT.get(n, n), transform=apos[0].get_xaxis_transform(),
                         ha="center", va="bottom", fontsize=8 if small_phase else 9)

        # position(t): full curves faint, growing solid curves on top (V17 velocity_animate)
        self.lz = []
        small = len(axes) > 1
        for ax, a in zip(apos, axes):
            lo, hi = (-0.03, 0.1) if a == 2 else (np.inf, -np.inf)
            for i in self.ids:
                for tr, c, lab in ((self.real[i], C_REAL, "real"), (self.sim[i], C_SIM, "sim")):
                    ax.plot(tr["t"], tr["p"][:, a], color=c, lw=1, alpha=0.2)
                    line, = ax.plot([], [], color=c, lw=1.6 if small else 2,
                                    label=f"CF{int(i):02d} {lab}" if len(self.ids) > 1 else lab)
                    self.lz.append((line, tr, a))
                    lo, hi = min(lo, np.nanmin(tr["p"][:, a])), max(hi, np.nanmax(tr["p"][:, a]))
                if sim_setpoint and i in sim_setpoint:
                    sp = sim_setpoint[i]
                    ax.plot(sp["t"], sp["p"][:, a], color=C_SP, lw=1.0 if small else 1.2, ls="--",
                            label="setpoint" if i == self.ids[0] else "_nolegend_")
            pad = 0.12 * (hi - lo) + 0.02
            ax.set_ylim(lo - (pad if a != 2 else 0.0), hi + pad)
            ax.set_ylabel(f"{AXES[a]} [m]", fontsize=9 if small else 10)
            if small:
                ax.tick_params(labelsize=8)
        apos[0].legend(loc="upper right", fontsize=7 if small else 8, ncol=3 if small else 1)
        self.cz = [ax.axvline(self.t_a, color="k", lw=1) for ax in apos]

        # error components and norm [cm]
        emax = 1.0
        self.le = []
        styles = [("x", 0, "#d62728"), ("y", 1, "#2ca02c"), ("z", 2, "#1f77b4")]
        for i in self.ids:
            E = self.err[i]
            for name, a, c in styles:
                line, = ae.plot([], [], color=c, lw=1.3, label=f"$e_{name}$")
                self.le.append((line, E["t"], 100 * E["e"][:, a]))
            line, = ae.plot([], [], color="k", lw=2, label=r"$\|e\|$")
            self.le.append((line, E["t"], 100 * E["n"]))
            emax = max(emax, 100 * np.nanmax(E["n"]))
        emin = min([0.0] + [100 * np.nanmin(self.err[i]["e"]) for i in self.ids])
        ae.axhline(0, color="k", lw=0.6)
        ae.set_ylim(1.15 * emin - 0.35 * emax, 1.15 * emax)   # room below for the legend
        ae.set_ylabel("real - sim [cm]")
        ae.legend(loc="lower left", fontsize=8, ncol=4)
        self.ce = ae.axvline(self.t_a, color="k", lw=1)
        self.etext = ae.text(0.99, 0.96, "", transform=ae.transAxes, ha="right", va="top", fontsize=9,
                             bbox=dict(fc="white", ec="0.7", alpha=0.9))

    def _build_right(self, dpi, cam):
        w, h = self.right_w, self.H
        self.fig_r = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        self.ax3 = self.fig_r.add_axes([0.0, 0.40, 1.0, 0.60], projection="3d")
        self.axt = self.fig_r.add_axes([0.17, 0.05, 0.78, 0.31])

        # --- 3D: arena floor, camera path, drones (V17 trajectory_animate)
        ax = self.ax3
        m = (cam.t >= self.t_a) & (cam.t <= self.t_b)
        cp = cam.pos[m] if m.any() else cam.pos
        allp = np.vstack([cp] + [self.real[i]["p"] for i in self.ids] + [self.sim[i]["p"] for i in self.ids])
        lo, hi = np.nanmin(allp, 0), np.nanmax(allp, 0)
        lo[:2] -= 0.3
        hi[:2] += 0.3
        lo[2], hi[2] = 0.0, max(hi[2] + 0.2, 1.0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.view_init(elev=28, azim=-60)
        gx = np.arange(np.floor(lo[0] * 2) / 2, hi[0] + 1e-9, 0.5)
        gy = np.arange(np.floor(lo[1] * 2) / 2, hi[1] + 1e-9, 0.5)
        for x in gx:
            ax.plot([x, x], [lo[1], hi[1]], [0, 0], color="0.8", lw=0.6)
        for y in gy:
            ax.plot([lo[0], hi[0]], [y, y], [0, 0], color="0.8", lw=0.6)
        ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], color=C_CAM, lw=0.8, alpha=0.35)
        self.cam_pt, = ax.plot([], [], [], "o", color=C_CAM, ms=5, label="camera")
        self.cam_fr = [ax.plot([], [], [], color=C_CAM, lw=1)[0] for _ in range(8)]
        self.ray = [ax.plot([], [], [], color=c, lw=0.8, ls=":")[0] for c in (C_REAL, C_SIM)]
        self.d3 = []
        for i in self.ids:
            r0 = self.real[i]["p"][0]
            ax.plot([r0[0]], [r0[1]], [r0[2]], "x", color="k", ms=6)
            arts = {}
            for key, c in (("real", C_REAL), ("sim", C_SIM)):
                arts[key + "_l"], = ax.plot([], [], [], color=c, lw=2)
                arts[key + "_p"], = ax.plot([], [], [], "o", color=c, ms=7, mec="k", mew=0.6, label=key)
            self.d3.append((i, arts))
        ax.legend(loc="upper right", fontsize=8)
        self.title3 = ax.set_title("", fontsize=10)

        # --- top view, centred on the take-off spot, cm
        a = self.axt
        r = 1.0
        for i in self.ids:
            for tr in (self.real[i], self.sim[i]):
                d = 100 * (tr["p"][:, :2] - self.origin[:2])
                r = max(r, np.nanmax(np.abs(d)))
        r = 1.15 * r
        # large motion (8traj): metres, else centimetres
        self.tv_scale = 1.0 if r > 100 else 100.0
        u = "m" if r > 100 else "cm"
        r = r / 100 * self.tv_scale
        a.set_xlim(-r, r)
        a.set_ylim(-r, r)
        a.set_aspect("equal")
        a.grid(True, alpha=0.4)
        a.set_xlabel(f"x - x$_0$ [{u}]")
        a.set_ylabel(f"y - y$_0$ [{u}]")
        a.set_title("top view", fontsize=10)
        a.plot([0], [0], "x", color="k", ms=8)
        self.dt = []
        for i in self.ids:
            arts = {}
            for key, c in (("real", C_REAL), ("sim", C_SIM)):
                arts[key + "_l"], = a.plot([], [], color=c, lw=1.5)
                arts[key + "_p"], = a.plot([], [], "o", color=c, ms=8, mec="k", mew=0.6)
            arts["link"], = a.plot([], [], color="k", lw=1)
            self.dt.append((i, arts))

    # ------------------------------------------------------------
    # per frame
    # ------------------------------------------------------------
    def _window(self, tr, t):
        a = -np.inf if self.trail is None else t - self.trail
        m = (tr["t"] >= a) & (tr["t"] <= t)
        return tr["p"][m]

    def _now(self, tr, t):
        if len(tr["t"]) == 0 or t < tr["t"][0]:
            return tr["p"][0] if len(tr["t"]) and t >= self.t_a else None
        return _interp(tr, np.array([min(t, tr["t"][-1])]))[0]

    def update(self, t, T_wc):
        # bottom: growing curves and cursor
        for line, tr, a in self.lz:
            m = tr["t"] <= t
            line.set_data(tr["t"][m], tr["p"][m, a])
        for line, tt, y in self.le:
            m = tt <= t
            line.set_data(tt[m], y[m])
        for c in (*self.cz, self.ce):
            c.set_xdata([t, t])
        txt = []
        for i in self.ids:
            E = self.err[i]
            m = (E["t"] <= t) & (E["t"] >= (self.ph[self.names[0]] if self.names else -np.inf))
            if m.any():
                txt.append(f"CF{int(i):02d}  |e| = {100 * E['n'][m][-1]:4.1f} cm   "
                           f"RMS = {100 * np.sqrt(np.mean(E['n'][m] ** 2)):4.1f} cm")
        self.etext.set_text("\n".join(txt))
        self.etext.set_visible(bool(txt))

        # 3D + top view
        for (i, a3), (_, at_) in zip(self.d3, self.dt):
            now = {}
            for key, tr in (("real", self.real[i]), ("sim", self.sim[i])):
                P = self._window(tr, t)
                p = self._now(tr, t)
                now[key] = p
                a3[key + "_l"].set_data(P[:, 0], P[:, 1])
                a3[key + "_l"].set_3d_properties(P[:, 2])
                D = self.tv_scale * (P[:, :2] - self.origin[:2])
                at_[key + "_l"].set_data(D[:, 0], D[:, 1])
                if p is None:
                    a3[key + "_p"].set_data([], [])
                    a3[key + "_p"].set_3d_properties([])
                    at_[key + "_p"].set_data([], [])
                else:
                    a3[key + "_p"].set_data([p[0]], [p[1]])
                    a3[key + "_p"].set_3d_properties([p[2]])
                    d = self.tv_scale * (p[:2] - self.origin[:2])
                    at_[key + "_p"].set_data([d[0]], [d[1]])
            if now["real"] is not None and now["sim"] is not None:
                d = self.tv_scale * (np.array([now["real"][:2], now["sim"][:2]]) - self.origin[:2])
                at_["link"].set_data(d[:, 0], d[:, 1])
            else:
                at_["link"].set_data([], [])

        # camera and its lines of sight to the first drone
        if T_wc is not None:
            c, cor = _frustum(T_wc, self.K, self.vw, self.vh, 0.45)
            self.cam_pt.set_data([c[0]], [c[1]])
            self.cam_pt.set_3d_properties([c[2]])
            segs = [(c, cor[k]) for k in range(4)] + [(cor[k], cor[(k + 1) % 4]) for k in range(4)]
            for ln, (p, q) in zip(self.cam_fr, segs):
                ln.set_data([p[0], q[0]], [p[1], q[1]])
                ln.set_3d_properties([p[2], q[2]])
            if self.ids:
                i0 = self.ids[0]
                for ln, tr in zip(self.ray, (self.real[i0], self.sim[i0])):
                    p = self._now(tr, t)
                    if p is None:
                        ln.set_data([], [])
                        ln.set_3d_properties([])
                    else:
                        ln.set_data([c[0], p[0]], [c[1], p[1]])
                        ln.set_3d_properties([c[2], p[2]])
        else:
            self.cam_pt.set_data([], [])
            self.cam_pt.set_3d_properties([])
        phase = ""
        for n in (*self.names, "EOF"):
            if n in self.ph and t >= self.ph[n]:
                phase = n
        t0 = self.ph[self.names[0]] if self.names else 0.0
        self.title3.set_text(f"t = {t - t0:5.2f} s   {SHORT.get(phase, phase) if phase != 'EOF' else 'landed'}")

    def compose(self, overlay_bgr, t, T_wc):
        """The dashboard frame (BGR, H x W) at flight time t."""
        self.update(t, T_wc)
        out = np.full((self.H, self.W, 3), 255, np.uint8)
        out[:self.ph_px, :self.pw] = cv2.resize(overlay_bgr, (self.pw, self.ph_px), interpolation=cv2.INTER_CUBIC)
        b = _fig_to_bgr(self.fig_b)
        out[self.ph_px:, :self.pw] = cv2.resize(b, (self.pw, self.H - self.ph_px)) if b.shape[:2] != (
            self.H - self.ph_px, self.pw) else b
        r = _fig_to_bgr(self.fig_r)
        out[:, self.pw:] = cv2.resize(r, (self.right_w, self.H)) if r.shape[:2] != (self.H, self.right_w) else r
        return out

    def close(self):
        plt.close(self.fig_b)
        plt.close(self.fig_r)
