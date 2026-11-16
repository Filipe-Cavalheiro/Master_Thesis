"""Shared helpers for the thesis-defence slides.

Visual conventions (kept consistent with the APCA 2026 paper presentation):
    * black background, white text, section title "N. Name" in the upper-left corner
    * RED   -> target UAV (p_T)
    * WHITE -> chaser UAVs (p_i)
    * BLUE  -> desired points (p*)
    * YELLOW-> highlights / headings

All 3D drawings are done with a light-weight orthographic projection on a normal 2D
``Slide`` (no ThreeDScene). This keeps rendering fast and robust and lets us
control depth cues (back-faces dimmed) explicitly.
"""
from __future__ import annotations

import atexit
import os

import numpy as np
from manim import *
import re

# ----------------------------------------------------------------------------
# language: the presentation is in Portuguese (pt-PT)
# ----------------------------------------------------------------------------
# The scene code keeps its text in English, used only as lookup keys: every Text / MathTex / Tex
# string goes through tr(), which returns its translation from i18n_pt.py. A string without a
# translation is shown as is and listed in i18n_missing.txt at exit.
LANG = "pt"
_ManimText, _ManimMathTex, _ManimTex = Text, MathTex, Tex
# Pin the font: an empty font name means the system default (Noto Serif on the Linux machine the
# slides were designed on), which differs on other systems, e.g. Windows. Noto Serif has no →≈≤≥:
# Linux takes them from DejaVu Serif by fallback, Windows does not fall back, so name it per glyph.
_ManimText.set_default(font="Noto Serif", t2f={c: "DejaVu Serif" for c in "→≈≤≥"})
_MISSING: set[str] = set()
if LANG == "pt":
    from i18n_pt import PT, PT_PATTERNS

    @atexit.register
    def _dump_missing():
        if _MISSING:
            with open("i18n_missing.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(repr(s) for s in sorted(_MISSING)) + "\n")


def tr(s):
    """Translate a slide string (identity in English)."""
    if LANG != "pt" or not isinstance(s, str):
        return s
    if s in PT:
        return PT[s]
    for pat, rep in PT_PATTERNS:
        if pat.fullmatch(s):
            return pat.sub(rep, s)
    if any(c.isalpha() for c in re.sub(r"\$[^$]*\$", "", s)):   # LaTeX-only strings need no translation
        _MISSING.add(s)
    return s


def Text(text, *args, **kwargs):  # noqa: F811
    """
    Slide text. Segments between $...$ are rendered as LaTeX (MathTex) inline
    with the rest, e.g. Text("radial error rate $\\ge k_r/e$"); a string with
    newlines gives one line per row, left aligned.
    """
    s = tr(text)
    if isinstance(s, str) and "$" in s:
        return mixed_text(s, **kwargs)
    if LANG == "pt" and kwargs.get("t2c"):
        kwargs["t2c"] = {tr(k): v for k, v in kwargs["t2c"].items()}
    return _ManimText(s, *args, **kwargs)


MATH_SCALE = 1.3   # MathTex next to Text of the same font_size looks smaller


def mixed_text(s, font_size=48, color=WHITE, line_spacing=1.0, **_ignored):
    """Text with inline $...$ LaTeX segments (see Text); returns a VGroup of lines."""
    lines = VGroup()
    unit = font_size / 24.0
    for line in s.split("\n"):
        parts = [p for p in re.split(r"(\$[^$]+\$)", line) if p != ""]
        row = VGroup()
        prev_space = False
        for k, p in enumerate(parts):
            if p.startswith("$") and p.endswith("$") and len(p) > 1:
                m = _ManimMathTex(p[1:-1], font_size=font_size * MATH_SCALE, color=color)
                lead, trail = False, False
            else:
                lead, trail = p[0] == " ", p[-1] == " "
                if not p.strip():
                    prev_space = True
                    continue
                m = _ManimText(p.strip(), font_size=font_size, color=color)
            if len(row):
                gap = 0.11 * unit if (prev_space or lead) else 0.035 * unit
                m.next_to(row[-1], RIGHT, buff=gap)
                # vertical: centre on the text line (the first text part), math slightly raised by its own centre
                ref = next((r for r in row if isinstance(r, _ManimText)), row[0])
                m.set_y(ref.get_y())
            row.add(m)
            prev_space = trail
        if len(row) == 0:
            row.add(_ManimText(" ", font_size=font_size, color=color))
        lines.add(row)
    lines.arrange(DOWN, aligned_edge=LEFT, buff=0.14 * unit * line_spacing)
    return lines


def MathTex(*strings, **kwargs):  # noqa: F811
    if LANG == "pt":
        strings = tuple(PT.get(s, s) if isinstance(s, str) else s for s in strings)
        for s in strings:
            if isinstance(s, str) and "\\text" in s and s not in PT.values():
                _MISSING.add(s)
    return _ManimMathTex(*strings, **kwargs)


def Tex(*strings, **kwargs):  # noqa: F811
    if LANG == "pt":
        for s in strings:
            if isinstance(s, str) and s not in PT:
                _MISSING.add(s)
        strings = tuple(PT.get(s, s) if isinstance(s, str) else s for s in strings)
    return _ManimTex(*strings, **kwargs)


ASSETS = "assets/images/"
DATA = "assets/data/"
VIDEOS = "assets/videos/"

TARGET_COLOR = RED
AGENT_COLOR = WHITE
DESIRED_COLOR = BLUE
HIGHLIGHT = YELLOW
MUTED = GREY_B
GREEN_OK = GREEN_C

SECTIONS = [
    "Motivation",
    "Problem",
    "Background",
    "Centralized",
    "Decentralized",
    "Generalized",
    "Results",
    "Conclusions",
]

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


# ----------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------
def section_title(text: str) -> Text:
    """Upper-left section title, same as in the paper presentation."""
    t = Text(text, font_size=28)
    if t.width > 13.2:                      # long (translated) titles stay inside the frame
        t.set(width=13.2)
    return t.to_corner(UL)


def progress_bar(active: str | None) -> VGroup:
    """Small navigation bar at the bottom of the frame."""
    labels = VGroup()
    for name in SECTIONS:
        col = HIGHLIGHT if name == active else GREY_D
        labels.add(Text(name, font_size=16, color=col))
    labels.arrange(RIGHT, buff=0.45)
    labels.to_edge(DOWN, buff=0.15)
    return labels


def bullet_list(items, font_size=26, buff=0.22, color=WHITE, bullet="•", t2c=None):
    rows = VGroup()
    for it in items:
        row = VGroup(
            Text(bullet, font_size=font_size, color=HIGHLIGHT),
            Text(it, font_size=font_size, color=color, t2c=t2c or {}),
        ).arrange(RIGHT, buff=0.18, aligned_edge=UP)
        rows.add(row)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return rows


def box(label, width=3.0, height=1.15, font_size=24, color=WHITE, fill=None):
    rect = RoundedRectangle(corner_radius=0.12, width=width, height=height, color=color)
    if fill is not None:
        rect.set_fill(fill, opacity=0.25)
    txt = Text(label, font_size=font_size)
    txt.move_to(rect)
    return VGroup(rect, txt)


def plot_panel(path: str, height: float = 3.0, width: float | None = None) -> Group:
    """Thesis plot (white background PNG) framed in a thin rounded border."""
    img = ImageMobject(path)
    if width is not None:
        img.set(width=width)
    else:
        img.set(height=height)
    bg = RoundedRectangle(
        corner_radius=0.08,
        width=img.width + 0.12,
        height=img.height + 0.12,
        stroke_color=GREY_C,
        stroke_width=1.5,
        fill_color=WHITE,
        fill_opacity=1,
    )
    bg.move_to(img)
    return Group(bg, img)


def captioned(panel: Mobject, caption: str, font_size=20) -> Group:
    cap = Text(caption, font_size=font_size, color=GREY_A)
    cap.next_to(panel, DOWN, buff=0.12)
    return Group(panel, cap)


def legend_entry(color, label, font_size=24, radius=0.08):
    return VGroup(Dot(radius=radius, color=color), Text(label, font_size=font_size, color=color)).arrange(
        RIGHT, buff=0.15
    )


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def fibonacci_points(n: int, R: float = 1.0) -> np.ndarray:
    """n-point Fibonacci lattice on a sphere of radius R (z is the polar axis)."""
    i = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    th = i * GOLDEN_ANGLE
    return R * np.column_stack((r * np.cos(th), r * np.sin(th), z))


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Proj:
    """Orthographic projection matching Manim's (phi, theta) camera convention."""

    def __init__(self, phi=70 * DEGREES, theta=-60 * DEGREES, scale=1.0, center=ORIGIN, world_center=None):
        self.phi = phi
        self.theta = theta
        self.scale = scale
        self.center = np.array(center, dtype=float)
        self.world_center = np.zeros(3) if world_center is None else np.array(world_center, dtype=float)

    def basis(self):
        ph, th = self.phi, self.theta
        view = np.array([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)])
        right = np.array([-np.sin(th), np.cos(th), 0.0])
        up = np.cross(view, right)
        return right, up, view

    def __call__(self, p):
        p = np.asarray(p, dtype=float) - self.world_center
        right, up, _ = self.basis()
        if p.ndim == 1:
            return self.center + self.scale * np.array([p @ right, p @ up, 0.0])
        out = np.zeros((p.shape[0], 3))
        out[:, 0] = p @ right
        out[:, 1] = p @ up
        return self.center + self.scale * out

    def depth(self, p):
        p = np.asarray(p, dtype=float) - self.world_center
        return p @ self.basis()[2]


def curve3d(points, proj: Proj, color=GREY_B, width=2.0, front_op=0.8, back_op=0.15, sphere_center=None):
    """Polyline in 3D. Segments facing away from the camera are dimmed.

    If ``sphere_center`` is given, 'back' means behind the sphere's silhouette plane.
    """
    pts = np.asarray(points, dtype=float)
    c = proj.world_center if sphere_center is None else np.asarray(sphere_center, dtype=float)
    d = (pts - c) @ proj.basis()[2]
    front = d >= 0
    group = VGroup()
    start = 0
    for k in range(1, len(pts) + 1):
        if k == len(pts) or front[k] != front[start]:
            seg = pts[max(start - 1, 0):k]
            if len(seg) >= 2:
                vm = VMobject(stroke_color=color, stroke_width=width)
                vm.set_points_as_corners(proj(seg))
                vm.set_stroke(opacity=front_op if front[start] else back_op)
                group.add(vm)
            start = k
    return group


def sphere_wire(proj: Proj, R=1.0, center=(0, 0, 0), n_lat=5, n_lon=6, color=GREY_B, width=1.5, front_op=0.45, back_op=0.08):
    center = np.asarray(center, dtype=float)
    g = VGroup()
    t = np.linspace(0, 2 * np.pi, 121)
    for lat in np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]:
        pts = np.column_stack([R * np.cos(lat) * np.cos(t), R * np.cos(lat) * np.sin(t), R * np.sin(lat) * np.ones_like(t)])
        g.add(curve3d(pts + center, proj, color, width, front_op, back_op, sphere_center=center))
    for lon in np.linspace(0, np.pi, n_lon, endpoint=False):
        pts = np.column_stack([R * np.cos(t) * np.cos(lon), R * np.cos(t) * np.sin(lon), R * np.sin(t)])
        g.add(curve3d(pts + center, proj, color, width, front_op, back_op, sphere_center=center))
    # silhouette
    sil = Circle(radius=R * proj.scale, color=color, stroke_width=width).move_to(proj(center))
    sil.set_stroke(opacity=front_op)
    g.add(sil)
    return g


def dots3d(points, proj: Proj, color=WHITE, radius=0.06, back_op=0.3, sphere_center=None, colors=None,
           stable=False, z_base=1.0):
    """
    Projected dots, back ones dimmed. Default: added back-to-front. stable=True: dot k is point k
    (so Transform between two calls moves every point to its own new position, as when animating
    the same agents), and the back-to-front drawing order comes from the z_index (z_base + rank).
    """
    pts = np.asarray(points, dtype=float)
    c = proj.world_center if sphere_center is None else np.asarray(sphere_center, dtype=float)
    d = (pts - c) @ proj.basis()[2]
    order = np.argsort(d)
    if stable:
        rank = np.empty(len(d), int)
        rank[order] = np.arange(len(d))
        g = VGroup()
        for k in range(len(pts)):
            dot = Dot(proj(pts[k]), radius=radius, color=colors[k] if colors is not None else color)
            if d[k] < 0:
                dot.set_opacity(back_op)
            dot.set_z_index(z_base + 1e-3 * rank[k])
            g.add(dot)
        return g
    g = VGroup()
    for k in order:
        col = colors[k] if colors is not None else color
        dot = Dot(proj(pts[k]), radius=radius, color=col)
        if d[k] < 0:
            dot.set_opacity(back_op)
        g.add(dot)
    return g


def axes3d(proj: Proj, length=1.0, origin=(0, 0, 0), labels=("x", "y", "z"), color=GREY_B):
    o = np.asarray(origin, dtype=float)
    g = VGroup()
    for k, lab in enumerate(labels):
        e = np.zeros(3)
        e[k] = length
        arr = Arrow(proj(o), proj(o + e), buff=0, stroke_width=3, color=color, max_tip_length_to_length_ratio=0.15)
        txt = MathTex(lab, font_size=28, color=color).move_to(proj(o + 1.15 * e))
        g.add(arr, txt)
    return g


# ----------------------------------------------------------------------------
# Voronoi / Lloyd helpers
# ----------------------------------------------------------------------------
def clip_halfplane(poly, a, b):
    """Keep the part of convex polygon `poly` satisfying a . x <= b."""
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        pin, qin = a @ P <= b, a @ Q <= b
        if pin:
            out.append(P)
        if pin != qin:
            t = (b - a @ P) / (a @ (Q - P))
            out.append(P + t * (Q - P))
    return out


def voronoi_cells_2d(sites, xmin, xmax, ymin, ymax):
    """Bounded Voronoi cells by successive half-plane clipping (exact)."""
    boxp = [np.array(p, float) for p in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]]
    cells = []
    for i, si in enumerate(sites):
        poly = list(boxp)
        for j, sj in enumerate(sites):
            if i == j:
                continue
            a = sj - si
            b = 0.5 * (sj @ sj - si @ si)
            poly = clip_halfplane(poly, a, b)
            if not poly:
                break
        cells.append(np.array(poly))
    return cells


def polygon_centroid(poly):
    x, y = poly[:, 0], poly[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cr = x * ys - xs * y
    A = cr.sum() / 2
    if abs(A) < 1e-12:
        return poly.mean(axis=0)
    return np.array([((x + xs) * cr).sum() / (6 * A), ((y + ys) * cr).sum() / (6 * A)])


def sphere_lloyd_step(sites, samples):
    """One Lloyd iteration on the unit sphere using a dense sample set."""
    owner = np.argmax(samples @ sites.T, axis=1)
    new = sites.copy()
    for i in range(len(sites)):
        m = samples[owner == i]
        if len(m):
            c = m.mean(axis=0)
            new[i] = c / np.linalg.norm(c)
    return new, owner


def nn_cv(sites):
    """Coefficient of variation of the nearest-neighbour geodesic distance."""
    G = np.clip(sites @ sites.T, -1, 1)
    D = np.arccos(G)
    np.fill_diagonal(D, np.inf)
    dnn = D.min(axis=1)
    return float(dnn.std() / dnn.mean())


# ----------------------------------------------------------------------------
# Velocity-weighted CVT on a sphere mesh (thesis Ch. 6, Python/demos_and_helpers/
# CVT_weighted_verticies/voronoi_mesh_funcs.py: Dijkstra cells, edge length scaled by the
# mean vertex weight, site -> mesh vertex closest to the cell centroid)
# ----------------------------------------------------------------------------
class SphereMesh:
    """Unit-sphere triangle mesh (Fibonacci vertices + convex hull) for the mesh CVT."""

    def __init__(self, n=1000):
        from scipy.spatial import ConvexHull

        self.V = fibonacci_points(n)
        F = ConvexHull(self.V).simplices
        E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        self.E = np.unique(np.sort(E, axis=1), axis=0)
        self.L = np.linalg.norm(self.V[self.E[:, 0]] - self.V[self.E[:, 1]], axis=1)

    def nearest(self, pts):
        """Indices of the mesh vertices closest to the unit vectors ``pts``."""
        return np.argmax(np.asarray(pts) @ self.V.T, axis=1)

    def velocity_weights(self, v, sigma=1.2):
        """w = exp(theta^2 / (2 sigma^2)), theta = angle from the direction behind the target (-v)."""
        if v is None:
            return np.ones(len(self.V))
        back = -np.asarray(v, float) / np.linalg.norm(v)
        th = np.arccos(np.clip(self.V @ back, -1, 1))
        return np.exp(th ** 2 / (2 * sigma ** 2))

    def lloyd_step(self, sites, w):
        """One synchronous Lloyd step; returns (new site indices, owner site of each vertex)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        c = self.L * (w[self.E[:, 0]] + w[self.E[:, 1]]) / 2
        i, j = self.E[:, 0], self.E[:, 1]
        n = len(self.V)
        G = coo_matrix((np.r_[c, c], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
        _, _, src = dijkstra(G, indices=sites, min_only=True, return_predecessors=True)
        occupied = set(int(s) for s in sites)
        new = []
        for s in sites:
            m = self.V[src == s].mean(axis=0)
            occupied.discard(int(s))
            k = next(int(k) for k in np.argsort(-(self.V @ m)) if int(k) not in occupied)
            occupied.add(k)
            new.append(k)
        return np.array(new), src

    def owner(self, sites, w):
        return self.lloyd_step(sites, w)[1]


# ----------------------------------------------------------------------------
# Mesh controller (re-implementation of Ch. 6, Python/V16/voronoi_mesh_funcs.py)
# ----------------------------------------------------------------------------
def precompute_meshes(P=400, R=5.0, legacy_cone=False):
    """
    Disc, cone and sphere of the mesh controller (thesis Ch. 6). The cone is
    lifted by R (1 - (r / r_max)^2), so its rim lies on the wall. legacy_cone:
    the V16 cone, R (1 - (r / R)^2), whose outer ring curls back past the wall
    (the simulation of the Results replay was run with it).
    """
    N = max(int(np.floor((P - 1) / 2)), 1)
    k = np.arange(P)
    r = R * np.sqrt((k + 1) / N)
    th = (k + 1) * GOLDEN_ANGLE
    v_circle = np.column_stack([np.zeros(P), r * np.cos(th), r * np.sin(th)])
    radial = np.sqrt(v_circle[:, 1] ** 2 + v_circle[:, 2] ** 2)
    r_max = R if legacy_cone else radial.max()
    v_cone = np.column_stack([R * (1 - (radial / r_max) ** 2), v_circle[:, 1], v_circle[:, 2]])
    phi_g = (1 + np.sqrt(5)) / 2
    i = np.arange(P)
    phi = 2 * np.pi * i / phi_g
    theta = np.arccos(1 - 2 * (i + 0.5) / P)
    v_sphere = np.column_stack([R * np.cos(theta), R * np.sin(theta) * np.sin(phi), R * np.sin(theta) * np.cos(phi)])
    return v_circle, v_cone, v_sphere


def mesh_state(target, meshes, R=5.0):
    """Return (vertices, u1, u2) for a target position (boundary plane x = 0, restricted side x > 0)."""
    v_circle, v_cone, v_sphere = meshes
    d = float(target[0])
    if d < -R / 2:
        u = 0.0
    elif d < R / 2:
        u = 0.5 + d / R
    else:
        u = d / (R / 2)
    if u <= 1:
        u1, u2 = u, 0.0
        pos = np.array([0.0, target[1], target[2]])
    else:
        u1, u2 = 1.0, min(u - 1, 1.0)
        pos = np.array([target[0] - R / 2 + u2 * R / 2, target[1], target[2]])
    v = (1 - u1) * v_circle + u1 * v_cone
    if u2 > 0:
        v = (1 - u2) * v + u2 * v_sphere
    return v + pos, u1, u2


# ----------------------------------------------------------------------------
# Bernstein / Bezier
# ----------------------------------------------------------------------------
def bezier_eval(ctrl, s):
    from math import comb

    ctrl = np.asarray(ctrl, float)
    n = len(ctrl) - 1
    s = np.atleast_1d(s)
    B = np.array([[comb(n, k) * (si ** k) * ((1 - si) ** (n - k)) for k in range(n + 1)] for si in s])
    return B @ ctrl


# ----------------------------------------------------------------------------
# video clips (real flights)
# ----------------------------------------------------------------------------
class _Decoder:
    """PyAV decoder; shared (not copied) when manim deep-copies the mobject."""

    def __init__(self, path):
        import av

        self.container = av.open(path)
        stream = self.container.streams.video[0]
        self.fps = float(stream.average_rate)
        self.duration = float(stream.frames / stream.average_rate)
        self.frames = self.container.decode(stream)

    def __deepcopy__(self, memo):
        return self


class VideoMobject(ImageMobject):
    """A video file shown as an image whose frame follows the scene time.

    Frames are decoded sequentially with PyAV (a manim dependency), so only one frame is held
    in memory. Call ``start()`` and then ``self.wait(video.duration)`` to play it; the last
    frame stays on screen afterwards.
    """

    def __init__(self, path: str, **kwargs):
        self._dec = _Decoder(path)
        self.fps = self._dec.fps
        self.duration = self._dec.duration
        self._index = 0
        self._time = 0.0
        super().__init__(self._rgba(next(self._dec.frames)), **kwargs)

    @staticmethod
    def _rgba(frame):
        return frame.to_ndarray(format="rgba")

    def _advance(self, mob, dt):
        self._time += dt
        target = int(self._time * self.fps)
        frame = None
        while self._index < target:
            try:
                frame = next(self._dec.frames)
            except StopIteration:
                break
            self._index += 1
        if frame is not None:
            self.pixel_array = self._rgba(frame)

    def start(self):
        self.add_updater(self._advance)
        return self

    def stop(self):
        self.clear_updaters()
        self._dec.container.close()
        return self
