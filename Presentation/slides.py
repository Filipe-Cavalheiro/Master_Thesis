"""MSc thesis defence - Cooperative Encirclement of Non-Cooperative Drones.

Filipe Cavalheiro, NOVA FCT, MSc in Electrical and Computer Engineering.

Render (low quality, quick check):
    manim-slides render -ql slides.py $(cat scenes.txt)
Render (final):
    manim-slides render -qh slides.py $(cat scenes.txt)
Present:
    manim-slides present $(cat scenes.txt)
See README.md for details.
"""
from __future__ import annotations

import numpy as np
from manim import *
from manim_slides import Slide

from helpers import *

rng_global = np.random.default_rng(7)


def _load_notes(path="SPEAKER_NOTES.md"):
    """Parse '#### [Scene.k] title' blocks of SPEAKER_NOTES.md into {scene: {k: text}}."""
    import os
    import re

    notes = {}
    if not os.path.exists(path):
        return notes
    txt = open(path, encoding="utf-8").read()
    pat = re.compile(r"^#### \[(\w+)\.(\d+)\][^\n]*\n(.*?)(?=^#### |^## |^---|\Z)", re.S | re.M)
    for m in pat.finditer(txt):
        notes.setdefault(m.group(1), {})[int(m.group(2))] = " ".join(m.group(3).split())
    return notes


NOTES = _load_notes()


class Base(Slide):
    """Common helpers for every scene."""

    SECTION: str | None = None

    # speaker notes: sub-slide k of scene X gets block [X.k] of SPEAKER_NOTES.md
    def setup(self):
        super().setup()
        self._note_idx = 1
        super().next_slide(notes=self._note(1))

    def _note(self, k):
        return NOTES.get(type(self).__name__, {}).get(k, "")

    def next_slide(self, **kwargs):
        # Manim writes the last frame of an animation just before it ends (alpha ~0.97), and the
        # presenter holds that frame until the key press: with lagged animations (Create of a group
        # draws its lines one after another) the last element would stay half drawn. Render the
        # settled final state first.
        self.wait(0.1)
        self._note_idx += 1
        kwargs.setdefault("notes", self._note(self._note_idx))
        super().next_slide(**kwargs)

    def header(self, title: str, animate=True):
        t = section_title(title)
        bar = progress_bar(self.SECTION)
        if animate:
            self.play(FadeIn(t), FadeIn(bar), run_time=0.6)
        else:
            self.add(t, bar)
        self._title = t
        self._bar = bar
        return t

    def keep(self):
        return [m for m in (getattr(self, "_title", None), getattr(self, "_bar", None)) if m is not None]

    def clear_body(self, run_time=0.6):
        keep = self.keep()
        mobs = [m for m in self.mobjects if m not in keep]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def clear_all(self, run_time=0.6):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=run_time)

    def new_title(self, text):
        t = section_title(text)
        self.play(Transform(self._title, t), run_time=0.5)

    def architecture(self, title, kind):
        """Sub-slide with the complete control architecture of a chapter (see arch_diagram)."""
        self.new_title(title)
        d = arch_diagram(kind)
        groups, body = d
        self.play(FadeIn(body), run_time=0.8)
        self.play(FadeIn(groups), run_time=0.6)

    def flight(self, clip, facts, speed, extra=()):
        """Play a real-flight clip (camera + simulation overlay) with a legend and facts below."""
        video = VideoMobject(VIDEOS + clip).set(width=10.6).move_to(UP * 0.5)
        frame = SurroundingRectangle(video, color=GREY_D, stroke_width=1.5, buff=0)
        leg = VGroup(
            legend_entry(GREEN, "real (OptiTrack)", font_size=20, radius=0.07),
            legend_entry(ORANGE, "simulation of the same flight", font_size=20, radius=0.07),
            *[legend_entry(c, lab, font_size=20, radius=0.07) for c, lab in extra],
            Text(speed if isinstance(speed, str) else ("real time" if speed == 1 else f"{speed}x speed"), font_size=20, color=GREY_B),
        ).arrange(RIGHT, buff=0.45)
        if leg.width > 12.8:
            leg.set(width=12.8)
        leg.next_to(video, DOWN, buff=0.15)
        info = VGroup(*[Text(f, font_size=21) for f in facts]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        if info.width > 12.8:
            info.set(width=12.8)
        info.next_to(leg, DOWN, buff=0.2)
        self.play(FadeIn(video), FadeIn(frame), FadeIn(leg), FadeIn(info), run_time=0.6)
        video.start()
        self.wait(video.duration)
        video.stop()


# =============================================================================
# Control architectures (one per chapter, shown before its simulation results)
# =============================================================================
def arch_diagram(kind: str) -> VGroup:
    """Complete control architecture of one chapter: 'centralized', 'decentralized' or 'generalized'.

    Merges the thesis figures into one diagram per chapter (Ch. 4: path planning + Mellinger +
    target estimation sub-diagrams; Ch. 5 and 6: complete architectures), with the Mellinger
    controller drawn as its position and attitude loops. Blocks introduced in that chapter are yellow.
    """
    TOP, XL, XR = 3.75, -7.4, 6.6
    yC, yE, yS, yL = -1.15, -2.55, -1.75, -3.85   # consensus/network row, EMA/measurement row, source block, L route
    xG, xA = -4.7, -1.8                      # path-planning columns; xA is also the consensus/EMA column
    B, W, J = arch_block, arch_wire, junction
    g = VGroup()

    # low level (every agent): Mellinger controller + quadrotor, state feedback
    pos = B("Position controller", (2.0, 2.4))
    att = B("Attitude controller", (2.0, 0.9))
    ruav = B("Quadrotor model", (5.0, 1.65), w=2.0)
    g.add(pos, att, ruav)
    g.add(
        W(port(pos, DOWN), port(att, UP), label=r"\mathbf{F}^\star", side=RIGHT),
        W(port(pos, RIGHT, -0.15), (3.65, 2.25), (3.65, 1.85), port(ruav, LEFT, 0.2), label="f", seg=2),
        W(port(att, RIGHT), (3.5, 0.9), (3.5, 1.45), port(ruav, LEFT, -0.2), label=r"\boldsymbol{\tau}", seg=2, side=DOWN),
        W(port(ruav, RIGHT), (XR, 1.65), label=r"\mathbf{x}", tip=False),
        W((XR, yC), (XR, TOP), (XL, TOP), tip=False),                                       # state bus
        W((XR, 2.65), port(pos, RIGHT, 0.25)),
        W((XR, 0.25), (2.0, 0.25), port(att, DOWN), label=r"R,\boldsymbol{\omega}", seg=0, side=DOWN, at=0.4),
        J((XR, 1.65)), J((XR, 2.65)), J((XR, 0.25)),
    )

    # target estimation (every agent): noisy measurement -> EMA -> consensus over the Delaunay graph
    cons = B("Consensus", (xA, yC))
    ema = B("Exponential moving\naverage (EMA)", (xA, yE))
    net = B("Spatial network\n(Delaunay)", (2.0, yC))
    noisy = B("Noisy measurement", (2.0, yE))
    g.add(cons, ema, net, noisy)
    g.add(
        W((XR, yC), port(net, RIGHT), label=r"\mathbf{p}_i"),
        W(port(net, LEFT), port(cons, RIGHT), label="L"),
        W((4.5, yE), port(noisy, RIGHT), label=r"\mathbf{p}_T", color=TARGET_COLOR),
        W(port(noisy, LEFT), port(ema, RIGHT), label=r"\hat{\mathbf{p}}_T", at=0.75),
        W(port(ema, UP), port(cons, DOWN), label=r"\hat{\mathbf{p}}_T", side=RIGHT),
    )
    groups = VGroup(
        arch_group([pos, att], "Mellinger controller", BLUE_C),
        arch_group([cons, ema, net, noisy], "Target estimation", ORANGE, label_at=DL),
    )

    def p_into(block, y, label=r"\mathbf{p}_i"):    # state from the left trunk into a block
        return VGroup(W((XL, y), (block[0].get_left()[0], y), label=label, at=0.45), J((XL, y)))

    def neighbours_into(*route):              # L from the spatial network, under the estimation group
        return VGroup(W((0.45, yC), (0.45, yL), *route, label="L", seg=1, side=DOWN, at=0.1), J((0.45, yC)))

    if kind in ("centralized", "decentralized"):
        geom = B("Geometric controller", (xG, 2.4))
        apf = B("Safeguard (APF)", (xA, 2.4))
        radii = B("Radii control", (xG, 0.9))
        src = (B("Fibonacci lattice\n+ Munkres", (-5.2, yS)) if kind == "centralized"
               else B("Lloyd's algorithm\n(Voronoi on the sphere)", (-5.2, yS), new=True))
        g.add(geom, apf, radii, src)
        y_est = 0.0
        g.add(
            W(port(radii, UP), port(geom, DOWN), label=r"\mathbf{q}_i^\star", side=RIGHT),
            W(port(geom, RIGHT), port(apf, LEFT), label=r"\bar{\mathbf{q}}_i"),
            W(port(apf, RIGHT), port(pos, LEFT), label=r"\tilde{\mathbf{q}}_i"),
            W((src[0].get_top()[0], src[0].get_top()[1]), (src[0].get_top()[0], radii[0].get_bottom()[1]),
              label=r"\mathbf{p}_i^\star", side=RIGHT),
            W(port(cons, UP), (xA, y_est), (-4.0, y_est), (-4.0, radii[0].get_bottom()[1]),
              label=r"\hat{\mathbf{p}}_T", seg=1, side=DOWN),
            W((-3.55, y_est), (-3.55, yE), port(ema, LEFT)), J((-3.55, y_est)),     # EMA uses the estimate
            W((XL, TOP), (XL, yS), tip=False),                                     # left trunk
            p_into(geom, 2.4), p_into(radii, 0.9),
        )
        if kind == "centralized":
            g.add(W((XL, yS), port(src, LEFT), label=r"\mathbf{p}_{1..N}", at=0.4))
            groups.add(arch_group([src], "Central node", GREY_B, label_at=DL))
        else:
            g.add(p_into(src, yS), neighbours_into((-5.2, yL), port(src, DOWN)))
        groups.add(arch_group([geom, apf, radii], "Path planning", RED_C))
    else:
        mpc = B("MPC (Bernstein)", (xG, 2.4), new=True)
        lloyd = B("Lloyd's algorithm\n(geodesic CVT\non the mesh)", (xG, 0.85), h=1.05, new=True)
        mesh = B("Mesh controller", (-1.2, 0.85), new=True)
        g.add(mpc, lloyd, mesh)
        g.add(
            W(port(mpc, RIGHT), port(pos, LEFT), label=r"\mathbf{p}_d"),
            W(port(lloyd, UP), port(mpc, DOWN), label=r"\mathbf{p}_i^\star", side=RIGHT),
            W(port(mesh, LEFT), port(lloyd, RIGHT), label=r"V,F"),
            W(port(cons, UP), (xA, mesh[0].get_bottom()[1]), label=r"\hat{\mathbf{p}}_T,\hat{\mathbf{v}}_T", side=RIGHT, at=0.6),
            W((xA, 0.0), (-3.55, 0.0), (-3.55, yE), port(ema, LEFT)), J((xA, 0.0)),
            p_into(mpc, 2.4, r"\mathbf{p}_i,\mathbf{v}_i"),
            W((XL, TOP), (XL, 1.05), port(lloyd, LEFT, 0.2), label=r"\mathbf{p}_i", seg=1, at=0.3),   # left trunk
            neighbours_into((-6.6, yL), (-6.6, 0.65), port(lloyd, LEFT, -0.2)),
        )
        groups.add(arch_group([mpc, lloyd, mesh], "Path planning", RED_C))
    if kind != "centralized":
        leg = VGroup(RoundedRectangle(corner_radius=0.08, width=0.45, height=0.28, stroke_color=HIGHLIGHT,
                                      stroke_width=3, fill_color=ARCH_FILL, fill_opacity=1),
                     Text("new in this chapter", font_size=20, color=HIGHLIGHT)).arrange(RIGHT, buff=0.15)
        leg.move_to(np.array([5.0, -3.55, 0]))
        groups.add(leg)
    d = VGroup(groups, g)
    d.scale_to_fit_width(13.3)
    if d.height > 6.35:
        d.scale_to_fit_height(6.35)
    return d.move_to(DOWN * 0.2)


# =============================================================================
# 0. Title
# =============================================================================
class TitleCard(Base):
    def construct(self):
        fct_logo = ImageMobject(ASSETS + "logo_nova_fct_pt_v.png").set(width=2)
        fct_logo.to_corner(UL, buff=0.5)

        title = VGroup(
            Text("Cooperative Encirclement of", font_size=52),
            Text("Non-Cooperative Drones", font_size=52),
        ).arrange(DOWN, buff=0.2)
        subtitle = VGroup(
            Text("Decentralized, Collision-Aware Formation Control", font_size=26, color=GREY_A),
            Text("on Spherical and Generalized Surfaces", font_size=26, color=GREY_A),
        ).arrange(DOWN, buff=0.12)
        author = Text("Filipe Cavalheiro", font_size=32, color=YELLOW)
        adviser = Text("Adviser: Prof. Bruno J. Guerreiro", font_size=24)
        degree = Text(
            "MSc in Electrical and Computer Engineering  ·  NOVA School of Science and Technology",
            font_size=20,
            color=GREY_B,
        )
        block = VGroup(title, subtitle, author, adviser, degree).arrange(DOWN, buff=0.3)
        block.move_to(DOWN * 0.55)

        # small decorative encirclement animation in the upper-right corner
        proj = Proj(phi=72 * DEGREES, theta=-40 * DEGREES, scale=0.75, center=np.array([5.0, 2.6, 0]))
        pts = fibonacci_points(13, 1.0)
        target = Dot(proj(np.zeros(3)), radius=0.07, color=TARGET_COLOR)
        start = pts * 2.6 + rng_global.normal(0, 0.4, pts.shape)
        tracker = ValueTracker(0.0)

        def chasers():
            s = tracker.get_value()
            p = (1 - s) * start + s * pts
            proj.theta = -40 * DEGREES + 40 * DEGREES * s
            return dots3d(p, proj, color=WHITE, radius=0.05, back_op=0.35)

        swarm = always_redraw(chasers)
        wire = always_redraw(lambda: sphere_wire(proj, 1.0, n_lat=3, n_lon=4, front_op=0.25 * tracker.get_value(), back_op=0.05))

        self.add(fct_logo)
        self.play(FadeIn(block, shift=UP * 0.2), FadeIn(target), FadeIn(swarm), run_time=1.2)
        self.add(wire)
        self.play(tracker.animate.set_value(1.0), run_time=3, rate_func=smooth)
        self.next_slide(auto_next=True)
        swarm.clear_updaters()
        wire.clear_updaters()
        self.clear_all()


# =============================================================================
# 1. Motivation
# =============================================================================
class Motivation(Base):
    SECTION = "Motivation"

    def construct(self):
        self.header("1. Motivation")

        def headline(logo, en, pt):
            txt = VGroup(Text(en, font_size=28), Text(pt, font_size=16, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            img = ImageMobject(ASSETS + logo).scale(1.2)
            return Group(img, txt).arrange(RIGHT, buff=0.6).scale(0.8)

        h2 = headline(
            "logo_RTP.png",
            "Brussels Airport closes due to the presence of drones in its airspace",
            'Original: "Aeroporto de Bruxelas encerra espaço aéreo devido à presença de drones"',
        )
        h1 = headline(
            "logo_Publico.png",
            "Airports in Belgium and Sweden close due to drones",
            'Original: "Drones levam a encerramento de aeroportos na Bélgica e Suécia"',
        )
        news = Group(h2, h1).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to(UP * 0.4)
        self.play(FadeIn(h2, shift=RIGHT * 0.3))
        self.play(FadeIn(h1, shift=RIGHT * 0.3))
        self.next_slide()

        # --- DFS numbers (thesis, Ch.1) --------------------------------------
        self.play(news.animate.scale(0.75).to_edge(UP, buff=0.9))
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 250, 50],
            x_length=4.2,
            y_length=3.2,
            axis_config={"include_tip": False, "color": GREY_B},
            y_axis_config={"include_numbers": True, "font_size": 20},
        ).shift(LEFT * 3.5 + DOWN * 1.35)
        vals = [(1, 161, "2024"), (2, 225, "2025")]
        bars = VGroup()
        labels = VGroup()
        for x, v, yr in vals:
            bottom = axes.c2p(x, 0)
            top = axes.c2p(x, v)
            bar = Rectangle(width=1.0, height=top[1] - bottom[1], fill_color=RED_D, fill_opacity=0.85, stroke_width=0)
            bar.move_to((bottom + top) / 2)
            bars.add(bar)
            labels.add(Text(str(v), font_size=26, color=WHITE).next_to(bar, UP, buff=0.08))
            labels.add(Text(yr, font_size=20, color=GREY_A).next_to(bottom, DOWN, buff=0.12))
        chart_title = Text("Drone-related disruptions at German airports (DFS)", font_size=18, color=GREY_A)
        chart_title.next_to(axes, UP, buff=0.35)
        self.play(Create(axes), FadeIn(chart_title))
        self.play(*[GrowFromEdge(b, DOWN) for b in bars], FadeIn(labels), run_time=1.2)

        impact = bullet_list(
            [
                "Flights delayed, diverted or cancelled",
                "Hospital heliports: emergency transport at risk",
                "Cheap, accessible drones",
            ],
            font_size=22,
        )
        impact.next_to(axes, RIGHT, buff=0.6).align_to(axes, UP).shift(DOWN * 0.1)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in impact], lag_ratio=0.25))
        self.next_slide()

        self.play(FadeOut(Group(news, axes, bars, labels, chart_title, impact)))
        self.airspace()
        self.next_slide()

        question = Text(
            "How can a drone entering restricted airspace be surrounded\n"
            "safely, visibly, and without destructive means?",
            font_size=30,
            color=YELLOW,
            line_spacing=0.9,
        )
        frame = SurroundingRectangle(question, color=YELLOW, buff=0.3, corner_radius=0.15)
        self.clear_body()
        self.play(Write(question), Create(frame), run_time=1.5)
        self.next_slide(auto_next=True)
        self.clear_all()

    def airspace(self):
        """Lisbon airport (LPPT) zones over the city map, then why the usual countermeasures do not fit."""
        from itertools import permutations

        def fit(m, w):
            return m.set(width=w) if m.width > w else m

        d = np.load(DATA + "lisbon_zones.npz")
        x0, x1, y0, y1 = d["extent"]
        H = 6.3
        k = H / (y1 - y0)
        c = np.array([-4.4, -0.27, 0])

        def P(e, n):                                   # ENU [m] -> frame
            return c + np.array([(e - (x0 + x1) / 2) * k, (n - (y0 + y1) / 2) * k, 0])

        city = ImageMobject(ASSETS + "lisbon_map.png").set(height=H).move_to(c)
        border = SurroundingRectangle(city, color=GREY_D, stroke_width=1.5, buff=0)

        def zone(poly, col, op):
            return Polygon(*[P(*q) for q in poly], stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=op)

        area1 = VGroup(*[zone(d[f], ORANGE, 0.22) for f in sorted(d.files) if f.startswith("orange")])
        red = zone(d["red"], RED, 0.38)
        arp = Dot(P(0, 0), radius=0.05, color=WHITE)
        arp_lab = Text("airport", font_size=16).next_to(arp, RIGHT, buff=0.08)
        X = border.get_right()[0] + 0.45               # left edge of everything right of the map
        msg = VGroup(
            Text("Restricted airspace is not just the airport:", font_size=24, color=HIGHLIGHT),
            Text("the approach corridors cross the city", font_size=24),
            Text("hospitals, stadiums, a university, the main rail hub, dense housing", font_size=19, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        fit(msg, 6.9 - X).move_to(np.array([X, 2.4, 0]), aligned_edge=LEFT)

        # infrastructure under the zones: dots on the map, labels stacked right of the map; the
        # label order is the one whose leader lines do not cross (and are shortest)
        names = VGroup(*[Text(str(s), font_size=18) for s in d["poi_names"]])
        col = VGroup(*[n.copy() for n in names]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        col.move_to(np.array([X, 0.1, 0]), aligned_edge=LEFT)
        slots = [m.get_left() + LEFT * 0.08 for m in col]
        pts = [P(*q) for q in d["poi_xy"]]

        def side(a, b, u):                             # sign of the 2D cross product (b - a) x (u - a)
            return np.sign((b[0] - a[0]) * (u[1] - a[1]) - (b[1] - a[1]) * (u[0] - a[0]))

        def crossings(perm):
            segs = [(pts[i], slots[s]) for i, s in enumerate(perm)]
            n = 0
            for a in range(len(segs)):
                for b in range(a + 1, len(segs)):
                    (p, q), (r, s) = segs[a], segs[b]
                    o = [side(p, q, r), side(p, q, s), side(r, s, p), side(r, s, q)]
                    n += o[0] != o[1] and o[2] != o[3]
            return n

        best = min(permutations(range(len(pts))), key=lambda pm: (crossings(pm), sum(np.linalg.norm(pts[i] - slots[s]) for i, s in enumerate(pm))))
        for i, s in enumerate(best):
            names[i].move_to(col[s], aligned_edge=LEFT)
        dots = VGroup(*[Dot(p, radius=0.045, color=HIGHLIGHT) for p in pts])
        leaders = VGroup(*[Line(pts[i], slots[s], stroke_width=1, color=HIGHLIGHT).set_opacity(0.55) for i, s in enumerate(best)])
        key = VGroup(
            VGroup(Square(0.22, stroke_color=RED, fill_color=RED, fill_opacity=0.5), Text("Prohibited area", font_size=17)).arrange(RIGHT, buff=0.15),
            VGroup(Square(0.22, stroke_color=ORANGE, fill_color=ORANGE, fill_opacity=0.35), Text("Area 1 (height-limited)", font_size=17)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        key.next_to(col, DOWN, buff=0.45).align_to(col, LEFT)
        src = Text("Lisbon airport (LPPT) drone zones, ANAC · map © OpenStreetMap contributors", font_size=13, color=GREY_B)
        fit(src, 6.9 - X).move_to(np.array([X, border.get_bottom()[1] + 0.08, 0]), aligned_edge=LEFT)

        self.play(FadeIn(city), FadeIn(border), FadeIn(src), run_time=0.8)
        self.play(FadeIn(area1), FadeIn(red), FadeIn(arp), FadeIn(arp_lab), FadeIn(key), run_time=1.0)
        self.play(FadeIn(msg, shift=LEFT * 0.2), run_time=0.6)
        self.play(LaggedStart(*[AnimationGroup(GrowFromCenter(dots[i]), Create(leaders[i]), FadeIn(names[i], shift=LEFT * 0.1))
                                for i in np.argsort([-p[1] for p in pts])], lag_ratio=0.3), run_time=2.0)
        self.next_slide()

        # --- countermeasures used for restricted airspace (thesis Ch. 1, Lykou et al. 2020) --
        RX, W = 1.75, 6.9 - 1.75

        def counter(head, effect):
            return VGroup(
                Text(head, font_size=20),
                VGroup(Text("→", font_size=20, color=RED_C), Text(effect, font_size=17, color=GREY_A, line_spacing=0.8)).arrange(RIGHT, buff=0.15, aligned_edge=UP),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        cm_title = Text("Usual counter-drone measures", font_size=22, color=HIGHLIGHT)
        cite = Text("Lykou et al. (2020)", font_size=13, color=GREY_B)
        head = VGroup(cm_title, cite).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        cms = VGroup(
            counter("Jamming / spoofing", "drone falls or flies away uncontrolled;\ninterferes with landing aids and radio"),
            counter("Interception: nets, interceptor drones", "capture or collision over the city:\nfalling drone and debris"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        cm = fit(VGroup(head, cms).arrange(DOWN, aligned_edge=LEFT, buff=0.25), W)
        cm.move_to(np.array([RX, 1.4, 0]), aligned_edge=UL)
        verdict = fit(Text("Over people and infrastructure:\nno jamming, no collision.\nContain the intruder at a safe distance.",
                           font_size=22, color=HIGHLIGHT, line_spacing=0.8), W - 0.3)
        verdict.next_to(cm, DOWN, buff=0.45).align_to(cm, LEFT).shift(RIGHT * 0.15)
        verdict_box = SurroundingRectangle(verdict, color=HIGHLIGHT, buff=0.15, corner_radius=0.1)

        self.play(FadeIn(head), run_time=0.4)
        for row in cms:
            self.play(FadeIn(row, shift=LEFT * 0.2), run_time=0.6)
        self.play(FadeIn(verdict), Create(verdict_box), run_time=0.8)


# =============================================================================
# 2. Problem statement
# =============================================================================
class Problem(Base):
    SECTION = "Problem"

    def construct(self):
        self.header("2. Problem statement")

        # ---- three notions of "encirclement" ---------------------------------
        def panel(title, highlight=False):
            rect = RoundedRectangle(corner_radius=0.15, width=3.9, height=3.6, color=YELLOW if highlight else GREY_C)
            t = Text(title, font_size=22, color=YELLOW if highlight else WHITE)
            if t.width > rect.width - 0.3:
                t.set(width=rect.width - 0.3)
            t.next_to(rect.get_top(), DOWN, buff=0.2)
            return VGroup(rect, t)

        p1 = panel("Convex containment")
        p2 = panel("Reachable-set overlap")
        p3 = panel("Visible formation (this thesis)", highlight=True)
        panels = VGroup(p1, p2, p3).arrange(RIGHT, buff=0.35).shift(DOWN * 0.15)

        c1 = p1[0].get_center() + DOWN * 0.25
        tri = [c1 + 1.2 * np.array([np.cos(a), np.sin(a), 0]) for a in (PI / 2 + 0.3, PI + 0.9, -0.4)]
        poly = Polygon(*tri, color=GREY_B, fill_color=GREY_B, fill_opacity=0.15)
        d1 = VGroup(poly, *[Dot(p, color=WHITE) for p in tri], Dot(c1 + np.array([0.1, -0.1, 0]), color=RED))

        c2 = p2[0].get_center() + DOWN * 0.25
        reach_t = Circle(radius=0.85, color=RED, fill_opacity=0.15).move_to(c2)
        reach_c = VGroup(
            *[
                Circle(radius=0.55, color=WHITE, fill_opacity=0.08, stroke_width=2).move_to(c2 + 1.0 * np.array([np.cos(a), np.sin(a), 0]))
                for a in np.linspace(0, 2 * PI, 5, endpoint=False)
            ]
        )
        d2 = VGroup(reach_t, reach_c, Dot(c2, color=RED), *[Dot(c.get_center(), color=WHITE, radius=0.05) for c in reach_c])

        c3 = p3[0].get_center() + DOWN * 0.25
        ring = Circle(radius=1.1, color=BLUE, stroke_opacity=0.5).move_to(c3)
        d3 = VGroup(ring, Dot(c3, color=RED), *[Dot(c3 + 1.1 * np.array([np.cos(a), np.sin(a), 0]), color=WHITE) for a in np.linspace(0, 2 * PI, 8, endpoint=False)])

        self.play(LaggedStart(*[Create(p) for p in panels], lag_ratio=0.2))
        self.play(FadeIn(d1), FadeIn(d2), FadeIn(d3))
        note = Text(
            "Goal: a visible barrier that warns the operator, not physical capture",
            font_size=24,
            color=GREY_A,
        ).next_to(panels, DOWN, buff=0.35)
        self.play(FadeIn(note))
        self.next_slide()
        self.play(FadeOut(VGroup(panels, d1, d2, d3, note)))

        # ---- 3D problem: chasers converge to evenly spaced points -----------
        proj = Proj(phi=70 * DEGREES, theta=-50 * DEGREES, scale=1.6, center=np.array([-3.7, -0.35, 0]))
        N = 13
        p_star = fibonacci_points(N, 1.0)
        rng = np.random.default_rng(3)
        start = rng.normal(0, 1, (N, 3))
        start = start / np.linalg.norm(start, axis=1, keepdims=True) * rng.uniform(1.7, 2.0, (N, 1))
        start[:, 2] = np.abs(start[:, 2]) * 0.4 - 1.2
        s = ValueTracker(0.0)
        rot = ValueTracker(-50 * DEGREES)

        def scene3d():
            proj.theta = rot.get_value()
            k = s.get_value()
            g = VGroup()
            g.add(sphere_wire(proj, 1.0, n_lat=4, n_lon=5, front_op=0.35 * k + 0.05, back_op=0.06))
            g.add(dots3d(p_star, proj, color=BLUE, radius=0.045, back_op=0.25))
            pos = (1 - k) * start + k * p_star
            g.add(Dot(proj(np.zeros(3)), radius=0.08, color=RED))
            g.add(dots3d(pos, proj, color=WHITE, radius=0.07, back_op=0.4))
            return g

        world = always_redraw(scene3d)
        def entry(color, before, tex, after=""):
            parts = [Dot(radius=0.08, color=color), Text(before, font_size=24, color=color), MathTex(tex, font_size=32, color=color)]
            if after:
                parts.append(Text(after, font_size=24, color=color))
            return VGroup(*parts).arrange(RIGHT, buff=0.15)

        legend = VGroup(
            entry(RED, "target UAV", r"\mathbf{p}_T", "(non-cooperative)"),
            entry(WHITE, "chaser UAVs", r"\mathbf{p}_i"),
            entry(BLUE, "desired points", r"\mathbf{p}_i^\star"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.move_to(RIGHT * 0.6 + UP * 2.3, aligned_edge=LEFT + UP)
        self.play(FadeIn(world), FadeIn(legend))
        self.play(s.animate.set_value(1.0), rot.animate.set_value(-10 * DEGREES), run_time=4, rate_func=smooth)

        objective = MathTex(r"\lim_{t\to\infty}\ \mathbf{p}_i(t)=\mathbf{p}_i^\star,\quad \forall i", font_size=40)
        reqs = bullet_list(
            [
                "Fully 3D, target static or moving",
                "Agents as evenly spaced as possible",
                "Collision-aware motion",
                "Only a noisy target position is known",
                "Decentralized, robust to agent loss",
            ],
            font_size=21,
            buff=0.16,
        )
        right = VGroup(objective, reqs).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        right.next_to(legend, DOWN, buff=0.5).align_to(legend, LEFT)
        self.play(Write(objective))
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in reqs], lag_ratio=0.2))
        self.next_slide()

        scope = Text(
            "Out of scope: malicious operators, physical capture / interception, legal follow-up",
            font_size=20,
            color=GREY_B,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(scope), rot.animate.set_value(20 * DEGREES), run_time=1.5)
        self.next_slide(auto_next=True)
        world.clear_updaters()
        self.clear_all()


# =============================================================================
# 3. Roadmap / contributions
# =============================================================================
def roadmap_grid(font_size=19):
    # the sphere-only decentralized stage (thesis Ch. 5) is left out of the talk: the generalized
    # stage (Ch. 6), decentralized as well, is presented as "Decentralized"
    cols = ["Centralized", "Decentralized"]
    rows = ["Where to go?\n(distribution)", "How to get there?\n(path planning)", "Safely?\n(collisions)"]
    cells = [
        ["Fibonacci lattice\n+ Munkres assignment", "Lloyd + Dijkstra on any mesh\n+ velocity weighting"],
        ["Lyapunov geometric\n+ radial controller", "Bernstein-polynomial MPC\n(v, a limits)"],
        ["APF safeguard\non the sphere", "MPC collision cost\n(ellipsoidal envelope)"],
    ]
    W, H, X0 = 5.0, 1.05, -1.85
    grid = VGroup()
    head = VGroup()
    for j, c in enumerate(cols):
        t = Text(c, font_size=24, color=YELLOW)
        t.move_to(np.array([X0 + j * (W + 0.15), 2.15, 0]))
        head.add(t)
    row_labels = VGroup()
    body = VGroup()
    for i, r in enumerate(rows):
        y = 1.25 - i * (H + 0.15)
        lab = Text(r, font_size=18, color=GREY_A, line_spacing=0.8).move_to(np.array([-5.85, y, 0]))
        row_labels.add(lab)
        row = VGroup()
        for j in range(len(cols)):
            rect = RoundedRectangle(corner_radius=0.1, width=W, height=H, color=GREY_C, stroke_width=1.5)
            rect.move_to(np.array([X0 + j * (W + 0.15), y, 0]))
            txt = Text(cells[i][j], font_size=font_size, line_spacing=0.8)
            if txt.width > W - 0.2:
                txt.set(width=W - 0.2)
            txt.move_to(rect)
            row.add(VGroup(rect, txt))
        body.add(row)
    est = Text(
        "All: target estimate = noisy measurement ($\\pm 5\\%$) → EMA → consensus over a Delaunay spatial network",
        font_size=18,
        color=GREY_A,
    )
    est.move_to(np.array([-0.6, 1.25 - 3 * (H + 0.15) + 0.15, 0]))
    grid.add(head, row_labels, body, est)
    return grid


class Roadmap(Base):
    SECTION = "Problem"

    def construct(self):
        self.header("3. Proposed solution: two stages")
        g = roadmap_grid()
        head, row_labels, body, est = g
        self.play(FadeIn(row_labels))
        for j in range(len(head)):
            self.play(FadeIn(head[j]), *[FadeIn(body[i][j], shift=DOWN * 0.1) for i in range(3)], run_time=0.8)
        self.play(FadeIn(est))
        self.next_slide(auto_next=True)
        self.clear_all()


# =============================================================================
# 4. State of the art
# =============================================================================
class StateOfTheArt(Base):
    SECTION = "Problem"

    def construct(self):
        self.header("4. State of the art (circular formations)")
        centers = [LEFT * 4.4 + UP * 0.9, UP * 0.9, RIGHT * 4.4 + UP * 0.9]
        names = ["Cyclic pursuit", "Leader-follower", "Artificial potential fields"]
        pros = [
            "+ minimal communication,\n   leaderless",
            "+ intuitive, explicit\n   formation geometry",
            "+ Lyapunov-based design,\n   obstacle avoidance",
        ]
        cons = [
            "- may collapse to a point\n   without extra terms",
            "- leader = single point\n   of failure",
            "- local minima, tuning;\n   N must be known",
        ]
        t = ValueTracker(0.0)

        # cyclic pursuit
        def cyc():
            c = centers[0]
            n = 5
            a = t.get_value()
            pts = [c + 1.0 * np.array([np.cos(a + 2 * PI * k / n), np.sin(a + 2 * PI * k / n), 0]) for k in range(n)]
            g = VGroup(Dot(c, color=RED))
            for k in range(n):
                g.add(Arrow(pts[k], pts[(k + 1) % n], buff=0.12, stroke_width=2.5, color=GREY_B, max_tip_length_to_length_ratio=0.2))
                g.add(Dot(pts[k], color=WHITE))
            return g

        # leader-follower
        def lf():
            c = centers[1] + np.array([0.5 * np.sin(t.get_value()), 0.25 * np.sin(2 * t.get_value()), 0])
            g = VGroup()
            lead = Dot(c, color=YELLOW, radius=0.1)
            for k in range(6):
                a = 2 * PI * k / 6 + 0.5 * t.get_value()
                p = c + 0.95 * np.array([np.cos(a), np.sin(a), 0])
                g.add(Line(c, p, color=GREY_D, stroke_width=1.5), Dot(p, color=WHITE))
            g.add(lead)
            return g

        # APF: vector field converging to a ring
        def apf():
            c = centers[2]
            g = VGroup(Circle(radius=0.9, color=BLUE, stroke_opacity=0.6).move_to(c), Dot(c, color=RED))
            for x in np.linspace(-1.5, 1.5, 7):
                for y in np.linspace(-1.2, 1.2, 6):
                    p = np.array([x, y, 0])
                    r = np.linalg.norm(p)
                    if r < 0.15:
                        continue
                    radial = (0.9 - r) * p / r
                    tang = 0.4 * np.array([-p[1], p[0], 0]) / r
                    v = radial + tang
                    v = 0.28 * v / (np.linalg.norm(v) + 1e-9)
                    g.add(Arrow(c + p, c + p + v, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.35))
            return g

        demos = [cyc(), lf(), apf()]   # static diagrams
        texts = VGroup()
        for k in range(3):
            n = Text(names[k], font_size=26, color=YELLOW).move_to(centers[k] + UP * 1.75)
            pr = Text(pros[k], font_size=19, color=GREEN_C, line_spacing=0.8)
            co = Text(cons[k], font_size=19, color=RED_C, line_spacing=0.8)
            pc = VGroup(pr, co).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(centers[k], DOWN, buff=1.5)
            texts.add(VGroup(n, pc))
        for k in range(3):
            self.play(FadeIn(demos[k]), FadeIn(texts[k]), run_time=1.0)

        gap = Text(
            "Gap: most methods are planar (2D); 3D ones assign agents to predefined sectors",
            font_size=24,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.65)
        self.play(FadeIn(gap), run_time=1.0)
        self.next_slide()
        self.clear_all()


# =============================================================================
# 5. Background
# =============================================================================
class Background(Base):
    SECTION = "Background"

    def construct(self):
        self.header("5. Background: Lloyd's algorithm & Voronoi diagrams")
        # --- 2D Lloyd animation -------------------------------------------------
        B = 2.6
        sq_center = LEFT * 3.7 + DOWN * 0.25
        S = 1.0  # screen units per world unit

        def to_screen(p):
            return sq_center + S * np.array([p[0], p[1], 0])

        palette = [BLUE_D, TEAL_D, GREEN_D, GOLD_D, MAROON_D, PURPLE_D, RED_D, ORANGE]
        rng = np.random.default_rng(11)
        sites = rng.uniform(-B * 0.9, -B * 0.1, (8, 2)) * np.array([1, 1]) + np.array([B * 0.6, B * 0.5])
        sites = np.clip(sites, -B + 0.1, B - 0.1)

        def draw_cells(sites):
            cells = voronoi_cells_2d(sites, -B, B, -B, B)
            g = VGroup()
            cents = []
            for k, c in enumerate(cells):
                poly = Polygon(*[to_screen(p) for p in c], stroke_color=WHITE, stroke_width=1.5, fill_color=palette[k], fill_opacity=0.35)
                g.add(poly)
                cents.append(polygon_centroid(c))
            return g, np.array(cents)

        border = Square(side_length=2 * B * S, color=GREY_B).move_to(sq_center)
        cells_mob, cents = draw_cells(sites)
        site_dots = VGroup(*[Dot(to_screen(s), color=WHITE, radius=0.07) for s in sites])

        steps = VGroup(
            Text("1. Partition: each point goes to its closest site", font_size=21),
            Text("2. Move every site to its cell's centroid", font_size=21),
            Text("3. Repeat → centroidal Voronoi tessellation", font_size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        steps.next_to(border, RIGHT, buff=0.5).align_to(border, UP).shift(DOWN * 0.1)
        vor_def = MathTex(
            r"V_i=\{x : d(x,\mathbf{s}_i)\le d(x,\mathbf{s}_j)\ \forall j\neq i\}",
            font_size=32,
        ).next_to(steps, DOWN, buff=0.45).align_to(steps, LEFT)

        self.play(Create(border), FadeIn(site_dots))
        self.play(FadeIn(cells_mob), FadeIn(steps[0]), Write(vor_def))
        self.bring_to_front(site_dots)
        self.play(FadeIn(steps[1]))
        it_label = Text("iteration 0", font_size=20, color=GREY_A).next_to(border, DOWN, buff=0.15)
        self.add(it_label)
        for it in range(1, 9):
            crosses = VGroup(*[Cross(scale_factor=0.08, stroke_width=3, stroke_color=YELLOW).move_to(to_screen(c)) for c in cents])
            rt = 0.9 if it <= 2 else 0.45
            if it <= 2:
                self.play(FadeIn(crosses), run_time=0.4)
            sites = cents
            new_cells, new_cents = draw_cells(sites)
            new_dots = VGroup(*[Dot(to_screen(s), color=WHITE, radius=0.07) for s in sites])
            new_label = Text(f"iteration {it}", font_size=20, color=GREY_A).next_to(border, DOWN, buff=0.15)
            anims = [Transform(site_dots, new_dots), Transform(cells_mob, new_cells), Transform(it_label, new_label)]
            if it <= 2:
                anims.append(FadeOut(crosses))
            self.play(*anims, run_time=rt)
            self.bring_to_front(site_dots)
            cents = new_cents
            if it == 2:
                self.play(FadeIn(steps[2]), run_time=0.4)
        self.next_slide()

        # --- Delaunay neighbours + consensus ----------------------------------
        from scipy.spatial import Delaunay

        tri = Delaunay(sites)
        edges = set()
        for simp in tri.simplices:
            for a in range(3):
                i, j = sorted((simp[a], simp[(a + 1) % 3]))
                edges.add((i, j))
        edge_mob = VGroup(*[Line(to_screen(sites[i]), to_screen(sites[j]), color=YELLOW, stroke_width=2.5) for i, j in edges])
        dual_txt = Text(
            "Neighbours = agents sharing a Voronoi edge\n(Delaunay triangulation → spatial network)",
            font_size=21,
            color=YELLOW,
            line_spacing=0.8,
        ).next_to(vor_def, DOWN, buff=0.5).align_to(steps, LEFT)
        self.play(cells_mob.animate.set_fill(opacity=0.12), Create(edge_mob), FadeIn(dual_txt), run_time=1.2)
        self.bring_to_front(site_dots)
        self.next_slide()

        # consensus on a scalar (target x-position) with noisy initial guesses
        L = np.zeros((8, 8))
        for i, j in edges:
            L[i, j] = L[j, i] = -1
        np.fill_diagonal(L, -L.sum(axis=1))
        x = 5.0 + rng.normal(0, 0.25, 8)
        vals = VGroup(
            *[DecimalNumber(x[k], num_decimal_places=2, font_size=24, color=WHITE).next_to(to_screen(sites[k]), UR, buff=0.02) for k in range(8)]
        )
        cons_txt = VGroup(
            Text("Consensus on the target estimate:", font_size=22),
            MathTex(r"\hat{\mathbf{p}}_{T,i}\leftarrow \hat{\mathbf{p}}_{T,i}+\varepsilon\sum_{j\in\mathcal{N}_i}(\hat{\mathbf{p}}_{T,j}-\hat{\mathbf{p}}_{T,i})", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        cons_txt.next_to(dual_txt, DOWN, buff=0.4).align_to(steps, LEFT)
        self.play(FadeIn(vals), FadeIn(cons_txt))
        eps = 0.12
        for _ in range(8):
            x = x - eps * L @ x
            new_vals = VGroup(
                *[DecimalNumber(x[k], num_decimal_places=2, font_size=24, color=WHITE).next_to(to_screen(sites[k]), UR, buff=0.02) for k in range(8)]
            )
            self.play(Transform(vals, new_vals), run_time=0.35)
        self.next_slide()
        self.clear_body()

        # --- safety toolbox -------------------------------------------------
        self.new_title("5. Background: collision-avoidance toolbox")
        header = ["Method", "Constrains", "Prediction", "Formal safety", "In thesis"]
        rows = [
            ["APF", "reference (potential)", "current state", "no invariance guarantee", "Ch. 4-5"],
            ["CBF-QP", "control input", "current state", "forward invariance", "not used"],
            ["MPC", "predicted positions + v, a", "finite horizon", "constraint/penalty based", "Ch. 6"],
            ["CBF-MPC", "trajectory + barrier", "finite horizon", "MPC + CBF safety", "future work"],
        ]
        widths = [2.0, 3.4, 2.2, 3.2, 1.8]
        table = VGroup()
        y = 1.6
        for r, row in enumerate([header] + rows):
            x = -sum(widths) / 2
            line = VGroup()
            for c, cell in enumerate(row):
                col = YELLOW if r == 0 else (GREEN_C if (r in (1, 3) and c == 0) else WHITE)   # used in the thesis: APF, MPC
                t = Text(cell, font_size=21 if r else 22, color=col)
                if t.width > widths[c] - 0.15:
                    t.set(width=widths[c] - 0.15)
                t.move_to(np.array([x + widths[c] / 2, y, 0]))
                line.add(t)
                x += widths[c]
            table.add(line)
            y -= 0.75
        sep = Line(LEFT * sum(widths) / 2, RIGHT * sum(widths) / 2, color=GREY_C).move_to(np.array([0, 1.6 - 0.375, 0]))
        # used in the thesis (APF, MPC): boxes as wide as the table, so they line up
        hl = VGroup(*[
            RoundedRectangle(corner_radius=0.08, width=table.width + 0.24, height=table[r].height + 0.24,
                             color=GREEN_C).move_to(np.array([table.get_x(), table[r].get_y(), 0]))
            for r in (1, 3)
        ])
        cbf = MathTex(r"\mathcal{C}=\{\mathbf{x}: h(\mathbf{x})\ge 0\},\qquad L_f h+L_g h\,\mathbf{u}+\alpha(h)\ge 0", font_size=32)
        cbf_t = Text("CBF: keeps the safe set forward-invariant", font_size=22, color=GREY_A)
        cbfg = VGroup(cbf_t, cbf).arrange(DOWN, buff=0.2).next_to(table, DOWN, buff=0.5)
        self.play(FadeIn(table[0]), Create(sep))
        self.play(LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in table[1:]], lag_ratio=0.25))
        self.play(Create(hl), FadeIn(cbfg))
        self.next_slide(auto_next=True)
        self.clear_all()


# =============================================================================
# 6. Centralized - point distribution
# =============================================================================
class CentralizedDistribution(Base):
    SECTION = "Centralized"

    def construct(self):
        self.header("6. Centralized: where should each agent go?")
        proj = Proj(phi=68 * DEGREES, theta=-60 * DEGREES, scale=2.3, center=np.array([-3.4, -0.35, 0]))
        n = ValueTracker(5)
        rot = ValueTracker(-60 * DEGREES)

        def cloud():
            proj.theta = rot.get_value()
            N = int(round(n.get_value()))
            if N % 2 == 0:
                N += 1
            pts = fibonacci_points(N, 1.0)
            r = 0.07 if N < 40 else (0.045 if N < 150 else 0.028)
            g = VGroup(sphere_wire(proj, 1.0, n_lat=4, n_lon=5, front_op=0.25, back_op=0.05))
            g.add(dots3d(pts, proj, color=BLUE, radius=r, back_op=0.2))
            return g

        sphere = always_redraw(cloud)
        counter = VGroup(MathTex("P =", font_size=40), Integer(5, font_size=40)).arrange(RIGHT, buff=0.15)
        counter.move_to(np.array([-3.4, -3.15, 0]))
        counter[1].add_updater(lambda m: m.set_value(int(round(n.get_value())) | 1))

        title = Text("Fibonacci lattice", font_size=30, color=YELLOW)
        eq = MathTex(
            r"\text{lat}_i=\arcsin\!\left(\tfrac{2i}{2n+1}\right),\quad \text{lon}_i=2\pi i\,\varphi^{-1}",
            font_size=34,
        )
        eq2 = MathTex(r"i=-n,\dots,n,\qquad P=2n+1,\qquad \varphi=\tfrac{1+\sqrt5}{2}", font_size=30)
        golden = MathTex(r"360^\circ(1-\varphi^{-1})\simeq 137.5^\circ\ \text{(golden angle)}", font_size=30)
        right = VGroup(title, eq, eq2, golden).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(UP * 1.75).to_edge(RIGHT, buff=0.45)

        self.play(FadeIn(sphere), FadeIn(counter))
        self.play(FadeIn(title), Write(eq), FadeIn(eq2), run_time=1.2)
        self.play(n.animate.set_value(401), rot.animate.set_value(20 * DEGREES), run_time=5, rate_func=rate_functions.ease_in_quad)
        self.play(FadeIn(golden))
        self.next_slide()

        # comparison table (thesis appendix, ~1000 points)
        head = ["", "Lat-Lon", "Fibonacci", "Poisson disk"]
        rows = [
            ["mean runtime", "$1.6\\times10^{-3}$ s", "$1.075\\times10^{-4}$ s", "$7.51\\times10^{-2}$ s"],
            ["$CV(d_{NN})$", "0.0068", "0.0105", "0.2533"],
            ["Riesz energy", "553 166", "483 587", "515 388"],
            ["any $P$?", "grid only", "odd $P$", "yes"],
        ]
        widths = [1.85, 1.35, 1.5, 1.6]
        tab = VGroup()
        y = 0.0
        for r, row in enumerate([head] + rows):
            x = 0.0
            line = VGroup()
            for c, cell in enumerate(row):
                col = YELLOW if (r == 0 or c == 2) else (GREY_A if c == 0 else WHITE)
                t = Text(cell, font_size=19, color=col)
                if t.width > widths[c] - 0.12:
                    t.set(width=widths[c] - 0.12)
                t.move_to(np.array([x + widths[c] / 2, y, 0]))
                line.add(t)
                x += widths[c]
            tab.add(line)
            y -= 0.48
        tab.next_to(right, DOWN, buff=0.4).to_edge(RIGHT, buff=0.4)
        box_f = SurroundingRectangle(VGroup(*[tab[r][2] for r in range(5)]), color=YELLOW, buff=0.08, corner_radius=0.06)
        verdict = Text("$\\approx 700\\times$ faster than Poisson disk,\nnear-uniform spacing", font_size=21, color=YELLOW, line_spacing=0.8)
        verdict.next_to(tab, DOWN, buff=0.3).align_to(tab, LEFT)
        src = Text("(thesis appendix, ~1000 points, 100 trials)", font_size=16, color=GREY_B).next_to(verdict, DOWN, buff=0.1).align_to(verdict, LEFT)
        self.play(FadeIn(tab), n.animate.set_value(13), run_time=1.2)
        self.play(Create(box_f), FadeIn(verdict), FadeIn(src))
        self.next_slide(auto_next=True)
        sphere.clear_updaters()
        counter[1].clear_updaters()
        self.clear_all()


# =============================================================================
# 7. Centralized - Lyapunov geometric control
# =============================================================================
def simulate_geometric(q0, r0, qd, rd, k_geo=1.2, k_r=0.6, dt=0.02, T=12.0):
    q = q0 / np.linalg.norm(q0)
    r = r0
    Q, Rr = [q.copy()], [r]
    for _ in range(int(T / dt)):
        c = np.clip(q @ qd, -1 + 1e-9, 1)
        e = skew(qd) @ q / np.sqrt(2 * (1 + c))
        qdot = -k_geo * skew(e) @ q
        rdot = k_r * np.exp(-(e @ e)) * (rd - r)
        q = q + dt * qdot
        q /= np.linalg.norm(q)
        r = r + dt * rdot
        Q.append(q.copy())
        Rr.append(r)
    return np.array(Q), np.array(Rr)


class CentralizedControl(Base):
    SECTION = "Centralized"

    def construct(self):
        self.header("7. Centralized: Lyapunov-based geometric control")

        decomp = MathTex(r"\mathbf{p}_i-\mathbf{p}_T", r"=", r"r_i", r"\,\mathbf{q}_i", r",\qquad \mathbf{q}_i\in\mathbb{S}^2", font_size=40)
        decomp[2].set_color(GREEN_C)
        decomp[3].set_color(YELLOW)
        goals = MathTex(r"r_i\to r^\star", r"\qquad", r"\mathbf{q}_i\to\mathbf{q}_i^\star", font_size=40)
        goals[0].set_color(GREEN_C)
        goals[2].set_color(YELLOW)
        top = VGroup(decomp, goals).arrange(DOWN, buff=0.3).move_to(UP * 2.3)
        sub = Text("radial and angular objectives are decoupled", font_size=22, color=GREY_A).next_to(top, DOWN, buff=0.2)
        # sketch: target, unit sphere, agent, direction q_i and distance r_i
        sk = Proj(phi=70 * DEGREES, theta=-50 * DEGREES, scale=1.25, center=np.array([0.0, -1.5, 0]))
        pa = np.array([2.3, -0.4, 1.4])
        qa = pa / np.linalg.norm(pa)
        sketch = VGroup(
            sphere_wire(sk, 1.0, n_lat=3, n_lon=4, front_op=0.3, back_op=0.06),
            Dot(sk(np.zeros(3)), color=RED, radius=0.08),
            Line(sk(np.zeros(3)), sk(pa), color=GREEN_C, stroke_width=3),
            Dot(sk(pa), color=WHITE, radius=0.09),
            Dot(sk(qa), color=YELLOW, radius=0.08),
            MathTex(r"\mathbf{p}_T", font_size=30, color=RED).next_to(sk(np.zeros(3)), LEFT, buff=0.15),
            MathTex(r"\mathbf{p}_i", font_size=30).next_to(sk(pa), RIGHT, buff=0.12),
            MathTex(r"\mathbf{q}_i", font_size=30, color=YELLOW).next_to(sk(qa), UL, buff=0.08),
            MathTex(r"r_i", font_size=30, color=GREEN_C).next_to(sk(0.75 * pa), DOWN, buff=0.12),
        )
        self.play(Write(decomp), FadeIn(sketch))
        self.play(Write(goals), FadeIn(sub))
        self.next_slide()
        self.play(FadeOut(sketch), run_time=0.4)

        # --- error functions comparison --------------------------------------
        axes = Axes(
            x_range=[0, PI, PI / 4],
            y_range=[0, 1.1, 0.5],
            x_length=5.6,
            y_length=3.0,
            axis_config={"include_tip": False, "color": GREY_B},
            y_axis_config={"include_numbers": True, "font_size": 20},
        ).move_to(LEFT * 3.2 + DOWN * 1.2)
        xl = MathTex(r"\theta=\angle(\mathbf{q},\mathbf{q}^\star)", font_size=28).next_to(axes, DOWN, buff=0.15)
        ticks = VGroup(
            MathTex(r"0", font_size=24).next_to(axes.c2p(0, 0), DOWN, buff=0.1),
            MathTex(r"\pi/2", font_size=24).next_to(axes.c2p(PI / 2, 0), DOWN, buff=0.1),
            MathTex(r"\pi", font_size=24).next_to(axes.c2p(PI, 0), DOWN, buff=0.1),
        )
        xl.next_to(ticks, DOWN, buff=0.1)
        classic = axes.plot(lambda t: np.sin(t), color=GREY_B, x_range=[0, PI])
        proposed = axes.plot(lambda t: np.sin(t / 2), color=YELLOW, x_range=[0, PI])
        lab_c = VGroup(
            Text("classical error", font_size=20, color=GREY_B),
            MathTex(r"\mathbf{e}_r=S(\mathbf{q}^\star)\,\mathbf{q}=\mathbf{q}^\star\times\mathbf{q}", font_size=30, color=GREY_B),
            MathTex(r"\|\mathbf{e}_r\|=\sin\theta", font_size=30, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        lab_p = VGroup(
            Text("proposed error [Ramp 2015]", font_size=20, color=YELLOW),
            MathTex(r"\mathbf{e}_q=\frac{S(\mathbf{q}^\star)\,\mathbf{q}}{\sqrt{2(1+\mathbf{q}^\top\mathbf{q}^\star)}}", font_size=30, color=YELLOW),
            MathTex(r"\|\mathbf{e}_q\|=\sin\tfrac{\theta}{2}", font_size=30, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        warn = Text("classical error vanishes near\nthe antipode: false convergence", font_size=20, color=RED_C, line_spacing=0.8)
        mono = Text("monotonic in $\\theta$: no false convergence", font_size=20, color=YELLOW)
        # right column: classical error, its problem, proposed error, its property
        labs = VGroup(lab_c, warn, lab_p, mono).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        labs.next_to(axes, RIGHT, buff=0.4).align_to(axes, UP).shift(UP * 0.75)
        if labs.get_bottom()[1] < -3.25:
            labs.set(height=labs.height - (-3.25 - labs.get_bottom()[1]))
            labs.next_to(axes, RIGHT, buff=0.4).align_to(axes, UP).shift(UP * 0.75)
        dot_bad = Dot(axes.c2p(PI * 0.97, np.sin(PI * 0.97)), color=RED_C)
        self.play(Create(axes), FadeIn(ticks), FadeIn(xl))
        self.play(Create(classic), FadeIn(lab_c))
        self.play(FadeIn(dot_bad), FadeIn(warn))
        self.play(Create(proposed), FadeIn(lab_p))
        self.play(FadeIn(mono))
        self.next_slide()
        self.play(FadeOut(VGroup(axes, xl, ticks, classic, proposed, labs, warn, dot_bad, mono, sub, top)))

        # --- control laws + theorem ------------------------------------------
        law1 = MathTex(r"\dot{\mathbf{q}}_i=-k_{geo}\,S(\mathbf{e}_{q_i})\,\mathbf{q}_i", font_size=36)
        law2 = MathTex(r"\dot r_i=k_r\,e^{-\|\mathbf{e}_{q_i}\|^2}\,(r^\star-r_i)", font_size=36)
        gate = Text("align first, then close in", font_size=20, color=GREEN_C)
        thm = VGroup(
            Text("Theorem (thesis Ch. 4)", font_size=22, color=YELLOW),
            MathTex(r"V=\Psi(\mathbf{q},\mathbf{q}^\star),\quad \dot V=-k_{geo}\|\mathbf{e}_q\|^2\le 0", font_size=30),
            Text("$(\\mathbf{q}^\\star, r^\\star)$ asymptotically stable on", font_size=20),
            MathTex(r"(\mathbb{S}^2\setminus\{-\mathbf{q}^\star\})\times\mathbb{R}_{>0}", font_size=30),
            Text("radial error: exponential, rate $\\ge k_r/e$", font_size=20),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        thm_box = SurroundingRectangle(thm, color=YELLOW, buff=0.2, corner_radius=0.1)
        laws = VGroup(law1, law2, gate, VGroup(thm_box, thm)).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        laws.move_to(RIGHT * 3.3 + DOWN * 0.25)

        # 3D animation of 5 agents
        proj = Proj(phi=70 * DEGREES, theta=-55 * DEGREES, scale=1.15, center=np.array([-3.3, -0.6, 0]))
        Nag = 5
        rd = 1.0
        qd = fibonacci_points(Nag, 1.0)
        rng = np.random.default_rng(5)
        q0 = rng.normal(0, 1, (Nag, 3))
        q0[0] = -qd[0] + np.array([0.02, 0.01, 0])  # near antipode: still converges
        r0 = rng.uniform(2.2, 2.7, Nag)
        trajs = []
        for i in range(Nag):
            Q, Rr = simulate_geometric(q0[i], r0[i], qd[i], rd)
            trajs.append(Q * Rr[:, None])
        trajs = np.array(trajs)  # (N, T, 3)
        Tn = trajs.shape[1]
        k = ValueTracker(0)

        def world():
            idx = int(k.get_value())
            g = VGroup(sphere_wire(proj, rd, n_lat=4, n_lon=5, front_op=0.3, back_op=0.05))
            g.add(dots3d(qd * rd, proj, color=BLUE, radius=0.055, back_op=0.3))
            g.add(Dot(proj(np.zeros(3)), color=RED, radius=0.08))
            for i in range(Nag):
                tr = trajs[i, : idx + 1 : 4]
                if len(tr) >= 2:
                    pl = VMobject(stroke_color=GREY_B, stroke_width=2).set_points_as_corners(proj(tr))
                    pl.set_stroke(opacity=0.6)
                    g.add(pl)
            g.add(dots3d(trajs[:, idx], proj, color=WHITE, radius=0.07, back_op=0.45))
            return g

        w = always_redraw(world)
        self.play(FadeIn(w), Write(law1))
        self.play(Write(law2), FadeIn(gate))
        self.play(k.animate.set_value(Tn - 1), run_time=6, rate_func=linear)
        self.play(Create(thm_box), FadeIn(thm))
        self.next_slide(auto_next=True)
        w.clear_updaters()
        self.clear_all()


# =============================================================================
# 8. Centralized - safety, estimation, results
# =============================================================================
class CentralizedResults(Base):
    SECTION = "Centralized"

    def construct(self):
        self.header("8. Centralized: collision safeguard & results")

        a1 = captioned(plot_panel(ASSETS + "apf_inactive.png", height=2.6), "APF inactive")
        a2 = captioned(plot_panel(ASSETS + "apf_active.png", height=2.6), "APF active")
        apf_imgs = Group(a1, a2).arrange(RIGHT, buff=0.35).to_edge(LEFT, buff=0.4).shift(UP * 0.5)
        apf_txt = VGroup(
            Text("APF safeguard on the sphere", font_size=26, color=YELLOW),
            Text("• acts on the reference only", font_size=20),
            Text("• tangential repulsion if $d_{ij}<d_{min}$", font_size=20),
            Text("• bounded, re-projected on the sphere", font_size=20),
            Text("• $d_{min}$ = half the lattice spacing", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        apf_txt.next_to(apf_imgs, RIGHT, buff=0.4)
        apf_txt.align_to(apf_imgs, UP)
        f = MathTex(
            r"\mathbf{f}_{ij}=(d_{min}-d_{ij})\frac{\Delta_{ij}^t}{\|\Delta_{ij}^t\|}",
            font_size=32,
        ).next_to(apf_txt, DOWN, buff=0.35).align_to(apf_txt, LEFT)
        self.play(FadeIn(a1), FadeIn(apf_txt[0]))
        self.play(FadeIn(a2), LaggedStart(*[FadeIn(t) for t in apf_txt[1:]], lag_ratio=0.2))
        self.play(Write(f))
        other = Text(
            "+ Munkres assignment of lattice points (min. total geodesic cost)",
            font_size=21,
            color=GREY_A,
        ).next_to(apf_imgs, DOWN, buff=0.45).align_to(apf_imgs, LEFT)
        self.play(FadeIn(other))
        self.next_slide()
        self.clear_body()

        # --- complete architecture (Ch. 4 sub-diagrams merged) ----------------
        self.architecture("8. Centralized: complete architecture", "centralized")
        self.next_slide()
        self.clear_body()

        # --- results 31 agents, moving target ------------------------------
        self.new_title("8. Centralized: 31 agents, moving target")
        p1 = captioned(plot_panel(ASSETS + "c31dyn_trajectory.png", height=4.3), "trajectories")
        p2 = captioned(plot_panel(ASSETS + "c31dyn_radial_hist.png", height=2.55), "distance to target ($r^\\star = 2$ m)")
        p3 = captioned(plot_panel(ASSETS + "c31dyn_inter_agent_dist.png", height=2.55), "inter-agent distance")
        right = Group(p2, p3).arrange(DOWN, buff=0.2)
        layout = Group(p1, right).arrange(RIGHT, buff=0.35).to_edge(LEFT, buff=0.35).shift(DOWN * 0.2)
        self.play(FadeIn(p1))
        self.play(FadeIn(right))
        self.next_slide()
        self.clear_body()

        # --- real-world test -------------------------------------------------
        self.new_title("8. Real-world test: Fibonacci lattice + geometric controller")
        self.flight(
            "flight_geometric.mp4",
            [
                "3 Crazyflies, virtual target on a circle (radius 1.5 m, 0.25 m/s), formation radius 0.8 m",
                "real vs simulated trajectories: about 10-11 cm RMS in the main phase",
            ],
            speed="main phase in real time, the rest at 3x",
            extra=[("#E63CE6", "target (sim)")],
        )
        self.next_slide(auto_next=True)
        self.clear_all()


# =============================================================================
# 9. Decentralized
# =============================================================================
class Decentralized(Base):
    SECTION = "Decentralized"

    def construct(self):
        self.header("9. Decentralized: local Lloyd on the sphere")
        why = VGroup(
            Text("Fibonacci lattice needs the total number of agents", font_size=22),
            Text("Munkres assignment needs every agent's position", font_size=22),
            Text("→ each agent: own Voronoi cell from neighbours → move to its centroid", font_size=22, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_edge(UP, buff=0.95)
        self.play(FadeIn(why[0]), FadeIn(why[1]))
        self.play(FadeIn(why[2]))

        proj = Proj(phi=65 * DEGREES, theta=-40 * DEGREES, scale=2.0, center=np.array([-4.2, -1.15, 0]))
        samples = fibonacci_points(1400, 1.0)
        front = samples @ proj.basis()[2] > 0.02
        samp_f = samples[front]
        palette = [BLUE_C, TEAL_C, GREEN_C, GOLD_C, MAROON_C, PURPLE_C, RED_C, ORANGE, BLUE_A, TEAL_A, YELLOW_E, PINK]
        rng = np.random.default_rng(21)
        sites = rng.normal(0, 1, (12, 3)) + np.array([0.0, -0.8, 0.9])
        sites /= np.linalg.norm(sites, axis=1, keepdims=True)
        ids = list(range(12))

        def cloud(sites, ids):
            owner = np.argmax(samp_f @ sites.T, axis=1)
            cols = [palette[ids[o]] for o in owner]
            g = VGroup(*[Dot(proj(p), radius=0.042, color=c, fill_opacity=0.85) for p, c in zip(samp_f, cols)])
            return g

        def site_mob(sites, ids):
            # stable order: dot k is always agent k, so the Lloyd updates move each agent (no swaps)
            return dots3d(sites, proj, color=WHITE, radius=0.075, back_op=0.3, stable=True)

        outline = Circle(radius=proj.scale, color=GREY_B, stroke_width=1.5).move_to(proj(np.zeros(3)))
        cl = cloud(sites, ids)
        sm = site_mob(sites, ids)
        cv_label = VGroup(MathTex(r"CV(d_{NN}) =", font_size=28), DecimalNumber(nn_cv(sites), num_decimal_places=3, font_size=32)).arrange(RIGHT, buff=0.15)
        n_label = VGroup(Text("agents:", font_size=22), Integer(12, font_size=32)).arrange(RIGHT, buff=0.15)
        info = VGroup(n_label, cv_label).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        info.next_to(outline, RIGHT, buff=0.35).align_to(outline, UP)
        illu = Text("illustrative re-implementation", font_size=16, color=GREY_B).next_to(outline, DOWN, buff=0.08)
        self.play(Create(outline), FadeIn(cl), FadeIn(sm), FadeIn(info), FadeIn(illu))

        def lloyd_iters(sites, ids, n_it, rt):
            for it in range(n_it):
                sites, _ = sphere_lloyd_step(sites, samples)
                new_cl = cloud(sites, ids)
                new_sm = site_mob(sites, ids)
                new_cv = DecimalNumber(nn_cv(sites), num_decimal_places=3, font_size=32).move_to(cv_label[1], aligned_edge=LEFT)
                self.play(Transform(cl, new_cl), Transform(sm, new_sm), Transform(cv_label[1], new_cv), run_time=rt)
            return sites

        sites = lloyd_iters(sites, ids, 3, 0.6)
        sites = lloyd_iters(sites, ids, 9, 0.25)
        self.next_slide()

        # drop 4 agents (closest to the camera so the hole is visible)
        dfront = sites @ proj.basis()[2]
        drop = list(np.argsort(-dfront)[:4])
        crosses = VGroup(*[Cross(scale_factor=0.13, stroke_color=RED, stroke_width=5).move_to(proj(sites[d])) for d in drop])
        lost = Text("4 agents lost", font_size=22, color=RED).next_to(info, DOWN, buff=0.3).align_to(info, LEFT)
        self.play(FadeIn(crosses), FadeIn(lost))
        keep = [i for i in range(12) if i not in drop]
        gone = VGroup(*[sm[d] for d in drop])
        sm.remove(*gone)                      # the remaining dots keep their agents
        sites = sites[keep]
        ids = [ids[i] for i in keep]
        new_n = Integer(8, font_size=32).move_to(n_label[1])
        self.play(FadeOut(crosses), FadeOut(gone), Transform(n_label[1], new_n), Transform(cl, cloud(sites, ids)),
                  Transform(sm, site_mob(sites, ids)), run_time=0.8)
        sites = lloyd_iters(sites, ids, 8, 0.3)
        self.next_slide()

        # --- two implementations ---------------------------------------------
        head = ["", "Convex hull", "Half-plane"]
        rows = [
            ["$CV(d_{NN})$", "0.120", "0.073"],
            ["$d_{NN,min}/d_{NN,avg}$", "0.859", "0.849"],
            ["mean run time", "0.033 s", "1.367 s"],
            ["converged / 100", "100", "93"],
            ["degenerate configs", "possible", "no"],
        ]
        widths = [2.6, 1.7, 1.6]
        tab = VGroup()
        y = 0
        for r, row in enumerate([head] + rows):
            x = 0
            line = VGroup()
            for c, cell in enumerate(row):
                col = YELLOW if r == 0 else (GREY_A if c == 0 else WHITE)
                t = Text(cell, font_size=20, color=col).move_to(np.array([x + widths[c] / 2, y, 0]))
                line.add(t)
                x += widths[c]
            tab.add(line)
            y -= 0.45
        tab.move_to(RIGHT * 3.6 + DOWN * 1.2)
        tab_t = Text("30 agents, 100 runs (thesis appendix)", font_size=18, color=GREY_B).next_to(tab, DOWN, buff=0.2)
        good = [tab[1][2], tab[5][2]]
        bad = [tab[3][2], tab[4][2]]
        self.play(FadeIn(tab), FadeIn(tab_t))
        self.play(*[t.animate.set_color(GREEN_C) for t in good], *[t.animate.set_color(RED_C) for t in bad])
        self.next_slide()
        self.clear_body()

        # --- complete architecture -------------------------------------------
        self.architecture("9. Decentralized: complete architecture", "decentralized")
        self.next_slide()
        self.clear_body()

        # --- results ---------------------------------------------------------
        self.new_title("9. Decentralized: moving target, loss of 4 agents (half-plane)")
        p1 = captioned(plot_panel(ASSETS + "hp_ddrop_radial_hist.png", height=3.05), "distance to target ($r^\\star = 2$ m)")
        p2 = captioned(plot_panel(ASSETS + "hp_ddrop_qd_separation.png", height=3.05), "spacing of desired points")
        p3 = captioned(plot_panel(ASSETS + "hp_ddrop_inter_agent_dist.png", height=3.05), "inter-agent distance")
        row = Group(p1, p2, p3).arrange(RIGHT, buff=0.3).move_to(UP * 0.75)
        if row.width > 13.2:
            row.set(width=13.2)
        facts = VGroup(
            Text("moving target, desired radius $r^\\star = 2$ m, $k_{geo}=100,\\ k_r=0.7$", font_size=20, color=GREY_A),
            Text("approach slower than with a static target: mean distance still ~4.3 m at 30 s", font_size=22),
            Text("desired points stay uniform: $CV(d_{NN})\\lesssim 0.1$, $d_{NN}$ 2.0 → 2.2 m after the loss", font_size=22),
            Text("min. distance ~1.7 m at the loss, no near-collisions", font_size=22),
            Text("no redesign needed when the formation size changes", font_size=22, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        if facts.width > 13.0:
            facts.set(width=13.0)
        facts.next_to(row, DOWN, buff=0.3)
        self.play(LaggedStart(*[FadeIn(p) for p in row], lag_ratio=0.3))
        self.play(LaggedStart(*[FadeIn(f) for f in facts], lag_ratio=0.2))
        self.next_slide(auto_next=True)
        self.clear_all()


# =============================================================================
# 9. Decentralized (thesis Ch. 6, generalized) - arbitrary meshes and mesh controller
# =============================================================================
class GeneralizedMesh(Base):
    SECTION = "Decentralized"

    def construct(self):
        self.header("9. Decentralized: any surface, barrier or enclosure")
        b = captioned(plot_panel(ASSETS + "dijkstra_bust.png", height=3.0), "bust mesh, 8 sites")
        f = captioned(plot_panel(ASSETS + "dijkstra_frog.png", height=3.0), "frog mesh, 8 sites")
        imgs = Group(b, f).arrange(RIGHT, buff=0.35)
        txt = VGroup(
            Text("Lloyd + Dijkstra", font_size=28, color=YELLOW),
            Text("mesh = weighted graph", font_size=22),
            Text("geodesic ≈ shortest path", font_size=22),
            Text("vertex → closest site", font_size=22),
            Text("works on any well-defined", font_size=22),
            Text("mesh (Thingi10K examples)", font_size=22),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        Group(imgs, txt).arrange(RIGHT, buff=0.45).move_to(DOWN * 0.1)
        self.play(FadeIn(imgs), LaggedStart(*[FadeIn(t) for t in txt], lag_ratio=0.15))
        self.next_slide()
        self.clear_body()

        # --- mesh controller animation ---------------------------------------
        self.new_title("9. Decentralized: mesh controller")
        R = 5.0
        meshes = precompute_meshes(320, R)
        # as the thesis figure (fig_mesh): seen from the restricted side, the wall normal pointing
        # right, the points coloured by their distance past the wall (viridis, 0 .. 2R)
        proj = Proj(phi=70 * DEGREES, theta=-62 * DEGREES, scale=0.27, center=np.array([-3.4, -0.1, 0]), world_center=np.array([2.0, 0, 0]))
        x_t = ValueTracker(-7.5)
        viridis = color_gradient([ManimColor(c) for c in ("#4a6fe3", "#21a3a3", "#5ec962", "#fde725")], 101)

        def vcol(x):
            return viridis[int(np.clip(x / (2 * R), 0, 1) * 100)]

        # the wall: large enough for the disc (radius sqrt(2) R), so no vertex leaves it
        L = 7.6
        plane_pts = np.array([[0, -L, -L], [0, L, -L], [0, L, L], [0, -L, L]], float)
        plane = Polygon(*proj(plane_pts), stroke_color=GREY_B, stroke_width=1.5, fill_color=GREY, fill_opacity=0.18)
        n_arrow = Arrow(proj(np.array([0, 0, -L])), proj(np.array([4.0, 0, -L])), buff=0, color=RED_C, stroke_width=3, max_tip_length_to_length_ratio=0.25)
        side_in = Text("restricted airspace", font_size=22, color=RED_C).next_to(n_arrow, RIGHT, buff=0.1)
        side_out = Text("outside", font_size=22, color=GREY_A).next_to(plane, LEFT, buff=0.15).align_to(side_in, DOWN)

        def world():
            tgt = np.array([x_t.get_value(), 0.0, 0.0])
            v, u1, u2 = mesh_state(tgt, meshes, R)
            g = VGroup()
            d = proj.depth(v)
            order = np.argsort(d)
            far = d < np.median(d) if u2 > 0.5 else np.zeros(len(v), bool)
            for kk in order:
                g.add(Dot(proj(v[kk]), radius=0.032, color=vcol(v[kk, 0]), fill_opacity=0.4 if far[kk] else 0.95))
            # the target is behind the wall (seen through it) while outside
            g.add(Star(outer_radius=0.13, color=RED, fill_opacity=1 if tgt[0] >= 0 else 0.55, stroke_width=0).move_to(proj(tgt)))
            return g

        w = always_redraw(world)

        def stage_text():
            tgt = np.array([x_t.get_value(), 0.0, 0.0])
            _, u1, u2 = mesh_state(tgt, meshes, R)
            if u1 < 1e-3:
                s = "planar disc (barrier)"
            elif u2 <= 0:
                s = "disc → cone"
            elif u2 < 1:
                s = "cone → sphere"
            else:
                s = "closed sphere (enclosure)"
            t = VGroup(
                Text(s, font_size=26, color=YELLOW),
                MathTex(rf"i_1={u1:.2f},\quad i_2={u2:.2f}", font_size=30),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            t.move_to(RIGHT * 1.6 + UP * 2.2, aligned_edge=LEFT)
            return t

        st = always_redraw(stage_text)
        formula = MathTex(
            r"\mathbf{V}=(1-i_2)\big[(1-i_1)\mathbf{V}_{\mathrm{disc}}+i_1\mathbf{V}_{\mathrm{cone}}\big]+i_2\mathbf{V}_{\mathrm{sphere}}",
            font_size=28,
        )
        d_txt = Tex(r"$i_1, i_2$: from the signed distance of the target\\to the boundary, normalised by the mesh radius $R$", font_size=28, color=GREY_A)
        why = Text("Agents warn before entry, enclose after it", font_size=22, color=GREEN_C)
        rgt = VGroup(formula, d_txt, why).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        if rgt.width > 5.4:
            rgt.set(width=5.4)
        rgt.move_to(RIGHT * 1.3 + DOWN * 1.6, aligned_edge=LEFT)
        self.play(FadeIn(plane), FadeIn(n_arrow), FadeIn(side_out), FadeIn(side_in), FadeIn(w), FadeIn(st))
        self.play(FadeIn(formula), FadeIn(d_txt))
        self.next_slide()
        # target approaches and crosses the boundary: disc -> cone (complete at d = R/2)
        self.play(x_t.animate.set_value(R / 2), run_time=4.0, rate_func=linear)
        self.next_slide()
        # past the boundary: cone -> sphere (complete at d = R), then the sphere follows the target
        self.play(x_t.animate.set_value(R), run_time=2.5, rate_func=linear)
        self.play(x_t.animate.set_value(8.0), run_time=1.5, rate_func=linear)
        self.play(FadeIn(why))
        self.next_slide()
        self.play(x_t.animate.set_value(-7.5), run_time=4, rate_func=smooth)
        self.next_slide(auto_next=True)
        w.clear_updaters()
        st.clear_updaters()
        self.clear_all()


# =============================================================================
# 10. Decentralized (thesis Ch. 6, generalized) - Bernstein MPC & velocity weighting
# =============================================================================
class GeneralizedMPC(Base):
    SECTION = "Decentralized"

    def construct(self):
        self.header("10. Decentralized: collision-aware MPC")
        cost = MathTex(
            r"\min_{\mathbf{c}}\ w_g\sum_{k}\|\mathbf{p}_i[k]-\mathbf{p}_i^\star\|^2+w_s\sum_k\|\ddot{\mathbf{p}}_i[k]\|^2",
            font_size=32,
        )
        cons = MathTex(
            r"\text{s.t.}\ \ \mathbf{p}_i^{(l)}[0]=\mathbf{p}_i^{(l)},\ \ \|\dot{\mathbf{p}}_i[k]\|\le\bar v,\ \ \|\ddot{\mathbf{p}}_i[k]\|\le\bar a",
            font_size=30,
        )
        col = MathTex(
            r"h_{ij}[k]=\big\|\Theta_{ij}^{-1}(\mathbf{p}_i[k]-\mathbf{p}_j[k])\big\|^2-1\ge 0",
            font_size=30,
            color=ORANGE,
        )
        eqs = VGroup(cost, cons, col).arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(UP, buff=0.9).shift(RIGHT * 0.2)
        self.play(Write(cost))
        self.play(FadeIn(cons))
        self.play(FadeIn(col))

        # 2D illustration: Bezier (degree 5) from agent to goal, avoiding a neighbour
        base = np.array([-6.2, -1.8, 0])
        agent = base + np.array([0.0, 0.0, 0])
        goal = base + np.array([6.0, 0.0, 0])
        nb_start = base + np.array([3.2, -1.3, 0])
        nb_end = base + np.array([2.8, 1.4, 0])

        straight = np.array([agent + (goal - agent) * s for s in np.linspace(0, 1, 6)])
        bent = straight.copy()
        bent[:, 1] += np.array([0, -0.2, -1.0, -1.3, -0.6, 0])

        def bez(ctrl, color=YELLOW):
            pts = bezier_eval(ctrl, np.linspace(0, 1, 60))
            return VMobject(stroke_color=color, stroke_width=4).set_points_smoothly(pts)

        def ctrl_poly(ctrl):
            return VGroup(
                DashedVMobject(VMobject(stroke_color=GREY_B, stroke_width=1.5).set_points_as_corners(ctrl), num_dashes=30),
                *[Square(side_length=0.12, color=GREY_A, fill_opacity=1).move_to(c) for c in ctrl],
            )

        def horizon(ctrl):
            pts = bezier_eval(ctrl, np.linspace(0, 0.8, 5))
            return VGroup(*[Dot(p, radius=0.06, color=YELLOW) for p in pts])

        a_dot = Dot(agent, color=WHITE, radius=0.1)
        a_lab = MathTex(r"\mathbf{p}_i[k]", font_size=30).next_to(a_dot, LEFT, buff=0.12)
        g_dot = Dot(goal, color=BLUE, radius=0.1)
        g_lab = MathTex(r"\mathbf{p}_i^\star", font_size=30, color=BLUE).next_to(g_dot, RIGHT, buff=0.12)
        nb_path = DashedLine(nb_start, nb_end, color=GREY_B)
        nb_mid = (nb_start + nb_end) / 2
        nb = Dot(nb_mid, color=WHITE, radius=0.1)
        ell = Ellipse(width=2.2, height=1.6, color=ORANGE).move_to(nb_mid)
        nb_dir = (nb_end - nb_start) / np.linalg.norm(nb_end - nb_start)
        nb_arrow = Arrow(nb_mid, nb_mid + 0.85 * nb_dir, buff=0.12, color=GREY_A, stroke_width=4,
                         max_tip_length_to_length_ratio=0.35)
        nb_lab = MathTex(r"\mathbf{p}_j[k]", font_size=30).next_to(nb, LEFT, buff=0.15)
        ell_lab = MathTex(r"\Theta_{ij}", font_size=28, color=ORANGE).next_to(ell, RIGHT, buff=0.08).shift(UP * 0.4)

        curve = bez(straight)
        cp = ctrl_poly(straight)
        hz = horizon(straight)
        self.play(FadeIn(a_dot), FadeIn(a_lab), FadeIn(g_dot), FadeIn(g_lab), Create(nb_path), FadeIn(nb), FadeIn(nb_lab), GrowArrow(nb_arrow))
        self.play(Create(curve), FadeIn(cp), FadeIn(hz))
        self.play(Create(ell), FadeIn(ell_lab))
        self.play(Transform(curve, bez(bent)), Transform(cp, ctrl_poly(bent)), Transform(hz, horizon(bent)), run_time=1.5)

        side = VGroup(
            Text("Bernstein polynomial, degree 5", font_size=21),
            Text("→ v, a are linear in the control points", font_size=21),
            Text("horizon 5, $\\|\\mathbf{v}\\|\\le 3$ m/s, $\\|\\mathbf{a}\\|\\le 5$ m/s², $\\Theta$: 1 m", font_size=21),
            Text("neighbours share predicted horizons", font_size=21),
            Text("receding horizon: apply 1st step, re-solve", font_size=21),
            Text("collision = high penalty ($\\rho=100$), not a", font_size=21, color=ORANGE),
            Text("hard constraint → no formal guarantee", font_size=21, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        if side.width > 5.8:  # keep the right edge inside the frame (Portuguese is longer)
            side.set(width=5.8)
        side.move_to(RIGHT * 4.0 + DOWN * 1.5)
        self.play(LaggedStart(*[FadeIn(s) for s in side], lag_ratio=0.15))

        # receding horizon: move agent along, re-solve
        for step in range(3):
            pts = bezier_eval(bent, [0.2])[0]
            new_bent = np.array([pts + (goal - pts) * s for s in np.linspace(0, 1, 6)])
            new_bent[:, 1] += np.array([0, -0.15, -0.6, -0.7, -0.3, 0]) * (2 - step) / 2
            self.play(
                a_dot.animate.move_to(pts),
                a_lab.animate.next_to(pts, LEFT, buff=0.12),
                Transform(curve, bez(new_bent)),
                Transform(cp, ctrl_poly(new_bent)),
                Transform(hz, horizon(new_bent)),
                nb.animate.shift((nb_end - nb_start) * 0.18),
                nb_lab.animate.shift((nb_end - nb_start) * 0.18),
                nb_arrow.animate.shift((nb_end - nb_start) * 0.18),
                ell.animate.shift((nb_end - nb_start) * 0.18),
                ell_lab.animate.shift((nb_end - nb_start) * 0.18),
                run_time=0.9,
            )
            bent = new_bent
        self.next_slide()
        self.clear_body()

        # --- velocity weighting ----------------------------------------------
        # 12 sites on a sphere mesh (1000 vertices): uniform CVT first, then Lloyd with the
        # velocity-weighted Dijkstra cells once the target velocity is known
        self.new_title("10. Decentralized: using the target velocity")
        mesh = SphereMesh(1000)
        vel = np.array([1.0, 0.0, 0.0])
        w_uni = mesh.velocity_weights(None)
        w_vel = mesh.velocity_weights(vel, sigma=1.2)
        sites = mesh.nearest(fibonacci_points(12))
        for _ in range(40):
            sites = mesh.lloyd_step(sites, w_uni)[0]
        uni = sites
        hist = []                               # sites per Lloyd iteration with the weighted metric
        for _ in range(30):
            new = mesh.lloyd_step(sites, w_vel)[0]
            if np.array_equal(new, sites):
                break
            hist.append(new)
            sites = new

        proj = Proj(phi=72 * DEGREES, theta=-72 * DEGREES, scale=2.4, center=np.array([-3.7, -0.4, 0]))
        depth = mesh.V @ proj.basis()[2]

        def agents(sites):
            g = dots3d(mesh.V[sites], proj, color=DESIRED_COLOR, radius=0.1, back_op=0.35, stable=True, z_base=2.0)
            for d, k in zip(g, sites):
                d.set_stroke(WHITE, width=2.0, opacity=1.0 if depth[k] >= 0 else 0.35)
            return g

        def stats(sites):
            p = mesh.V[sites] @ vel
            n_fwd = int((p > 0).sum())
            return VGroup(
                MathTex(rf"A = {p.mean():.2f}", font_size=34, color=YELLOW),
                MathTex(rf"F_{{\mathrm{{fwd}}}} = {n_fwd}/12 = {n_fwd / 12:.2f}", font_size=34, color=YELLOW),
            )

        t = np.linspace(0, 2 * np.pi, 121)
        divide = curve3d(np.column_stack([np.zeros_like(t), np.cos(t), np.sin(t)]), proj, color=YELLOW,
                         width=2.0, front_op=0.6, back_op=0.15, sphere_center=np.zeros(3))
        sil = Circle(radius=proj.scale, color=GREY_B, stroke_width=1.5).move_to(proj(ORIGIN))
        tgt = Dot(proj(ORIGIN), radius=0.11, color=TARGET_COLOR).set_z_index(1.5)
        agent_g = agents(uni)

        cap = Text("target velocity unknown: uniform CVT", font_size=22, color=GREY_A)
        cap.move_to(np.array([-6.6, 2.55, 0]), aligned_edge=LEFT)
        w = MathTex(r"w_i=\exp\!\Big(\frac{\theta_i^2}{2\sigma^2}\Big),\ \ \sigma=1.2", font_size=32)
        wt = Text("$\\theta_i$: angle from the direction\nbehind the target\n→ more weight in front", font_size=20, color=GREY_A, line_spacing=0.8)
        st = stats(uni)
        st_lab = VGroup(
            Text("mean directional alignment\n(≈ 0 for a symmetric formation)", font_size=18, color=GREY_A, line_spacing=0.8),
            Text("agents in front of the target", font_size=18, color=GREY_A),
        )
        it_lab = Text("iteration 0", font_size=22, color=GREY_B)
        rcol_x = 1.25

        # right column: weight (top), iteration, A + label, F_fwd + label
        VGroup(w, wt).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(np.array([rcol_x, 1.15, 0]), aligned_edge=LEFT)
        it_lab.move_to(np.array([rcol_x, -0.4, 0]), aligned_edge=LEFT)
        for m in st_lab:
            if m.get_right()[0] - m.get_left()[0] > 6.9 - rcol_x:
                m.set(width=6.9 - rcol_x)

        def place_stats(s):
            s[0].move_to(np.array([rcol_x, -1.0, 0]), aligned_edge=LEFT)
            s[1].next_to(st_lab[0], DOWN, buff=0.3).align_to(s[0], LEFT)
            return s

        st[0].move_to(np.array([rcol_x, -1.0, 0]), aligned_edge=LEFT)
        st_lab[0].next_to(st[0], DOWN, buff=0.1).align_to(st[0], LEFT)
        place_stats(st)
        st_lab[1].next_to(st[1], DOWN, buff=0.1).align_to(st[0], LEFT)

        self.play(FadeIn(sil), FadeIn(divide), FadeIn(tgt), FadeIn(cap))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in agent_g], lag_ratio=0.05), run_time=1.0)
        self.play(FadeIn(st), FadeIn(st_lab))
        self.next_slide()

        # the velocity becomes known: the weighted metric moves the sites, Lloyd re-converges
        tip = proj(1.45 * vel)
        v_arrow = Arrow(proj(ORIGIN), tip, buff=0, color=TARGET_COLOR, stroke_width=7,
                        max_tip_length_to_length_ratio=0.12).set_z_index(3)
        v_lab = MathTex(r"\mathbf{v}", font_size=40, color=TARGET_COLOR).next_to(tip, UP, buff=0.12).set_z_index(3)
        cap2 = Text("target velocity known: weighted CVT (Lloyd)", font_size=22, color=YELLOW).move_to(cap, aligned_edge=LEFT)
        self.play(GrowArrow(v_arrow), FadeIn(v_lab), Transform(cap, cap2), run_time=1.0)
        self.play(Write(w), FadeIn(wt), FadeIn(it_lab), run_time=1.2)
        for k, s in enumerate(hist, start=1):
            new_st = place_stats(stats(s))
            new_it = Text(f"iteration {k}", font_size=22, color=GREY_B).move_to(it_lab, aligned_edge=LEFT)
            self.play(
                Transform(agent_g, agents(s)),
                Transform(st, new_st),
                Transform(it_lab, new_it),
                run_time=0.7,
            )
        self.next_slide()
        self.clear_body()

        # --- complete architecture, before the results -----------------------
        self.architecture("10. Decentralized: complete architecture", "generalized")
        self.next_slide(auto_next=True)
        self.clear_all()


# =============================================================================
# 11. Results - generalized simulation replay + plots
# =============================================================================
class Results(Base):
    SECTION = "Results"

    def construct(self):
        self.header("11. Results: 12 agents, moving target, 4 lost at t = 10 s")
        data = np.load(DATA + "generalized_sim_12agents.npz")
        t = data["t"]
        P = data["P"]  # (12, 3, T)
        PD = data["pd"]
        T = data["T"]  # (3, T)
        active = set(int(a) for a in data["active"])
        R = float(data["R"])
        Tn = len(t)
        meshes = precompute_meshes(260, R)   # the thesis cone (rim on the wall), as in the simulated run

        proj = Proj(phi=68 * DEGREES, theta=-115 * DEGREES, scale=0.205, center=np.array([-2.6, -0.6, 0]), world_center=np.array([-5.0, 0.0, 5.0]))
        plane_pts = np.array([[0, -20, 0], [0, 20, 0], [0, 20, 13], [0, -20, 13]], float)
        plane = Polygon(*proj(plane_pts), stroke_color=GREY_B, stroke_width=1.5, fill_color=RED_E, fill_opacity=0.12)
        ground = VGroup(
            *[DashedLine(proj(np.array([x, -20, 0])), proj(np.array([x, 20, 0])), color=GREY_D, stroke_width=1) for x in (-20, -10, 10)],
            *[DashedLine(proj(np.array([-22, y, 0])), proj(np.array([12, y, 0])), color=GREY_D, stroke_width=1) for y in (-20, -10, 0, 10, 20)],
        )
        lab_in = Text("restricted ($x>0$)", font_size=20, color=RED_C).move_to(proj(np.array([6, 22, 13])))
        k = ValueTracker(0)

        def world():
            i = int(round(k.get_value()))
            g = VGroup()
            tg = T[:, i]
            v, _, _ = mesh_state(tg, meshes, R)
            g.add(*[Dot(proj(p), radius=0.018, color=BLUE_E, fill_opacity=0.6) for p in v])
            path = T[:, : i + 1 : 2].T
            if len(path) >= 2:
                g.add(VMobject(stroke_color=RED_C, stroke_width=2).set_points_as_corners(proj(path)).set_stroke(opacity=0.6))
            for a in range(P.shape[0]):
                lost = (a not in active) and t[i] >= 10.0
                tr = P[a, :, max(0, i - 40) : i + 1 : 2].T
                if len(tr) >= 2:
                    g.add(VMobject(stroke_color=GREY_C if not lost else GREY_D, stroke_width=1.5).set_points_as_corners(proj(tr)).set_stroke(opacity=0.6))
                pd = PD[a, :, i]
                if not np.any(np.isnan(pd)):
                    g.add(Square(side_length=0.08, color=BLUE, fill_opacity=1).rotate(PI / 4).move_to(proj(pd)))
                if lost:
                    g.add(Cross(scale_factor=0.07, stroke_color=GREY_C, stroke_width=3).move_to(proj(P[a, :, i])))
                else:
                    g.add(Dot(proj(P[a, :, i]), radius=0.06, color=WHITE))
            g.add(Dot(proj(tg), radius=0.08, color=RED))
            return g

        w = always_redraw(world)
        clock = always_redraw(
            lambda: VGroup(MathTex("t =", font_size=36), DecimalNumber(t[int(round(k.get_value()))], num_decimal_places=1, font_size=36), Text("s", font_size=28))
            .arrange(RIGHT, buff=0.12)
            .move_to(RIGHT * 2.5 + UP * 2.4, aligned_edge=LEFT)
        )
        leg = VGroup(
            legend_entry(RED, "target"),
            legend_entry(WHITE, "agents"),
            VGroup(Square(side_length=0.12, color=BLUE, fill_opacity=1).rotate(PI / 4), Text("Lloyd goals", font_size=24, color=BLUE)).arrange(RIGHT, buff=0.15),
            legend_entry(BLUE_E, "mesh", font_size=24, radius=0.04),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        leg.move_to(RIGHT * 2.5 + UP * 1.25, aligned_edge=LEFT)
        events = VGroup(
            Text("$t\\approx 1$ s: disc at the boundary", font_size=20),
            Text("$t\\approx 2.5$ s: cone", font_size=20),
            Text("$t\\approx 4$ s: closed sphere", font_size=20),
            Text("$t=10$ s: 4 agents lose power", font_size=20, color=RED_C),
            Text("$t\\approx 13$ s: back to planar barrier", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        events.next_to(leg, DOWN, buff=0.4).align_to(leg, LEFT)
        src = Text("Ch. 6 scenario, Python simulation with the thesis mesh controller, 2x speed", font_size=16, color=GREY_B).to_edge(DOWN, buff=0.55).to_edge(LEFT, buff=0.4)

        self.play(FadeIn(plane), FadeIn(ground), FadeIn(lab_in), FadeIn(w), FadeIn(clock), FadeIn(leg), FadeIn(src))
        # one key press per phase: the replay stops at every event
        marks = [(1.0, 0), (2.5, 1), (4.0, 2), (10.0, 3), (13.0, 4)]
        prev = 0.0
        for n_mark, (tm, e) in enumerate(marks):
            if n_mark:
                self.next_slide()
            idx = int(np.searchsorted(t, tm))
            self.play(k.animate.set_value(idx), run_time=max((tm - prev) / 2, 0.5), rate_func=linear)
            self.play(FadeIn(events[e]), run_time=0.3)
            prev = tm
        self.next_slide()
        self.play(k.animate.set_value(Tn - 1), run_time=(t[-1] - prev) / 2, rate_func=linear)
        self.next_slide()
        w.clear_updaters()
        clock.clear_updaters()
        self.clear_body()

        # --- plots ------------------------------------------------------------
        self.new_title("11. Results: quantitative view")
        p1 = captioned(plot_panel(ASSETS + "g_radial_hist_4_dropped_agents.png", height=2.4), "distance to target ($R=5$ m)", 18)
        p2 = captioned(plot_panel(ASSETS + "g_inter_agent_dist_12_agents.png", height=2.4), "inter-agent distance (threshold 1 m)", 18)
        p3 = captioned(plot_panel(ASSETS + "g_velocity_12_agents.png", height=2.4), "velocity components ($\\|\\mathbf{v}\\|\\le 3$ m/s)", 18)
        p4 = captioned(plot_panel(ASSETS + "g_low_level_tracking_error_12_agents.png", height=2.4), "distance to Lloyd goal", 18)
        grid = Group(p1, p2, p3, p4).arrange_in_grid(rows=2, cols=2, buff=(0.3, 0.2)).move_to(LEFT * 2.3 + DOWN * 0.2)
        if grid.height > 6.2:
            grid.set(height=6.2)
        facts = VGroup(
            Text("+ min. distance 3-4 m after", font_size=21, color=GREEN_C),
            Text("   the initial transient", font_size=21, color=GREEN_C),
            Text("+ velocity bounds respected", font_size=21, color=GREEN_C),
            Text("+ recovers after losing 4 agents", font_size=21, color=GREEN_C),
            Text("- one agent passes close to", font_size=21, color=RED_C),
            Text("   the target (2-5 s): no", font_size=21, color=RED_C),
            Text("   agent-target constraint", font_size=21, color=RED_C),
            Text("- goal error up to ~4 m when", font_size=21, color=RED_C),
            Text("   velocity-limited; 1 Hz", font_size=21, color=RED_C),
            Text("   Voronoi update → staircase", font_size=21, color=RED_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        facts.next_to(grid, RIGHT, buff=0.35)
        self.play(LaggedStart(*[FadeIn(p) for p in grid], lag_ratio=0.25))
        self.play(LaggedStart(*[FadeIn(f) for f in facts], lag_ratio=0.1))
        self.next_slide()
        self.clear_body()

        # --- real-world test -------------------------------------------------
        self.new_title("11. Real-world test: geodesic CVT + Bernstein MPC")
        self.flight(
            "flight_mpc.mp4",
            [
                "4 Crazyflies, virtual target on a circle (radius 1.6 m), 1001-vertex mesh, CVT every 0.1 s",
                "real vs simulated trajectories: about 3-7 cm RMS in the formation phase",
            ],
            speed="main phase in real time, the rest at 3x",
            extra=[("#E63CE6", "target (sim)"), ("#5ABEEB", "formation mesh")],
        )
        self.next_slide()
        self.clear_body()

        # --- Lisbon airport scenario (simulation) ----------------------------
        self.new_title("11. Lisbon airport: intrusion across the prohibited zone (simulation)")
        self.lisbon()
        self.next_slide(auto_next=True)
        self.clear_all()

    def lisbon(self):
        """3D replay of thesis trajectory 5 (12 chasers, 4 lost at 25 s) in the wall frame."""
        data = np.load(DATA + "lisbon_traj5.npz")
        t, P, T, G, alive = data["t"], data["P"], data["T"], data["G"], data["alive"]
        x_red = float(data["x_red"])
        Tn = len(t)
        speed = 3

        proj = Proj(phi=66 * DEGREES, theta=-112 * DEGREES, scale=0.105, center=np.array([-3.55, -0.75, 0]), world_center=np.array([9.0, 2.0, 8.0]))
        y0, y1, zt = -24.0, 28.0, 24.0

        def plane(x):
            return proj(np.array([[x, y0, 0], [x, y1, 0], [x, y1, zt], [x, y0, zt]], float))

        red = Polygon(*plane(x_red), stroke_color=RED_C, stroke_width=2, fill_color=RED_E, fill_opacity=0.22)
        wall = DashedVMobject(Polygon(*plane(0.0), stroke_color=GREY_A, stroke_width=2), num_dashes=60)
        ground = VGroup(
            *[DashedLine(proj(np.array([x, y0, 0])), proj(np.array([x, y1, 0])), color=GREY_D, stroke_width=1) for x in (-20, -10, 10, 20, 40)],
            *[DashedLine(proj(np.array([-24, y, 0])), proj(np.array([42, y, 0])), color=GREY_D, stroke_width=1) for y in (y0, -12, 0, 12, y1)],
        )
        lab_wall = Text("wall (30 m outside)", font_size=18, color=GREY_A).next_to(proj(np.array([0, y1, zt])), UP, buff=0.08)
        lab_red = Text("prohibited zone", font_size=18, color=RED_C).next_to(proj(np.array([x_red, y1, zt])), UP, buff=0.08)
        k = ValueTracker(0)

        def world():
            i = int(round(k.get_value()))
            g = VGroup()
            path = T[:, : i + 1 : 4].T
            if len(path) >= 2:
                g.add(VMobject(stroke_color=RED_C, stroke_width=2).set_points_as_corners(proj(path)).set_stroke(opacity=0.6))
            for a in range(P.shape[0]):
                if not alive[a, i]:
                    g.add(Cross(scale_factor=0.06, stroke_color=GREY_C, stroke_width=3).move_to(proj(P[a, :, i])))
                    continue
                tr = P[a, :, max(0, i - 60) : i + 1 : 3].T
                if len(tr) >= 2:
                    g.add(VMobject(stroke_color=GREY_C, stroke_width=1.5).set_points_as_corners(proj(tr)).set_stroke(opacity=0.6))
                g.add(Square(side_length=0.07, color=BLUE, fill_opacity=1).rotate(PI / 4).move_to(proj(G[a, :, i])))
                g.add(Dot(proj(P[a, :, i]), radius=0.055, color=WHITE))
            g.add(Dot(proj(T[:, i]), radius=0.08, color=RED))
            return g

        w = always_redraw(world)
        air = captioned(plot_panel(ASSETS + "g_lisbon_airspace.png", width=3.3), "Lisbon airport (LPPT), ANAC zones", 16)
        air.move_to(RIGHT * 0.9 + UP * 1.85, aligned_edge=LEFT)
        clock = always_redraw(
            lambda: VGroup(MathTex("t =", font_size=36), DecimalNumber(t[int(round(k.get_value()))], num_decimal_places=1, font_size=36), Text("s", font_size=28))
            .arrange(RIGHT, buff=0.12)
            .move_to(LEFT * 6.7 + UP * 2.65, aligned_edge=LEFT)
        )
        events = VGroup(
            Text("barrier on the wall, slides with the target", font_size=19),
            Text("$t=13.9$ s: target crosses the wall → sphere", font_size=19),
            Text("$t=25$ s: 4 of 12 chasers lose power", font_size=19, color=RED_C),
            Text("$t=34.1$ s: target enters → barrier on the border", font_size=19),
            Text("8 chasers hold 1.00 m from the border", font_size=19, color=GREEN_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        if events.width > 6.0:
            events.set(width=6.0)
        events.move_to(RIGHT * 0.9 + DOWN * 0.75, aligned_edge=LEFT)
        facts = VGroup(
            Text("min. inter-agent distance 1.43 m (envelope 1 m)", font_size=17, color=GREY_A),
            Text("peak speed 3.09 m/s (limit 3 m/s) · X500 model", font_size=17, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        facts.next_to(events, DOWN, buff=0.25).align_to(events, LEFT)
        src = Text(f"thesis simulation data (Ch. 6, trajectory 5), {speed}x speed", font_size=16, color=GREY_B).to_edge(DOWN, buff=0.55).shift(LEFT * 3.4)

        self.play(FadeIn(ground), FadeIn(red), FadeIn(wall), FadeIn(lab_wall), FadeIn(lab_red), FadeIn(w), FadeIn(clock), FadeIn(air), FadeIn(src), FadeIn(events[0]))
        prev = 0.0
        for n_mark, (tm, e) in enumerate([(13.9, 1), (25.0, 2), (34.1, 3)]):
            if n_mark:
                self.next_slide()          # one key press per phase
            idx = int(np.searchsorted(t, tm))
            self.play(k.animate.set_value(idx), run_time=(tm - prev) / speed, rate_func=linear)
            self.play(FadeIn(events[e]), run_time=0.3)
            prev = tm
        self.next_slide()
        self.play(k.animate.set_value(Tn - 1), run_time=(t[-1] - prev) / speed, rate_func=linear)
        self.play(FadeIn(events[4]), FadeIn(facts), run_time=0.5)
        w.clear_updaters()
        clock.clear_updaters()


# =============================================================================
# 12. Conclusions
# =============================================================================
class Conclusions(Base):
    SECTION = "Conclusions"

    def construct(self):
        self.header("12. Conclusions")
        items = VGroup(
            VGroup(
                Text("Centralized", font_size=26, color=YELLOW),
                Text("Fibonacci lattice + Lyapunov geometric controller: simplest, scales to 31 agents,", font_size=21),
                Text("almost-global convergence proof (all initial conditions except the antipode)", font_size=21),
            ),
            VGroup(
                Text("Decentralized", font_size=26, color=YELLOW),
                Text("Lloyd + Dijkstra on any mesh: barrier and enclosure; velocity weighting biases", font_size=21),
                Text("agents forward; one Bernstein MPC enforces $v$, $a$ limits, penalizes collisions;", font_size=21),
                Text("consensus over the Delaunay network: formation recovers after losing 4 of 12 agents", font_size=21),
            ),
        )
        for it in items:
            it.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(UP * 0.55)
        lim = VGroup(
            Text("Limitations", font_size=24, color=RED_C),
            Text("indoor flights only (Crazyflies, virtual target) · instantaneous communication ·", font_size=20, color=GREY_A),
            Text("ground-truth target + noise · collisions penalized, not formally excluded ·", font_size=20, color=GREY_A),
            Text("lost agents ignored, not obstacles", font_size=20, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        lim.next_to(items, DOWN, buff=0.4).align_to(items, LEFT)
        for it in items:
            self.play(FadeIn(it, shift=RIGHT * 0.2), run_time=0.7)
        self.play(FadeIn(lim))
        self.next_slide(auto_next=True)
        self.clear_all()


class FutureWork(Base):
    SECTION = "Conclusions"

    def construct(self):
        self.header("13. Future work")
        left = VGroup(
            Text("System maturity", font_size=26, color=YELLOW),
            Text("• real UAVs: PX4, ROS 2, Gazebo, Pixhawk", font_size=21),
            Text("• realistic communication (latency, loss)", font_size=21),
            Text("• real perception: detection, LiDAR / RGB-D", font_size=21),
            Text("• regulatory framework, authorities", font_size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        right = VGroup(
            Text("New functionality", font_size=26, color=YELLOW),
            Text("• match target velocity inside the MPC", font_size=21),
            Text("• velocity-/shape-aware mesh deformation", font_size=21),
            Text("• reachability-set based formations", font_size=21),
            Text("• CBF-MPC: formal safety guarantees", font_size=21),
            Text("• cheaper obstacle avoidance", font_size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        cols = VGroup(left, right).arrange(RIGHT, buff=1.0, aligned_edge=UP).move_to(UP * 0.3)
        if cols.width > 12.8:
            cols.set(width=12.8)
        self.play(FadeIn(left, shift=RIGHT * 0.2))
        self.play(FadeIn(right, shift=LEFT * 0.2))
        self.next_slide(auto_next=True)
        self.clear_all()


class ThankYou(Base):
    def construct(self):
        proj = Proj(phi=70 * DEGREES, theta=0, scale=1.6, center=UP * 0.7)
        pts = fibonacci_points(21, 1.0)
        rot = ValueTracker(0.0)

        def world():
            proj.theta = rot.get_value()
            return VGroup(
                sphere_wire(proj, 1.0, n_lat=4, n_lon=5, front_op=0.25, back_op=0.05),
                Dot(proj(np.zeros(3)), color=RED, radius=0.09),
                dots3d(pts, proj, color=WHITE, radius=0.06, back_op=0.3),
            )

        w = always_redraw(world)
        thanks = Text("Thank you!", font_size=56).shift(DOWN * 1.7)
        self.play(FadeIn(w), Write(thanks))
        self.play(rot.animate.set_value(PI), run_time=3, rate_func=linear)
        self.next_slide(loop=True)
        self.play(rot.animate.set_value(3 * PI), run_time=6, rate_func=linear)
        self.next_slide(auto_next=True)
        w.clear_updaters()
        self.clear_all()


# =============================================================================
# Backup slides
# =============================================================================
class BackupLyapunov(Base):
    def construct(self):
        self.header("Backup: stability of the geometric controller")
        lines = VGroup(
            MathTex(r"\Psi(\mathbf{q},\mathbf{q}^\star)=2-\sqrt{2(1+\mathbf{q}^\top\mathbf{q}^\star)},\qquad \mathbf{e}_q=\frac{S(\mathbf{q}^\star)\mathbf{q}}{\sqrt{2(1+\mathbf{q}^\top\mathbf{q}^\star)}}", font_size=34),
            MathTex(r"\Psi(\mathbf{q}^\star)=0,\quad \Psi(-\mathbf{q}^\star)=2,\quad 0\le\Psi\le 2", font_size=32),
            MathTex(r"\dot{\mathbf{q}}=S(\boldsymbol\omega)\mathbf{q},\ \ \boldsymbol\omega=-k_{geo}\mathbf{e}_q\ \Rightarrow\ \dot V=\dot\Psi=\mathbf{e}_q^\top\boldsymbol\omega=-k_{geo}\|\mathbf{e}_q\|^2\le 0", font_size=32),
            MathTex(r"D_c=\{\mathbf{q}:\Psi\le c\},\ c<2:\ \text{compact, positively invariant, excludes }-\mathbf{q}^\star", font_size=32),
            MathTex(r"\text{LaSalle on }D_c:\ \mathbf{e}_q=0\iff\mathbf{q}=\mathbf{q}^\star;\ \ D_c\uparrow\mathbb{S}^2\setminus\{-\mathbf{q}^\star\}\ \text{as}\ c\to2", font_size=32),
            MathTex(r"\dot e_r=-\lambda(t)e_r,\ \ \lambda=k_r e^{-\|\mathbf{e}_q\|^2}\ge k_r e^{-1}\ \ (\|\mathbf{e}_q\|^2=\sin^2\tfrac{\theta}{2}<1)", font_size=32),
            MathTex(r"\Rightarrow |e_r(t)|\le|e_r(0)|e^{-\lambda_{min}t}\ \ \text{(cascade: attitude independent of } r)", font_size=32),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        if lines.width > 13:
            lines.set(width=13)
        lines.move_to(DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(l) for l in lines], lag_ratio=0.15))
        self.next_slide(auto_next=True)
        self.clear_all()


class BackupControllers(Base):
    def construct(self):
        self.header("Backup: path-planning comparison (3 agents, $r^\\star = 2$ m, 30 s)")
        head = ["Metric", "fmincon (SQP)", "APF", "Lyapunov"]
        rows = [
            ["time to goal (all < 0.1 m)", "> 30 s", "4.460 s", "17.610 s"],
            ["final goal error, max [m]", "1.0459", "$9.1\\times10^{-14}$", "0.0024"],
            ["final ref. tracking error, max [m]", "1.1673", "$1.5\\times10^{-11}$", "$7.4\\times10^{-5}$"],
            ["mean pairwise distance [m]", "4.7923", "3.3397", "3.4645"],
            ["total computation time [s]", "14.761", "9.249", "7.826"],
        ]
        widths = [4.4, 2.4, 2.0, 2.0]
        tab = VGroup()
        y = 0
        for r, row in enumerate([head] + rows):
            x = 0
            line = VGroup()
            for c, cell in enumerate(row):
                col = YELLOW if r == 0 else (GREY_A if c == 0 else WHITE)
                line.add(Text(cell, font_size=22, color=col).move_to(np.array([x + widths[c] / 2, y, 0])))
                x += widths[c]
            tab.add(line)
            y -= 0.6
        tab.move_to(UP * 0.4)
        note = Text(
            "Lyapunov chosen: near-zero tracking error, least computation, stability proof;\n"
            "slower because it aligns the direction before closing the radius.\n"
            "APF as a distribution method needs N known in advance (kept only as a safeguard).",
            font_size=21,
            color=GREY_A,
            line_spacing=0.85,
        ).next_to(tab, DOWN, buff=0.5)
        self.play(FadeIn(tab))
        self.play(FadeIn(note))
        self.next_slide(auto_next=True)
        self.clear_all()


def table_grid(head, rows, widths, font_size=20, row_h=0.5, first_col_color=GREY_A, highlight=None):
    """Simple table: header row (yellow), first column grey; highlight = column index framed in yellow."""
    tab = VGroup()
    y = 0.0
    for r, row in enumerate([head] + rows):
        x = 0.0
        line = VGroup()
        for c, cell in enumerate(row):
            col = YELLOW if r == 0 else (first_col_color if c == 0 else WHITE)
            t = Text(cell, font_size=font_size, color=col)
            if t.width > widths[c] - 0.12:
                t.set(width=widths[c] - 0.12)
            t.move_to(np.array([x + widths[c] / 2, y, 0]))
            line.add(t)
            x += widths[c]
        tab.add(line)
        y -= row_h
    sep = Line(np.array([0, -row_h / 2, 0]), np.array([sum(widths), -row_h / 2, 0]), color=GREY_C, stroke_width=1.5)
    group = VGroup(tab, sep)
    if highlight is not None:
        group.add(SurroundingRectangle(VGroup(*[tab[r][highlight] for r in range(len(tab))]), color=YELLOW,
                                       buff=0.06, corner_radius=0.06))
    return group


def sphere_row(items, height=2.3):
    """Row of point-distribution images (thesis figures) with captions."""
    return Group(*[captioned(plot_panel(ASSETS + f, height=height), cap, 18) for f, cap in items]).arrange(RIGHT, buff=0.35)


class BackupDirect(Base):
    def construct(self):
        self.header("Backup: direct computation methods (about 1000 points, 100 trials)")
        spheres = sphere_row([("dir_latlon.png", "Lat-Lon, $K=29$ (1070 points)"),
                              ("dir_fibonacci.png", "Fibonacci (1001 points)"),
                              ("dir_poisson.png", "Poisson disk (1012 points)")])
        head = ["", "Lat-Lon", "Fibonacci", "Poisson disk"]
        rows = [
            ["mean run time [s]", "$1.6\\times10^{-3}$", "$1.075\\times10^{-4}$", "$7.51\\times10^{-2}$"],
            ["std. deviation [s]", "$7.69\\times10^{-4}$", "$3.66\\times10^{-5}$", "$1.4\\times10^{-3}$"],
            ["$d_{NN,min}/d_{NN,avg}$", "0.9712", "0.9171", "0.0640"],
            ["$CV(d_{NN})$", "0.0068", "0.0105", "0.2533"],
            ["Riesz energy ($s=1$)", "553166", "483587", "515388"],
            ["number of points", "grid only", "odd $P$", "any"],
        ]
        tab = table_grid(head, rows, [3.6, 2.4, 2.4, 2.4], font_size=19, row_h=0.42, highlight=2)
        g = Group(spheres, tab).arrange(DOWN, buff=0.35).move_to(DOWN * 0.2)
        if g.height > 6.3:
            g.set(height=6.3)
        self.play(FadeIn(spheres))
        self.play(FadeIn(tab))
        self.next_slide()
        self.clear_body()
        self.new_title("Backup: direct computation methods, run time (100 trials)")
        box = captioned(plot_panel(ASSETS + "dir_runtime.png", height=5.4),
                        "Fibonacci $\\approx 700\\times$ faster than Poisson disk, with a smaller spread", 20)
        box.move_to(DOWN * 0.15)
        self.play(FadeIn(box))
        self.next_slide(auto_next=True)
        self.clear_all()


class BackupIterative(Base):
    def construct(self):
        self.header("Backup: iterative computation methods (30 points, 100 runs)")
        spheres = sphere_row([("it_thomson.png", "Thomson, minimum energy"),
                              ("it_convex.png", "Lloyd, convex hull"),
                              ("it_halfplane.png", "Lloyd, half-plane")])
        head = ["", "Thomson", "Lloyd (convex hull)", "Lloyd (half-plane)"]
        rows = [
            ["mean run time [s]", "0.085", "0.033", "1.367"],
            ["mean iterations to converge", "–", "20.47", "29.11"],
            ["converged / 100", "0", "100", "93"],
            ["$d_{NN,min}/d_{NN,avg}$", "0.3246", "0.8585", "0.8486"],
            ["$CV(d_{NN})$", "0.6950", "0.1202", "0.0728"],
            ["Riesz energy ($s=1$)", "461.79", "363.06", "360.77"],
        ]
        tab = table_grid(head, rows, [4.0, 2.2, 3.0, 3.0], font_size=19, row_h=0.42)
        g = Group(spheres, tab).arrange(DOWN, buff=0.35).move_to(DOWN * 0.2)
        if g.height > 6.3:
            g.set(height=6.3)
        self.play(FadeIn(spheres))
        self.play(FadeIn(tab))
        self.next_slide()
        self.clear_body()
        self.new_title("Backup: iterative computation methods, run time (100 runs)")
        box = captioned(plot_panel(ASSETS + "it_runtime.png", height=5.4),
                        "half-plane: most uniform, but about 40 times slower than the convex hull", 20)
        box.move_to(DOWN * 0.15)
        self.play(FadeIn(box))
        self.next_slide(auto_next=True)
        self.clear_all()


class BackupDecentralized(Base):
    def construct(self):
        self.header("Backup: convex-hull failure on the sphere")
        a = captioned(plot_panel(ASSETS + "convex_problem.png", height=5.2), "convex hull picks the short arc: wrong Voronoi cell")
        a.move_to(DOWN * 0.15)
        self.play(FadeIn(a))
        self.next_slide(auto_next=True)
        self.clear_all()


class BackupParameters(Base):
    def construct(self):
        self.header("Backup: decentralized simulation parameters")
        params = VGroup(
            Text("12 agents, 30 s, $\\Delta t=0.01$ s, $R=5$ m", font_size=26),
            Text("4 agents lost at 10 s, EMA $\\alpha=0.2$, noise $\\pm 5\\%$", font_size=26),
            Text("Bernstein degree 5, horizon 5", font_size=26),
            Text("$\\rho_{obs}=100,\\ \\rho_{ineq}=20,\\ w_g=1000,\\ w_s=1$", font_size=26),
            Text("$\\|\\mathbf{v}\\|\\le 3$ m/s, $\\|\\mathbf{a}\\|\\le 5$ m/s², $\\Theta=1$ m", font_size=26),
            Text("SQP $\\le 100$ iterations, $\\sigma=1.2$", font_size=26),
            Text("Voronoi update every 100 steps (1 Hz)", font_size=26),
            Text("Mellinger: $m=5$ kg, $K_p=150\\,I$, $K_v=15\\,I$", font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(DOWN * 0.15)
        self.play(FadeIn(params))
        self.next_slide(auto_next=True)
        self.clear_all()
