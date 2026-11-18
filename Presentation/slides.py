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
            legend_entry(ORANGE, "simulação do mesmo voo", font_size=20, radius=0.07),
            *[legend_entry(c, lab, font_size=20, radius=0.07) for c, lab in extra],
            Text(speed if isinstance(speed, str) else ("tempo real" if speed == 1 else f"velocidade {speed}x"), font_size=20, color=GREY_B),
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
    pos = B("Controlador de\nposição", (2.0, 2.4))
    att = B("Controlador de\natitude", (2.0, 0.9))
    ruav = B("Modelo do\nquadrirrotor", (5.0, 1.65), w=2.0)
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
    cons = B("Consenso", (xA, yC))
    ema = B("Média móvel\nexponencial (EMA)", (xA, yE))
    net = B("Rede espacial\n(Delaunay)", (2.0, yC))
    noisy = B("Medição com\nruído", (2.0, yE))
    g.add(cons, ema, net, noisy)
    g.add(
        W((XR, yC), port(net, RIGHT), label=r"\mathbf{p}_i"),
        W(port(net, LEFT), port(cons, RIGHT), label="L"),
        W((4.5, yE), port(noisy, RIGHT), label=r"\mathbf{p}_T", color=TARGET_COLOR),
        W(port(noisy, LEFT), port(ema, RIGHT), label=r"\hat{\mathbf{p}}_T", at=0.75),
        W(port(ema, UP), port(cons, DOWN), label=r"\hat{\mathbf{p}}_T", side=RIGHT),
    )
    groups = VGroup(
        arch_group([pos, att], "Controlador de Mellinger", BLUE_C),
        arch_group([cons, ema, net, noisy], "Estimação do alvo", ORANGE, label_at=DL),
    )

    def p_into(block, y, label=r"\mathbf{p}_i"):    # state from the left trunk into a block
        return VGroup(W((XL, y), (block[0].get_left()[0], y), label=label, at=0.45), J((XL, y)))

    def neighbours_into(*route):              # L from the spatial network, under the estimation group
        return VGroup(W((0.45, yC), (0.45, yL), *route, label="L", seg=1, side=DOWN, at=0.1), J((0.45, yC)))

    if kind in ("centralized", "decentralized"):
        # as in the code (swarm_geometric_Fibonacci_sim_target/main.py) and thesis eqs. (11), (14),
        # (15): the geometric controller moves q_i towards q_i*, the APF adjusts it, the radii
        # control is driven by the geometric error e_q, and p_d = p_T + r_i q_i
        xP, xC = -2.45, -0.45                    # APF and radii column, reference block
        geom = B("Controlador\ngeométrico", (xG, 2.4), w=2.0)
        apf = B("Salvaguarda\n(APF)", (xP, 2.4), w=1.8)
        radii = B("Controlador\nradial", (xP, 0.9), w=1.8)
        comb = VGroup(RoundedRectangle(corner_radius=0.14, width=1.15, height=0.85, stroke_color=ARCH_STROKE,
                                       stroke_width=2, fill_color=ARCH_FILL, fill_opacity=1),
                      MathTex(r"\hat{\mathbf{p}}_T+r_i\tilde{\mathbf{q}}_i", font_size=24, color=WHITE))
        comb[1].set(width=min(comb[1].width, 1.0))
        comb.move_to(np.array([xC, 2.4, 0]))
        src = (B("Rede de Fibonacci\n+ Munkres", (-5.2, yS)) if kind == "centralized"
               else B("Algoritmo de Lloyd\n(Voronoi na esfera)", (-5.2, yS), new=True))
        g.add(geom, apf, radii, comb, src)
        y_est = 0.0
        x_src = src[0].get_top()[0]
        g.add(
            W((x_src, src[0].get_top()[1]), (x_src, geom[0].get_bottom()[1]), label=r"\mathbf{q}_i^\star", side=LEFT, at=0.75),
            W(port(geom, DOWN, 0.55), (xG + 0.55, 0.9), port(radii, LEFT), label=r"\mathbf{e}_{q_i}", seg=1, side=DOWN),
            W(port(geom, RIGHT), port(apf, LEFT), label=r"\bar{\mathbf{q}}_i"),
            W(port(apf, RIGHT), port(comb, LEFT), label=r"\tilde{\mathbf{q}}_i"),
            W(port(radii, RIGHT), (xC - 0.3, 0.9), port(comb, DOWN, -0.3), label="r_i", seg=0),
            W(port(comb, RIGHT), port(pos, LEFT), label=r"\mathbf{p}_d"),
            W(port(cons, UP), (xA, y_est), (xC + 0.3, y_est), port(comb, DOWN, 0.3),
              label=r"\hat{\mathbf{p}}_T", seg=1, side=DOWN),
            W((xA, y_est), (-3.55, y_est), (-3.55, yE), port(ema, LEFT)), J((xA, y_est)),   # EMA uses the estimate
            W((XL, TOP), (XL, yS), tip=False),                                     # left trunk
            p_into(geom, 2.4),
        )
        if kind == "centralized":
            g.add(W((XL, yS), port(src, LEFT), label=r"\mathbf{p}_{1..N}", at=0.4))
            groups.add(arch_group([src], "Nó central", GREY_B, label_at=DL))
        else:
            g.add(p_into(src, yS), neighbours_into((-5.2, yL), port(src, DOWN)))
        groups.add(arch_group([geom, apf, radii, comb], "Planeamento de trajetória", RED_C))
    else:
        mpc = B("MPC\n(Bernstein)", (xG, 2.4), new=True)
        lloyd = B("Algoritmo de Lloyd\n(CVT geodésica\nna malha)", (xG, 0.85), h=1.05, new=True)
        mesh = B("Controlador\nde malha", (-1.2, 0.85), new=True)
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
        groups.add(arch_group([mpc, lloyd, mesh], "Planeamento de trajetória", RED_C))
    if kind != "centralized":
        leg = VGroup(RoundedRectangle(corner_radius=0.08, width=0.45, height=0.28, stroke_color=HIGHLIGHT,
                                      stroke_width=3, fill_color=ARCH_FILL, fill_opacity=1),
                     Text("novo neste capítulo", font_size=20, color=HIGHLIGHT)).arrange(RIGHT, buff=0.15)
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
        adviser = Text("Orientador: Prof. Bruno J. Guerreiro", font_size=24)
        degree = Text(
            "Mestrado em Engenharia Eletrotécnica e de Computadores  ·  NOVA FCT",
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
    SECTION = "Motivação"

    def construct(self):
        self.header("1. Motivação")

        def headline(logo, en, pt):
            txt = VGroup(Text(en, font_size=28), Text(pt, font_size=16, color=GREY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            img = ImageMobject(ASSETS + logo).scale(1.2)
            return Group(img, txt).arrange(RIGHT, buff=0.6).scale(0.8)

        h2 = headline(
            "logo_RTP.png",
            "Aeroporto de Bruxelas encerra espaço aéreo devido à presença de drones",
            "RTP",
        )
        h1 = headline(
            "logo_Publico.png",
            "Drones levam a encerramento de aeroportos na Bélgica e Suécia",
            "Público",
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
        chart_title = Text("Perturbações com drones em aeroportos alemães (DFS)", font_size=18, color=GREY_A)
        chart_title.next_to(axes, UP, buff=0.35)
        self.play(Create(axes), FadeIn(chart_title))
        self.play(*[GrowFromEdge(b, DOWN) for b in bars], FadeIn(labels), run_time=1.2)

        impact = bullet_list(
            [
                "Voos atrasados, desviados ou cancelados",
                "Heliportos hospitalares: emergências em risco",
                "Drones baratos e acessíveis",
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
            "Como circundar um drone que entra em espaço aéreo restrito\nde forma segura, visível e sem meios destrutivos?",
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
        arp_lab = Text("aeroporto", font_size=16).next_to(arp, RIGHT, buff=0.08)
        X = border.get_right()[0] + 0.45               # left edge of everything right of the map
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
            VGroup(Square(0.22, stroke_color=RED, fill_color=RED, fill_opacity=0.5), Text("Área proibida", font_size=17)).arrange(RIGHT, buff=0.15),
            VGroup(Square(0.22, stroke_color=ORANGE, fill_color=ORANGE, fill_opacity=0.35), Text("Área 1 (altura limitada)", font_size=17)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        key.next_to(col, DOWN, buff=0.45).align_to(col, LEFT)
        src = Text("Zonas de drones do aeroporto de Lisboa (LPPT), ANAC · mapa © contribuidores OpenStreetMap", font_size=13, color=GREY_B)
        fit(src, 6.9 - X).move_to(np.array([X, border.get_bottom()[1] + 0.08, 0]), aligned_edge=LEFT)

        self.play(FadeIn(city), FadeIn(border), FadeIn(src), run_time=0.8)
        self.play(FadeIn(area1), FadeIn(red), FadeIn(arp), FadeIn(arp_lab), FadeIn(key), run_time=1.0)
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

        cm_title = Text("Contramedidas habituais", font_size=22, color=HIGHLIGHT)
        cite = Text("Lykou et al. (2020)", font_size=13, color=GREY_B)
        head = VGroup(cm_title, cite).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        cms = VGroup(
            counter("Interferência (jamming / spoofing)", "o drone cai ou foge descontrolado;\ninterfere com ajudas à aterragem e rádio"),
            counter("Interceção: redes, drones intercetores", "captura ou colisão sobre a cidade:\nqueda do drone e de destroços"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        cm = fit(VGroup(head, cms).arrange(DOWN, aligned_edge=LEFT, buff=0.25), W)
        cm.move_to(np.array([RX, 1.4, 0]), aligned_edge=UL)

        self.play(FadeIn(head), run_time=0.4)
        for row in cms:
            self.play(FadeIn(row, shift=LEFT * 0.2), run_time=0.6)


# =============================================================================
# 2. Problem statement
# =============================================================================
class Problem(Base):
    SECTION = "Problema"

    def construct(self):
        self.header("2. Definição do problema")

        # ---- three notions of "encirclement": left out of the talk for time (uncomment to restore) --
        # # ---- three notions of "encirclement" ---------------------------------
        # def panel(title, highlight=False):
        #     rect = RoundedRectangle(corner_radius=0.15, width=3.9, height=3.6, color=YELLOW if highlight else GREY_C)
        #     t = Text(title, font_size=22, color=YELLOW if highlight else WHITE)
        #     if t.width > rect.width - 0.3:
        #         t.set(width=rect.width - 0.3)
        #     t.next_to(rect.get_top(), DOWN, buff=0.2)
        #     return VGroup(rect, t)

        # p1 = panel("Convex containment")
        # p2 = panel("Reachable-set overlap")
        # p3 = panel("Visible formation (this thesis)", highlight=True)
        # panels = VGroup(p1, p2, p3).arrange(RIGHT, buff=0.35).shift(DOWN * 0.15)

        # c1 = p1[0].get_center() + DOWN * 0.25
        # tri = [c1 + 1.2 * np.array([np.cos(a), np.sin(a), 0]) for a in (PI / 2 + 0.3, PI + 0.9, -0.4)]
        # poly = Polygon(*tri, color=GREY_B, fill_color=GREY_B, fill_opacity=0.15)
        # d1 = VGroup(poly, *[Dot(p, color=WHITE) for p in tri], Dot(c1 + np.array([0.1, -0.1, 0]), color=RED))

        # c2 = p2[0].get_center() + DOWN * 0.25
        # reach_t = Circle(radius=0.85, color=RED, fill_opacity=0.15).move_to(c2)
        # reach_c = VGroup(
        #     *[
        #         Circle(radius=0.55, color=WHITE, fill_opacity=0.08, stroke_width=2).move_to(c2 + 1.0 * np.array([np.cos(a), np.sin(a), 0]))
        #         for a in np.linspace(0, 2 * PI, 5, endpoint=False)
        #     ]
        # )
        # d2 = VGroup(reach_t, reach_c, Dot(c2, color=RED), *[Dot(c.get_center(), color=WHITE, radius=0.05) for c in reach_c])

        # c3 = p3[0].get_center() + DOWN * 0.25
        # ring = Circle(radius=1.1, color=BLUE, stroke_opacity=0.5).move_to(c3)
        # d3 = VGroup(ring, Dot(c3, color=RED), *[Dot(c3 + 1.1 * np.array([np.cos(a), np.sin(a), 0]), color=WHITE) for a in np.linspace(0, 2 * PI, 8, endpoint=False)])

        # self.play(LaggedStart(*[Create(p) for p in panels], lag_ratio=0.2))
        # self.play(FadeIn(d1), FadeIn(d2), FadeIn(d3))
        # note = Text(
        #     "Goal: a visible barrier that warns the operator, not physical capture",
        #     font_size=24,
        #     color=GREY_A,
        # ).next_to(panels, DOWN, buff=0.35)
        # self.play(FadeIn(note))
        # self.next_slide()
        # self.play(FadeOut(VGroup(panels, d1, d2, d3, note)))


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
            entry(RED, "UAV alvo", r"\mathbf{p}_T", "(não cooperativo)"),
            entry(WHITE, "UAV perseguidores", r"\mathbf{p}_i"),
            entry(BLUE, "pontos desejados", r"\mathbf{p}_i^\star"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.move_to(RIGHT * 0.6 + UP * 2.3, aligned_edge=LEFT + UP)
        self.play(FadeIn(world), FadeIn(legend))
        self.play(s.animate.set_value(1.0), run_time=2, rate_func=smooth)

        objective = MathTex(r"\lim_{t\to\infty}\ \mathbf{p}_i(t)=\mathbf{p}_i^\star,\quad \forall i", font_size=40)
        objective.next_to(legend, DOWN, buff=0.6).align_to(legend, LEFT)
        self.play(Write(objective))
        self.next_slide(auto_next=True)
        world.clear_updaters()
        self.clear_all()


# =============================================================================
# 3. Roadmap / contributions
# =============================================================================
def roadmap_grid(font_size=19):
    # the sphere-only decentralized stage (thesis Ch. 5) is left out of the talk: the generalized
    # stage (Ch. 6), decentralized as well, is presented as "Decentralized"
    cols = ["Centralizado", "Descentralizado"]
    rows = ["Para onde ir?\n(distribuição)", "Como lá chegar?\n(planeamento)", "Em segurança?\n(colisões)"]
    cells = [
        ["Rede de Fibonacci\n+ atribuição de Munkres", "Lloyd + Dijkstra em qualquer malha\n+ ponderação pela velocidade"],
        ["Controlador geométrico\nde Lyapunov + radial", "MPC com polinómios de Bernstein\n(limites de v, a)"],
        ["Salvaguarda APF\nna esfera", "Custo de colisão no MPC\n(envelope elipsoidal)"],
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
        r"Ambas: estimativa do alvo = medição com ruído ($\pm 5\%$) → EMA → consenso numa rede espacial de Delaunay",
        font_size=18,
        color=GREY_A,
    )
    est.move_to(np.array([-0.6, 1.25 - 3 * (H + 0.15) + 0.15, 0]))
    grid.add(head, row_labels, body, est)
    return grid


class Roadmap(Base):
    SECTION = "Problema"

    def construct(self):
        self.header("3. Solução proposta: duas etapas")
        g = roadmap_grid()
        head, row_labels, body, est = g
        self.play(FadeIn(row_labels))
        for j in range(len(head)):
            self.play(FadeIn(head[j]), *[FadeIn(body[i][j], shift=DOWN * 0.1) for i in range(3)], run_time=0.8)
        self.play(FadeIn(est))
        self.next_slide(auto_next=True)
        self.clear_all()


# ---- left out of the talk (also removed from scenes.txt); uncomment to restore ----
# # =============================================================================
# # 4. State of the art
# # =============================================================================
# class StateOfTheArt(Base):
#     SECTION = "Problema"
#
#     def construct(self):
#         self.header("4. Estado da arte (formações circulares)")
#         centers = [LEFT * 4.4 + UP * 0.9, UP * 0.9, RIGHT * 4.4 + UP * 0.9]
#         names = ["Perseguição cíclica", "Líder-seguidor", "Campos de potencial (APF)"]
#         pros = [
#             "+ comunicação mínima,\n   sem líder",
#             "+ intuitivo, geometria\n   da formação explícita",
#             "+ projeto baseado em Lyapunov,\n   desvio de obstáculos",
#         ]
#         cons = [
#             "- pode colapsar num ponto\n   sem termos adicionais",
#             "- líder = ponto único\n   de falha",
#             "- mínimos locais, afinação;\n   N tem de ser conhecido",
#         ]
#         t = ValueTracker(0.0)
#
#         # cyclic pursuit
#         def cyc():
#             c = centers[0]
#             n = 5
#             a = t.get_value()
#             pts = [c + 1.0 * np.array([np.cos(a + 2 * PI * k / n), np.sin(a + 2 * PI * k / n), 0]) for k in range(n)]
#             g = VGroup(Dot(c, color=RED))
#             for k in range(n):
#                 g.add(Arrow(pts[k], pts[(k + 1) % n], buff=0.12, stroke_width=2.5, color=GREY_B, max_tip_length_to_length_ratio=0.2))
#                 g.add(Dot(pts[k], color=WHITE))
#             return g
#
#         # leader-follower
#         def lf():
#             c = centers[1] + np.array([0.5 * np.sin(t.get_value()), 0.25 * np.sin(2 * t.get_value()), 0])
#             g = VGroup()
#             lead = Dot(c, color=YELLOW, radius=0.1)
#             for k in range(6):
#                 a = 2 * PI * k / 6 + 0.5 * t.get_value()
#                 p = c + 0.95 * np.array([np.cos(a), np.sin(a), 0])
#                 g.add(Line(c, p, color=GREY_D, stroke_width=1.5), Dot(p, color=WHITE))
#             g.add(lead)
#             return g
#
#         # APF: vector field converging to a ring
#         def apf():
#             c = centers[2]
#             g = VGroup(Circle(radius=0.9, color=BLUE, stroke_opacity=0.6).move_to(c), Dot(c, color=RED))
#             for x in np.linspace(-1.5, 1.5, 7):
#                 for y in np.linspace(-1.2, 1.2, 6):
#                     p = np.array([x, y, 0])
#                     r = np.linalg.norm(p)
#                     if r < 0.15:
#                         continue
#                     radial = (0.9 - r) * p / r
#                     tang = 0.4 * np.array([-p[1], p[0], 0]) / r
#                     v = radial + tang
#                     v = 0.28 * v / (np.linalg.norm(v) + 1e-9)
#                     g.add(Arrow(c + p, c + p + v, buff=0, stroke_width=2, color=GREY_B, max_tip_length_to_length_ratio=0.35))
#             return g
#
#         demos = [cyc(), lf(), apf()]   # static diagrams
#         texts = VGroup()
#         for k in range(3):
#             n = Text(names[k], font_size=26, color=YELLOW).move_to(centers[k] + UP * 1.75)
#             pr = Text(pros[k], font_size=19, color=GREEN_C, line_spacing=0.8)
#             co = Text(cons[k], font_size=19, color=RED_C, line_spacing=0.8)
#             pc = VGroup(pr, co).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(centers[k], DOWN, buff=1.5)
#             texts.add(VGroup(n, pc))
#         for k in range(3):
#             self.play(FadeIn(demos[k]), FadeIn(texts[k]), run_time=1.0)
#
#         gap = Text(
#             "Lacuna: a maioria dos métodos é planar (2D); os 3D usam setores predefinidos",
#             font_size=24,
#             color=YELLOW,
#         ).to_edge(DOWN, buff=0.65)
#         self.play(FadeIn(gap), run_time=1.0)
#         self.next_slide()
#         self.clear_all()


# =============================================================================
# 5. Background
# =============================================================================
class Background(Base):
    SECTION = "Fundamentos"

    def construct(self):
        self.header("5. Fundamentos: algoritmo de Lloyd e diagramas de Voronoi")
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

        vor_def = MathTex(
            r"V_i=\{x : d(x,\mathbf{s}_i)\le d(x,\mathbf{s}_j)\ \forall j\neq i\}",
            font_size=36,
        ).next_to(border, RIGHT, buff=0.5).align_to(border, UP).shift(DOWN * 0.3)

        self.play(Create(border), FadeIn(site_dots))
        self.play(FadeIn(cells_mob), Write(vor_def))
        self.bring_to_front(site_dots)
        it_label = Text("iteração 0", font_size=20, color=GREY_A).next_to(border, DOWN, buff=0.15)
        self.add(it_label)
        for it in range(1, 9):
            crosses = VGroup(*[Cross(scale_factor=0.08, stroke_width=3, stroke_color=YELLOW).move_to(to_screen(c)) for c in cents])
            rt = 0.9 if it <= 2 else 0.45
            if it <= 2:
                self.play(FadeIn(crosses), run_time=0.4)
            sites = cents
            new_cells, new_cents = draw_cells(sites)
            new_dots = VGroup(*[Dot(to_screen(s), color=WHITE, radius=0.07) for s in sites])
            new_label = Text(f"iteração {it}", font_size=20, color=GREY_A).next_to(border, DOWN, buff=0.15)
            anims = [Transform(site_dots, new_dots), Transform(cells_mob, new_cells), Transform(it_label, new_label)]
            if it <= 2:
                anims.append(FadeOut(crosses))
            self.play(*anims, run_time=rt)
            self.bring_to_front(site_dots)
            cents = new_cents
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
        dual_txt = Text("Triangulação de Delaunay", font_size=24, color=YELLOW).next_to(vor_def, DOWN, buff=0.6).align_to(vor_def, LEFT)
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
            Text("Consenso na estimativa do alvo:", font_size=22),
            MathTex(r"\hat{\mathbf{p}}_{T,i}\leftarrow \hat{\mathbf{p}}_{T,i}+\varepsilon\sum_{j\in\mathcal{N}_i}(\hat{\mathbf{p}}_{T,j}-\hat{\mathbf{p}}_{T,i})", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        cons_txt.next_to(dual_txt, DOWN, buff=0.5).align_to(vor_def, LEFT)
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
        self.new_title("5. Fundamentos: métodos de prevenção de colisões")
        header = ["Método", "Restringe", "Predição", "Segurança formal", "Na tese"]
        rows = [
            ["APF", "referência (potencial)", "estado atual", "sem invariância", "Cap. 4-5"],
            ["CBF-QP", "entrada de controlo", "estado atual", "invariância positiva", "não usado"],
            ["MPC", "posições previstas + v, a", "horiz. finito", "restrições/penalizações", "Cap. 6"],
            ["CBF-MPC", "trajetória + barreira", "horiz. finito", "segurança MPC + CBF", "trabalho futuro"],
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
        cbf_t = Text("CBF: mantém o conjunto seguro positivamente invariante", font_size=22, color=GREY_A)
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
    SECTION = "Centralizado"

    def construct(self):
        self.header("6. Centralizado: para onde deve ir cada agente?")
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

        title = Text("Rede de Fibonacci", font_size=30, color=YELLOW)
        eq = MathTex(
            r"\text{lat}_i=\arcsin\!\left(\tfrac{2i}{2n+1}\right),\quad \text{lon}_i=2\pi i\,\varphi^{-1}",
            font_size=34,
        )
        eq2 = MathTex(r"i=-n,\dots,n,\qquad P=2n+1,\qquad \varphi=\tfrac{1+\sqrt5}{2}", font_size=30)
        golden = MathTex(r"360^\circ(1-\varphi^{-1})\simeq 137.5^\circ\ \text{(ângulo de ouro)}", font_size=30)
        right = VGroup(title, eq, eq2, golden).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(UP * 1.75).to_edge(RIGHT, buff=0.45)

        self.play(FadeIn(sphere), FadeIn(counter))
        self.play(FadeIn(title), Write(eq), FadeIn(eq2), run_time=1.2)
        self.play(n.animate.set_value(401), rot.animate.set_value(20 * DEGREES), run_time=5, rate_func=rate_functions.ease_in_quad)
        self.play(FadeIn(golden))
        self.next_slide()

        # comparison table (thesis appendix, ~1000 points)
        head = ["", "Lat-Lon", "Fibonacci", "Disco de Poisson"]
        rows = [
            ["tempo médio", "$1.6\\times10^{-3}$ s", "$1.075\\times10^{-4}$ s", "$7.51\\times10^{-2}$ s"],
            ["$CV(d_{NN})$", "0.0068", "0.0105", "0.2533"],
            ["Energia de Riesz", "553 166", "483 587", "515 388"],
            ["qualquer $P$?", "só em grelha", "$P$ ímpar", "sim"],
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
        self.play(FadeIn(tab), n.animate.set_value(13), run_time=1.2)
        self.play(Create(box_f))
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
    SECTION = "Centralizado"

    def construct(self):
        self.header("7. Centralizado: controlo geométrico baseado em Lyapunov")

        decomp = MathTex(r"\mathbf{p}_i-\mathbf{p}_T", r"=", r"r_i", r"\,\mathbf{q}_i", r",\qquad \mathbf{q}_i\in\mathbb{S}^2", font_size=40)
        decomp[2].set_color(GREEN_C)
        decomp[3].set_color(YELLOW)
        goals = MathTex(r"r_i\to r^\star", r"\qquad", r"\mathbf{q}_i\to\mathbf{q}_i^\star", font_size=40)
        goals[0].set_color(GREEN_C)
        goals[2].set_color(YELLOW)
        top = VGroup(decomp, goals).arrange(DOWN, buff=0.3).move_to(UP * 2.3)
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
        self.play(Write(goals))
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
            MathTex(r"\mathbf{e}_r=S(\mathbf{q}^\star)\,\mathbf{q}=\mathbf{q}^\star\times\mathbf{q}", font_size=30, color=GREY_B),
            MathTex(r"\|\mathbf{e}_r\|=\sin\theta", font_size=30, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        lab_p = VGroup(
            Text("erro proposto [Ramp 2015]", font_size=20, color=YELLOW),
            MathTex(r"\mathbf{e}_q=\frac{S(\mathbf{q}^\star)\,\mathbf{q}}{\sqrt{2(1+\mathbf{q}^\top\mathbf{q}^\star)}}", font_size=30, color=YELLOW),
            MathTex(r"\|\mathbf{e}_q\|=\sin\tfrac{\theta}{2}", font_size=30, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        mono = Text(r"monótono em $\theta$: sem falsa convergência", font_size=20, color=YELLOW)
        # right column: classical error, its problem, proposed error, its property
        labs = VGroup(lab_c, lab_p, mono).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        labs.next_to(axes, RIGHT, buff=0.4).align_to(axes, UP).shift(UP * 0.75)
        if labs.get_bottom()[1] < -3.25:
            labs.set(height=labs.height - (-3.25 - labs.get_bottom()[1]))
            labs.next_to(axes, RIGHT, buff=0.4).align_to(axes, UP).shift(UP * 0.75)
        dot_bad = Dot(axes.c2p(PI * 0.97, np.sin(PI * 0.97)), color=RED_C)
        self.play(Create(axes), FadeIn(ticks), FadeIn(xl))
        self.play(Create(classic), FadeIn(lab_c))
        self.play(FadeIn(dot_bad))
        self.play(Create(proposed), FadeIn(lab_p))
        self.play(FadeIn(mono))
        self.next_slide()
        self.play(FadeOut(VGroup(axes, xl, ticks, classic, proposed, labs, dot_bad, top)))

        # --- control laws + theorem ------------------------------------------
        law1 = MathTex(r"\dot{\mathbf{q}}_i=-k_{geo}\,S(\mathbf{e}_{q_i})\,\mathbf{q}_i", font_size=36)
        law2 = MathTex(r"\dot r_i=k_r\,e^{-\|\mathbf{e}_{q_i}\|^2}\,(r^\star-r_i)", font_size=36)
        thm = VGroup(
            Text("Teorema (Cap. 4 da tese)", font_size=22, color=YELLOW),
            MathTex(r"V=\Psi(\mathbf{q},\mathbf{q}^\star),\quad \dot V=-k_{geo}\|\mathbf{e}_q\|^2\le 0", font_size=30),
            Text(r"$(\mathbf{q}^\star, r^\star)$ assintoticamente estável em", font_size=20),
            MathTex(r"(\mathbb{S}^2\setminus\{-\mathbf{q}^\star\})\times\mathbb{R}_{>0}", font_size=30),
            Text(r"erro radial: exponencial, taxa $\ge k_r/e$", font_size=20),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        thm_box = SurroundingRectangle(thm, color=YELLOW, buff=0.2, corner_radius=0.1)
        laws = VGroup(law1, law2, VGroup(thm_box, thm)).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
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
        self.play(Write(law2))
        self.play(Create(thm_box), FadeIn(thm))
        self.play(k.animate.set_value(Tn - 1), run_time=6, rate_func=linear)
        self.next_slide(auto_next=True)
        w.clear_updaters()
        self.clear_all()


# =============================================================================
# 8. Centralized - safety, estimation, results
# =============================================================================
class CentralizedResults(Base):
    SECTION = "Centralizado"

    def construct(self):
        self.header("8. Centralizado: salvaguarda de colisões e resultados")

        # top views of the APF test (thesis Ch. 4): without and with the safeguard
        a1 = captioned(plot_panel(ASSETS + "apf_top_inactive.png", height=4.6), "APF inativo")
        a2 = captioned(plot_panel(ASSETS + "apf_top_active.png", height=4.6), "APF ativo")
        apf_imgs = Group(a1, a2).arrange(RIGHT, buff=0.35)
        apf_title = Text("Salvaguarda APF na esfera", font_size=26, color=YELLOW)
        f = MathTex(
            r"\mathbf{f}_{ij}=(d_{min}-d_{ij})\frac{\Delta_{ij}^t}{\|\Delta_{ij}^t\|}",
            font_size=34,
        )
        apf_txt = VGroup(apf_title, f).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        other = Text(
            "+ atribuição de Munkres dos pontos da rede (custo geodésico total mínimo)",
            font_size=21,
            color=GREY_A,
        )
        top_row = Group(apf_imgs, apf_txt).arrange(RIGHT, buff=0.5)
        if top_row.width > 13.4:                          # keep the row inside the frame
            top_row.set(width=13.4)
        Group(top_row, other).arrange(DOWN, buff=0.35).move_to(np.array([0, (3.1 - 3.45) / 2, 0]))
        self.play(FadeIn(a1), FadeIn(apf_title))
        self.play(FadeIn(a2))
        self.play(Write(f))
        self.play(FadeIn(other))
        self.next_slide()
        self.clear_body()

        # --- complete architecture (Ch. 4 sub-diagrams merged) ----------------
        self.architecture("8. Centralizado: arquitetura completa", "centralized")
        self.next_slide()
        self.clear_body()

        # --- results 31 agents, moving target ------------------------------
        self.new_title("8. Centralizado: 31 agentes, alvo móvel")
        # fill the body (between the title and the section bar): the stacked pair sets the height
        TOP, BOT, CAP = 3.1, -3.45, 0.42                  # body limits, caption + its gap
        r2, r3 = 1580 / 980, 1780 / 980                   # aspect ratios of the two right plots
        W = (TOP - BOT - 2 * CAP - 0.2 - 0.24) / (1 / r2 + 1 / r3)
        p1 = captioned(plot_panel(ASSETS + "c31dyn_trajectory.png", height=TOP - BOT - CAP - 0.12), "trajetórias")
        p2 = captioned(plot_panel(ASSETS + "c31dyn_radial_hist.png", width=W), r"distância ao alvo ($r^\star = 2$ m)")
        p3 = captioned(plot_panel(ASSETS + "c31dyn_inter_agent_dist.png", width=W), "distância entre agentes")
        right = Group(p2, p3).arrange(DOWN, buff=0.2)
        layout = Group(p1, right).arrange(RIGHT, buff=0.4, aligned_edge=UP)
        layout.move_to(np.array([0, (TOP + BOT) / 2, 0]))
        self.play(FadeIn(p1))
        self.play(FadeIn(right))
        self.next_slide()
        self.clear_body()

        # --- real-world test -------------------------------------------------
        self.new_title("8. Teste real: rede de Fibonacci + controlador geométrico")
        self.flight(
            "flight_geometric.mp4",
            [
                "3 Crazyflies, alvo virtual numa circunferência (raio 1.5 m, 0.25 m/s), raio da formação 0.8 m",
                "trajetórias reais vs simuladas: cerca de 10-11 cm RMS na fase principal",
            ],
            speed="fase principal em tempo real, o resto a 3x",
            extra=[("#E63CE6", "alvo (simulado)")],
        )
        self.next_slide(auto_next=True)
        self.clear_all()


# =============================================================================
# 9. Decentralized
# =============================================================================
class Decentralized(Base):
    SECTION = "Descentralizado"

    def construct(self):
        self.header("9. Descentralizado: Lloyd local na esfera")
        why = VGroup(
            Text("A rede de Fibonacci precisa do número total de agentes", font_size=22),
            Text("A atribuição de Munkres precisa da posição de todos os agentes", font_size=22),
            Text("→ cada agente: a sua célula de Voronoi a partir dos vizinhos → mover para o centroide", font_size=22, color=YELLOW),
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
        n_label = VGroup(Text("agentes:", font_size=22), Integer(12, font_size=32)).arrange(RIGHT, buff=0.15)
        info = VGroup(n_label, cv_label).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        info.next_to(outline, RIGHT, buff=0.35).align_to(outline, UP)
        illu = Text("reimplementação ilustrativa", font_size=16, color=GREY_B).next_to(outline, DOWN, buff=0.08)
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
        lost = Text("4 agentes perdidos", font_size=22, color=RED).next_to(info, DOWN, buff=0.3).align_to(info, LEFT)
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
        head = ["", "Inv. convexo", "Semiplano"]
        rows = [
            ["$CV(d_{NN})$", "0.120", "0.073"],
            ["$d_{NN,min}/d_{NN,avg}$", "0.859", "0.849"],
            ["tempo médio", "0.033 s", "1.367 s"],
            ["convergiu / 100", "100", "93"],
            ["config. degeneradas", "possíveis", "não"],
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
        tab_t = Text("30 agentes, 100 execuções (apêndice da tese)", font_size=18, color=GREY_B).next_to(tab, DOWN, buff=0.2)
        good = [tab[1][2], tab[5][2]]
        bad = [tab[3][2], tab[4][2]]
        self.play(FadeIn(tab), FadeIn(tab_t))
        self.play(*[t.animate.set_color(GREEN_C) for t in good], *[t.animate.set_color(RED_C) for t in bad])
        self.next_slide()
        self.clear_body()

        # --- complete architecture -------------------------------------------
        self.architecture("9. Descentralizado: arquitetura completa", "decentralized")
        self.next_slide()
        self.clear_body()

        # --- results ---------------------------------------------------------
        self.new_title("9. Descentralizado: alvo móvel, perda de 4 agentes (semiplano)")
        p1 = captioned(plot_panel(ASSETS + "hp_ddrop_radial_hist.png", height=3.05), r"distância ao alvo ($r^\star = 2$ m)")
        p2 = captioned(plot_panel(ASSETS + "hp_ddrop_qd_separation.png", height=3.05), "espaçamento dos pontos desejados")
        p3 = captioned(plot_panel(ASSETS + "hp_ddrop_inter_agent_dist.png", height=3.05), "distância entre agentes")
        row = Group(p1, p2, p3).arrange(RIGHT, buff=0.3).move_to(UP * 0.75)
        if row.width > 13.2:
            row.set(width=13.2)
        facts = VGroup(
            Text(r"alvo móvel, raio desejado $r^\star = 2$ m, $k_{geo}=100,\ k_r=0.7$", font_size=20, color=GREY_A),
            Text("aproximação mais lenta do que com o alvo estático: distância média ainda ~4.3 m aos 30 s", font_size=22),
            Text(r"os pontos desejados mantêm-se uniformes: $CV(d_{NN})\lesssim 0.1$, $d_{NN}$ 2.0 → 2.2 m após a perda", font_size=22),
            Text("distância mín. ~1.7 m na perda, sem quase-colisões", font_size=22),
            Text("não é preciso reprojetar quando o tamanho da formação muda", font_size=22, color=YELLOW),
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
    SECTION = "Descentralizado"

    def construct(self):
        self.header("9. Descentralizado: qualquer superfície, barreira ou envolvimento")
        W = (13.7 - 0.35) / 2                             # two figures across the whole width
        b = captioned(plot_panel(ASSETS + "dijkstra_bust.png", width=W), "malha de busto, 8 sítios")
        f = captioned(plot_panel(ASSETS + "dijkstra_frog.png", width=W), "malha de sapo, 8 sítios")
        imgs = Group(b, f).arrange(RIGHT, buff=0.35)
        head = Text("Lloyd", font_size=32, color=YELLOW)
        Group(head, imgs).arrange(DOWN, buff=0.25).move_to(np.array([0, (3.1 - 3.45) / 2, 0]))
        self.play(FadeIn(head), FadeIn(imgs))
        self.next_slide()
        self.clear_body()

        # --- mesh controller animation ---------------------------------------
        self.new_title("9. Descentralizado: controlador de malha")
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
        side_in = Text("espaço aéreo restrito", font_size=22, color=RED_C).next_to(n_arrow, RIGHT, buff=0.1)
        side_out = Text("exterior", font_size=22, color=GREY_A).next_to(plane, LEFT, buff=0.15).align_to(side_in, DOWN)

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
                s = "disco plano (barreira)"
            elif u2 <= 0:
                s = "disco → cone"
            elif u2 < 1:
                s = "cone → esfera"
            else:
                s = "esfera fechada (envolvimento)"
            t = VGroup(
                Text(s, font_size=26, color=YELLOW),
                MathTex(rf"i_1={u1:.2f},\quad i_2={u2:.2f}", font_size=30),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            t.move_to(RIGHT * 1.6 + UP * 2.2, aligned_edge=LEFT)
            return t

        st = always_redraw(stage_text)
        formula = MathTex(
            r"\mathbf{V}=\;&(1-i_2)\big[(1-i_1)\mathbf{V}_{\mathrm{disc}}+i_1\mathbf{V}_{\mathrm{cone}}\big]\\&+i_2\mathbf{V}_{\mathrm{sphere}}",
            font_size=40,
        )
        d_txt = Tex(r"$i_1, i_2$: obtidos da distância com sinal do alvo\\à fronteira, normalizada pelo raio da malha $R$", font_size=30, color=GREY_A)
        rgt = VGroup(formula, d_txt).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        if rgt.width > 5.2:
            rgt.set(width=5.2)
        rgt.move_to(RIGHT * 1.6 + UP * 1.0, aligned_edge=LEFT + UP)   # below the i_1, i_2 readout
        self.play(FadeIn(plane), FadeIn(n_arrow), FadeIn(side_out), FadeIn(side_in), FadeIn(w), FadeIn(st))
        self.play(FadeIn(formula), FadeIn(d_txt))
        self.next_slide()
        # target approaches and crosses the boundary: disc -> cone (complete at d = R/2)
        self.play(x_t.animate.set_value(R / 2), run_time=4.0, rate_func=linear)
        self.next_slide()
        # past the boundary: cone -> sphere (complete at d = R), then the sphere follows the target
        self.play(x_t.animate.set_value(R), run_time=2.5, rate_func=linear)
        self.play(x_t.animate.set_value(8.0), run_time=1.5, rate_func=linear)
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
    SECTION = "Descentralizado"

    def construct(self):
        self.header("10. Descentralizado: MPC com prevenção de colisões")
        cost = MathTex(
            r"\min_{\mathbf{c}}\ w_g\sum_{k}\|\mathbf{p}_i[k]-\mathbf{p}_i^\star\|^2+w_s\sum_k\|\ddot{\mathbf{p}}_i[k]\|^2",
            font_size=32,
        )
        cons = MathTex(
            r"\text{s.a.}\ \ \mathbf{p}_i^{(l)}[0]=\mathbf{p}_i^{(l)},\ \ \|\dot{\mathbf{p}}_i[k]\|\le\bar v,\ \ \|\ddot{\mathbf{p}}_i[k]\|\le\bar a",
            font_size=30,
        )
        col = MathTex(
            r"h_{ij}[k]=\big\|\Theta_{ij}^{-1}(\mathbf{p}_i[k]-\mathbf{p}_j[k])\big\|^2-1\ge 0",
            font_size=30,
            color=ORANGE,
        )
        eqs = VGroup(cost, cons, col).arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(UP, buff=0.9).shift(RIGHT * 0.2)

        # 2D illustration: Bezier (degree 5) from agent to goal, avoiding a neighbour
        base = np.array([-3.0, -1.6, 0])   # centred: agent at x=-3, goal at x=+3
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
        nb_mid = (nb_start + nb_end) / 2
        nb = Dot(nb_mid, color=WHITE, radius=0.1)
        ell = Ellipse(width=2.2, height=1.6, color=ORANGE).move_to(nb_mid)
        nb_dir = (nb_end - nb_start) / np.linalg.norm(nb_end - nb_start)
        nb_arrow = Arrow(nb_mid, nb_mid + 0.85 * nb_dir, buff=0.12, color=GREY_A, stroke_width=4,
                         max_tip_length_to_length_ratio=0.35)
        nb_lab = MathTex(r"\mathbf{p}_j[k]", font_size=30).next_to(nb, LEFT, buff=0.15).shift(UP * 0.3)   # clear of the trajectory
        ell_lab = MathTex(r"\Theta_{ij}", font_size=28, color=ORANGE).next_to(ell, RIGHT, buff=0.08).shift(UP * 0.4)
        nb_goal = nb_end + 0.25 * nb_dir                  # the neighbour's own goal, beyond where it ends up
        gj_dot = Dot(nb_goal, color=BLUE, radius=0.1)
        gj_lab = MathTex(r"\mathbf{p}_j^\star", font_size=30, color=BLUE).next_to(gj_dot, RIGHT, buff=0.12)

        curve = bez(straight)
        cp = ctrl_poly(straight)
        hz = horizon(straight)
        # the equations and the whole illustration are on screen from the start ...
        self.play(Write(cost), FadeIn(cons), FadeIn(col),
                  FadeIn(VGroup(a_dot, a_lab, g_dot, g_lab, gj_dot, gj_lab, nb, nb_lab, nb_arrow, curve, cp, hz, ell, ell_lab)))
        self.next_slide()   # ... and the motion starts on the next key press
        self.play(Transform(curve, bez(bent)), Transform(cp, ctrl_poly(bent)), Transform(hz, horizon(bent)), run_time=1.5)

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
        self.new_title("10. Descentralizado: usar a velocidade do alvo")
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

        cap = Text("velocidade do alvo desconhecida: CVT uniforme", font_size=22, color=GREY_A)
        cap.move_to(np.array([-6.6, 2.55, 0]), aligned_edge=LEFT)
        w = MathTex(r"w_i=\exp\!\Big(\frac{\theta_i^2}{2\sigma^2}\Big),\ \ \sigma=1.2", font_size=32)
        wt = Text("$\\theta_i$: ângulo desde a direção\natrás do alvo\n→ mais peso à frente", font_size=20, color=GREY_A, line_spacing=0.8)
        st = stats(uni)
        st_lab = VGroup(
            Text("alinhamento direcional médio\n(≈ 0 para uma formação simétrica)", font_size=18, color=GREY_A, line_spacing=0.8),
            Text("agentes à frente do alvo", font_size=18, color=GREY_A),
        )
        it_lab = Text("iteração 0", font_size=22, color=GREY_B)
        rcol_x = 1.25

        # right column: A + label, F_fwd + label (top, shown from the start), then the
        # iteration counter and the weight (bottom, filled in once the velocity is known)
        for m in st_lab:
            if m.get_right()[0] - m.get_left()[0] > 6.9 - rcol_x:
                m.set(width=6.9 - rcol_x)
        A_Y = 2.1

        def place_stats(s):
            s[0].move_to(np.array([rcol_x, A_Y, 0]), aligned_edge=LEFT)
            s[1].next_to(st_lab[0], DOWN, buff=0.3).align_to(s[0], LEFT)
            return s

        st[0].move_to(np.array([rcol_x, A_Y, 0]), aligned_edge=LEFT)
        st_lab[0].next_to(st[0], DOWN, buff=0.1).align_to(st[0], LEFT)
        place_stats(st)
        st_lab[1].next_to(st[1], DOWN, buff=0.1).align_to(st[0], LEFT)
        it_lab.next_to(st_lab[1], DOWN, buff=0.45).align_to(st[0], LEFT)
        VGroup(w, wt).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(it_lab, DOWN, buff=0.4).align_to(st[0], LEFT)

        self.play(FadeIn(sil), FadeIn(divide), FadeIn(tgt), FadeIn(cap))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in agent_g], lag_ratio=0.05), run_time=1.0)
        self.play(FadeIn(st), FadeIn(st_lab))
        self.next_slide()

        # the velocity becomes known: the weighted metric moves the sites, Lloyd re-converges
        tip = proj(1.45 * vel)
        v_arrow = Arrow(proj(ORIGIN), tip, buff=0, color=TARGET_COLOR, stroke_width=7,
                        max_tip_length_to_length_ratio=0.12).set_z_index(3)
        v_lab = MathTex(r"\mathbf{v}", font_size=40, color=TARGET_COLOR).next_to(tip, UP, buff=0.12).set_z_index(3)
        cap2 = Text("velocidade do alvo conhecida: CVT ponderada (Lloyd)", font_size=22, color=YELLOW).move_to(cap, aligned_edge=LEFT)
        self.play(GrowArrow(v_arrow), FadeIn(v_lab), Transform(cap, cap2), run_time=1.0)
        self.play(Write(w), FadeIn(wt), FadeIn(it_lab), run_time=1.2)
        for k, s in enumerate(hist, start=1):
            new_st = place_stats(stats(s))
            new_it = Text(f"iteração {k}", font_size=22, color=GREY_B).move_to(it_lab, aligned_edge=LEFT)
            self.play(
                Transform(agent_g, agents(s)),
                Transform(st, new_st),
                Transform(it_lab, new_it),
                run_time=0.7,
            )
        self.next_slide()
        self.clear_body()

        # --- complete architecture, before the results -----------------------
        self.architecture("10. Descentralizado: arquitetura completa", "generalized")
        self.next_slide(auto_next=True)
        self.clear_all()


# =============================================================================
# 11. Results - generalized simulation replay + plots
# =============================================================================
class Results(Base):
    SECTION = "Resultados"

    def construct(self):
        self.header("11. Resultados: 12 agentes, alvo móvel, 4 perdidos em t = 10 s")
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
        lab_in = Text("restrito ($x>0$)", font_size=20, color=RED_C).move_to(proj(np.array([6, 22, 13])))
        k = ValueTracker(0)

        def world():
            i = int(round(k.get_value()))
            g = VGroup()
            tg = T[:, i]
            v, _, _ = mesh_state(tg, meshes, R)
            v[:, 2] = np.maximum(v[:, 2], 1.5)   # as in the sim: no vertex below the 1.5 m ground clearance
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
            legend_entry(RED, "alvo"),
            legend_entry(WHITE, "agentes"),
            VGroup(Square(side_length=0.12, color=BLUE, fill_opacity=1).rotate(PI / 4), Text("objetivos de Lloyd", font_size=24, color=BLUE)).arrange(RIGHT, buff=0.15),
            legend_entry(BLUE_E, "malha", font_size=24, radius=0.04),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        leg.move_to(RIGHT * 2.5 + UP * 1.25, aligned_edge=LEFT)
        events = VGroup(
            Text(r"$t\approx 1$ s: disco na fronteira", font_size=20),
            Text("$t\\approx 2.5$ s: cone", font_size=20),
            Text(r"$t\approx 4$ s: esfera fechada", font_size=20),
            Text("$t=10$ s: 4 agentes sem energia", font_size=20, color=RED_C),
            Text(r"$t\approx 13$ s: regresso à barreira plana", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        events.next_to(leg, DOWN, buff=0.4).align_to(leg, LEFT)
        src = Text("cenário do Cap. 6, simulação em Python com o controlador de malha da tese, velocidade 2x", font_size=16, color=GREY_B).to_edge(DOWN, buff=0.55).to_edge(LEFT, buff=0.4)

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
        self.new_title("11. Resultados: análise quantitativa")
        p1 = captioned(plot_panel(ASSETS + "g_radial_hist_4_dropped_agents.png", height=2.4), "distância ao alvo ($R=5$ m)", 18)
        p2 = captioned(plot_panel(ASSETS + "g_inter_agent_dist_12_agents.png", height=2.4), "distância entre agentes (limiar 1 m)", 18)
        p3 = captioned(plot_panel(ASSETS + "g_velocity_12_agents.png", height=2.4), r"componentes da velocidade ($\|\mathbf{v}\|\le 3$ m/s)", 18)
        p4 = captioned(plot_panel(ASSETS + "g_low_level_tracking_error_12_agents.png", height=2.4), "distância ao objetivo de Lloyd", 18)
        grid = Group(p1, p2, p3, p4).arrange_in_grid(rows=2, cols=2, buff=(0.3, 0.2))
        grid.scale(min(13.7 / grid.width, 6.55 / grid.height))   # fill the body below the title
        grid.move_to(np.array([0, (3.1 - 3.45) / 2, 0]))
        self.play(LaggedStart(*[FadeIn(p) for p in grid], lag_ratio=0.25))
        self.next_slide()
        self.clear_body()

        # --- real-world test -------------------------------------------------
        self.new_title("11. Teste real: CVT geodésica + MPC de Bernstein")
        self.flight(
            "flight_mpc.mp4",
            [
                "4 Crazyflies, alvo virtual numa circunferência (raio 1.6 m), malha de 1001 vértices, CVT a cada 0.1 s",
                "trajetórias reais vs simuladas: cerca de 3-7 cm RMS na fase de formação",
            ],
            speed="fase principal em tempo real, o resto a 3x",
            extra=[("#E63CE6", "alvo (simulado)"), ("#5ABEEB", "malha da formação")],
        )
        self.next_slide()
        self.clear_body()

        # --- Lisbon airport scenario (simulation) ----------------------------
        self.new_title("11. Aeroporto de Lisboa: intrusão na zona proibida (simulação)")
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
        lab_wall = Text("parede (30 m fora)", font_size=18, color=GREY_A).next_to(proj(np.array([0, y1, zt])), UP, buff=0.08)
        lab_red = Text("zona proibida", font_size=18, color=RED_C).next_to(proj(np.array([x_red, y1, zt])), UP, buff=0.08)
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
        air = captioned(plot_panel(ASSETS + "g_lisbon_airspace.png", width=3.3), "Aeroporto de Lisboa (LPPT), zonas da ANAC", 16)
        air.move_to(RIGHT * 0.9 + UP * 1.85, aligned_edge=LEFT)
        clock = always_redraw(
            lambda: VGroup(MathTex("t =", font_size=36), DecimalNumber(t[int(round(k.get_value()))], num_decimal_places=1, font_size=36), Text("s", font_size=28))
            .arrange(RIGHT, buff=0.12)
            .move_to(LEFT * 6.7 + UP * 2.65, aligned_edge=LEFT)
        )
        events = VGroup(
            Text("barreira na parede, desliza com o alvo", font_size=19),
            Text("$t=13.9$ s: o alvo atravessa a parede → esfera", font_size=19),
            Text("$t=25$ s: 4 de 12 perseguidores ficam sem energia", font_size=19, color=RED_C),
            Text("$t=34.1$ s: o alvo entra → barreira na fronteira", font_size=19),
            Text("8 perseguidores mantêm-se a 1.00 m da fronteira", font_size=19, color=GREEN_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        if events.width > 6.0:
            events.set(width=6.0)
        events.move_to(RIGHT * 0.9 + DOWN * 0.75, aligned_edge=LEFT)
        facts = VGroup(
            Text("distância mín. entre agentes 1.43 m (envelope 1 m)", font_size=17, color=GREY_A),
            Text("velocidade máx. 3.09 m/s (limite 3 m/s) · modelo X500", font_size=17, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        facts.next_to(events, DOWN, buff=0.25).align_to(events, LEFT)
        src = Text(f"dados de simulação da tese (Cap. 6, trajetória 5), velocidade {speed}x", font_size=16, color=GREY_B).to_edge(DOWN, buff=0.55).shift(LEFT * 3.4)

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
    SECTION = "Conclusões"

    def construct(self):
        self.header("12. Conclusões")
        items = VGroup(
            VGroup(
                Text("Centralizado", font_size=26, color=YELLOW),
                Text("Rede de Fibonacci + controlador geométrico de Lyapunov: o mais simples, escala até 31 agentes,", font_size=21),
                Text("prova de convergência quase global (todas as condições iniciais exceto o antípoda)", font_size=21),
            ),
            VGroup(
                Text("Descentralizado", font_size=26, color=YELLOW),
                Text("Lloyd + Dijkstra em qualquer malha: barreira e envolvimento; a ponderação pela velocidade", font_size=21),
                Text("desloca os agentes para a frente; um MPC de Bernstein impõe limites de $v$, $a$ e penaliza colisões;", font_size=21),
                Text("consenso na rede de Delaunay: a formação recupera após perder 4 de 12 agentes", font_size=21),
            ),
        )
        for it in items:
            it.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(UP * 0.55)
        lim = VGroup(
            Text("Limitações", font_size=24, color=RED_C),
            Text("voos reais só em interior (Crazyflies, alvo virtual) · comunicação instantânea ·", font_size=20, color=GREY_A),
            Text("posição real do alvo + ruído · colisões penalizadas, não excluídas formalmente ·", font_size=20, color=GREY_A)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        lim.next_to(items, DOWN, buff=0.4).align_to(items, LEFT)
        for it in items:
            self.play(FadeIn(it, shift=RIGHT * 0.2), run_time=0.7)
        self.play(FadeIn(lim))
        self.next_slide(auto_next=True)
        self.clear_all()


class FutureWork(Base):
    SECTION = "Conclusões"

    def construct(self):
        self.header("13. Trabalho futuro")
        left = VGroup(
            Text("Maturidade do sistema", font_size=26, color=YELLOW),
            Text("• UAV reais: PX4, ROS 2, Gazebo, Pixhawk", font_size=21),
            Text("• comunicação realista (latência, perdas)", font_size=21),
            Text("• perceção real: deteção, LiDAR / RGB-D", font_size=21),
            Text("• enquadramento regulamentar, autoridades", font_size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        right = VGroup(
            Text("Novas funcionalidades", font_size=26, color=YELLOW),
            Text("• igualar a velocidade do alvo no MPC", font_size=21),
            Text("• malha deformada pela velocidade / forma", font_size=21),
            Text("• formações baseadas em conjuntos alcançáveis", font_size=21),
            Text("• CBF-MPC: garantias formais de segurança", font_size=21),
            Text("• desvio de obstáculos menos dispendioso", font_size=21),
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
        thanks = Text("Obrigado!", font_size=56).shift(DOWN * 1.7)
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
        self.header("Extra: estabilidade do controlador geométrico")
        lines = VGroup(
            MathTex(r"\Psi(\mathbf{q},\mathbf{q}^\star)=2-\sqrt{2(1+\mathbf{q}^\top\mathbf{q}^\star)},\qquad \mathbf{e}_q=\frac{S(\mathbf{q}^\star)\mathbf{q}}{\sqrt{2(1+\mathbf{q}^\top\mathbf{q}^\star)}}", font_size=34),
            MathTex(r"\Psi(\mathbf{q}^\star)=0,\quad \Psi(-\mathbf{q}^\star)=2,\quad 0\le\Psi\le 2", font_size=32),
            MathTex(r"\dot{\mathbf{q}}=S(\boldsymbol\omega)\mathbf{q},\ \ \boldsymbol\omega=-k_{geo}\mathbf{e}_q\ \Rightarrow\ \dot V=\dot\Psi=\mathbf{e}_q^\top\boldsymbol\omega=-k_{geo}\|\mathbf{e}_q\|^2\le 0", font_size=32),
            MathTex(r"D_c=\{\mathbf{q}:\Psi\le c\},\ c<2:\ \text{compacto, positivamente invariante, exclui }-\mathbf{q}^\star", font_size=32),
            MathTex(r"\text{LaSalle em }D_c:\ \mathbf{e}_q=0\iff\mathbf{q}=\mathbf{q}^\star;\ \ D_c\uparrow\mathbb{S}^2\setminus\{-\mathbf{q}^\star\}\ \text{quando}\ c\to2", font_size=32),
            MathTex(r"\dot e_r=-\lambda(t)e_r,\ \ \lambda=k_r e^{-\|\mathbf{e}_q\|^2}\ge k_r e^{-1}\ \ (\|\mathbf{e}_q\|^2=\sin^2\tfrac{\theta}{2}<1)", font_size=32),
            MathTex(r"\Rightarrow |e_r(t)|\le|e_r(0)|e^{-\lambda_{min}t}\ \ \text{(cascata: atitude independente de } r)", font_size=32),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        if lines.width > 13:
            lines.set(width=13)
        lines.move_to(DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(l) for l in lines], lag_ratio=0.15))
        self.next_slide(auto_next=True)
        self.clear_all()


class BackupControllers(Base):
    def construct(self):
        self.header(r"Extra: comparação de planeamento de trajetórias (3 agentes, $r^\star = 2$ m, 30 s)")
        head = ["Métrica", "fmincon (SQP)", "APF", "Lyapunov"]
        rows = [
            ["tempo até ao objetivo (todos < 0.1 m)", "> 30 s", "4.460 s", "17.610 s"],
            ["erro final ao objetivo, máx. [m]", "1.0459", "$9.1\\times10^{-14}$", "0.0024"],
            ["erro final de seguimento, máx. [m]", "1.1673", "$1.5\\times10^{-11}$", "$7.4\\times10^{-5}$"],
            ["distância média entre pares [m]", "4.7923", "3.3397", "3.4645"],
            ["tempo total de computação [s]", "14.761", "9.249", "7.826"],
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
            "Lyapunov escolhido: erro quase nulo, menor computação, prova de estabilidade;\nmais lento porque alinha a direção antes de fechar o raio.\nAPF como distribuição exige N conhecido à partida (fica só como salvaguarda).",
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
        self.header("Extra: métodos de cálculo direto (cerca de 1000 pontos, 100 ensaios)")
        spheres = sphere_row([("dir_latlon.png", "Lat-Lon, $K=29$ (1070 pontos)"),
                              ("dir_fibonacci.png", "Fibonacci (1001 pontos)"),
                              ("dir_poisson.png", "Disco de Poisson (1012 pontos)")])
        head = ["", "Lat-Lon", "Fibonacci", "Disco de Poisson"]
        rows = [
            ["tempo médio [s]", "$1.6\\times10^{-3}$", "$1.075\\times10^{-4}$", "$7.51\\times10^{-2}$"],
            ["desvio padrão [s]", "$7.69\\times10^{-4}$", "$3.66\\times10^{-5}$", "$1.4\\times10^{-3}$"],
            ["$d_{NN,min}/d_{NN,avg}$", "0.9712", "0.9171", "0.0640"],
            ["$CV(d_{NN})$", "0.0068", "0.0105", "0.2533"],
            ["Energia de Riesz ($s=1$)", "553166", "483587", "515388"],
            ["número de pontos", "só em grelha", "$P$ ímpar", "qualquer"],
        ]
        tab = table_grid(head, rows, [3.6, 2.4, 2.4, 2.4], font_size=19, row_h=0.42, highlight=2)
        g = Group(spheres, tab).arrange(DOWN, buff=0.35).move_to(DOWN * 0.2)
        if g.height > 6.3:
            g.set(height=6.3)
        self.play(FadeIn(spheres))
        self.play(FadeIn(tab))
        self.next_slide()
        self.clear_body()
        self.new_title("Extra: métodos de cálculo direto, tempo de execução (100 ensaios)")
        box = captioned(plot_panel(ASSETS + "dir_runtime.png", height=5.4),
                        r"Fibonacci $\approx 700\times$ mais rápido que o disco de Poisson, com menor dispersão", 20)
        box.move_to(DOWN * 0.15)
        self.play(FadeIn(box))
        self.next_slide(auto_next=True)
        self.clear_all()


class BackupIterative(Base):
    def construct(self):
        self.header("Extra: métodos de cálculo iterativo (30 pontos, 100 execuções)")
        spheres = sphere_row([("it_thomson.png", "Thomson, energia mínima"),
                              ("it_convex.png", "Lloyd, invólucro convexo"),
                              ("it_halfplane.png", "Lloyd, semiplano")])
        head = ["", "Thomson", "Lloyd (invólucro convexo)", "Lloyd (semiplano)"]
        rows = [
            ["tempo médio [s]", "0.085", "0.033", "1.367"],
            ["iterações médias até convergir", "–", "20.47", "29.11"],
            ["convergiu / 100", "0", "100", "93"],
            ["$d_{NN,min}/d_{NN,avg}$", "0.3246", "0.8585", "0.8486"],
            ["$CV(d_{NN})$", "0.6950", "0.1202", "0.0728"],
            ["Energia de Riesz ($s=1$)", "461.79", "363.06", "360.77"],
        ]
        tab = table_grid(head, rows, [4.0, 2.2, 3.0, 3.0], font_size=19, row_h=0.42)
        g = Group(spheres, tab).arrange(DOWN, buff=0.35).move_to(DOWN * 0.2)
        if g.height > 6.3:
            g.set(height=6.3)
        self.play(FadeIn(spheres))
        self.play(FadeIn(tab))
        self.next_slide()
        self.clear_body()
        self.new_title("Extra: métodos de cálculo iterativo, tempo de execução (100 execuções)")
        box = captioned(plot_panel(ASSETS + "it_runtime.png", height=5.4),
                        "semiplano: o mais uniforme, mas cerca de 40 vezes mais lento que o invólucro convexo", 20)
        box.move_to(DOWN * 0.15)
        self.play(FadeIn(box))
        self.next_slide(auto_next=True)
        self.clear_all()


class BackupDecentralized(Base):
    def construct(self):
        self.header("Extra: falha do invólucro convexo na esfera")
        a = captioned(plot_panel(ASSETS + "convex_problem.png", height=5.2), "o invólucro convexo escolhe o arco curto: célula de Voronoi errada")
        a.move_to(DOWN * 0.15)
        self.play(FadeIn(a))
        self.next_slide(auto_next=True)
        self.clear_all()


class BackupParameters(Base):
    def construct(self):
        self.header("Extra: parâmetros da simulação do Cap. 6")
        params = VGroup(
            Text(r"12 agentes, 30 s, $\Delta t=0.01$ s, $R=5$ m", font_size=26),
            Text(r"4 agentes perdidos aos 10 s, EMA $\alpha=0.2$, ruído $\pm 5\%$", font_size=26),
            Text("Bernstein de grau 5, horizonte 5", font_size=26),
            Text("$\\rho_{obs}=100,\\ \\rho_{ineq}=20,\\ w_g=1000,\\ w_s=1$", font_size=26),
            Text("$\\|\\mathbf{v}\\|\\le 3$ m/s, $\\|\\mathbf{a}\\|\\le 5$ m/s², $\\Theta=1$ m", font_size=26),
            Text(r"SQP $\le 100$ iterações, $\sigma=1.2$", font_size=26),
            Text("atualização de Voronoi a cada 100 passos (1 Hz)", font_size=26),
            Text("Mellinger: $m=5$ kg, $K_p=150\\,I$, $K_v=15\\,I$", font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(DOWN * 0.15)
        self.play(FadeIn(params))
        self.next_slide(auto_next=True)
        self.clear_all()
