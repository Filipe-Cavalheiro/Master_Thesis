# MSc thesis defence — Cooperative Encirclement of Non-Cooperative Drones

Manim + manim-slides presentation for Filipe Cavalheiro's MSc defence
(NOVA FCT, Electrical and Computer Engineering; adviser Prof. Bruno J. Guerreiro).
The presentation is in Portuguese (pt-PT).
The visual style (black background, section titles in the upper-left corner, red target,
white chasers, blue desired points, yellow highlights) follows the APCA 2026 paper
presentation, since it is the same research line.

Target length: **~19:30** of talk (the flight videos play their main phase in real time, the rest at 3x), over the 20 min hard limit (see the budget in `SPEAKER_NOTES.md`). See `SPEAKER_NOTES.md` for the
per-slide script and timing budget.

## Files

| File | Content |
|---|---|
| `slides.py` | all scenes (one `Slide` class per section, sub-slides via `next_slide()`) |
| `helpers.py` | colours, 3D projection helper, Fibonacci lattice, Voronoi/Lloyd, mesh controller, Bezier |
| `scenes.txt` | scene order (main talk + backup slides) |
| `manim.cfg` | 1920x1080, 30 fps, black background |
| `requirements.txt` | direct dependencies; `requirements.lock.txt` = exact versions that were tested |
| `assets/images/` | thesis figures converted from `Thesis_report/5-Figures` (PDF -> PNG, 200-450 dpi) + logos |
| `assets/data/generalized_sim_12agents.npz` | thesis Ch. 6 scenario (12 agents, 4 lost at 10 s) simulated with `Python/flight_sims/swarm_MPC_voronoi`, every 5th sample; made by `tools/make_generalized_replay.py` |
| `assets/videos/*.mp4` | real-flight clips shown at the end of the `CentralizedResults` and `Results` scenes, cut from `../Results/*/overlay/overlay.mp4` by `assets/videos/make_clips.sh` |
| `assets/images/lisbon_map.png`, `assets/data/lisbon_zones.npz` | Motivation airspace slide: dark top view of the Lisbon OSM meshes of the Gazebo world, LPPT ANAC zones (clipped to the map) and infrastructure points, ENU frame; made by `tools/make_lisbon_map.py` |
| `assets/data/lisbon_traj5.npz` | thesis Ch. 6 Lisbon-airport simulation (trajectory 5, 12 chasers, 4 lost at 25 s) in the wall frame, exported by `assets/data/make_lisbon_data.py` |
| `SPEAKER_NOTES.md` | speaker script (pt-PT) + timing |

## Setup

```bash
cd Presentation
uv venv .venv            # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt   # exact tested versions
source .venv/bin/activate
```

System requirements: ffmpeg, pango/cairo, a LaTeX distribution (for `MathTex`).

## Render

```bash
# quick check (480p15)
manim-slides render -ql slides.py $(cat scenes.txt)

# final quality (1080p30, as in manim.cfg) - takes considerably longer
manim-slides render slides.py $(cat scenes.txt)
# or 4K: manim-slides render -qk slides.py $(cat scenes.txt)
```

A single scene can be re-rendered on its own, e.g. `manim-slides render slides.py Results`.

### `render.sh`: render on the lab machine when it is reachable

`./render.sh` renders every scene and builds `presentation.html`. If the Windows render machine
(`user@192.168.50.210`, override with `RENDER_HOST`) answers over ssh with key login, the scenes
are rendered there, all in parallel, and copied back into `slides/`; otherwise they are rendered
here. Options: scene names (only those), `-q l|m|h|p|k`, `--local`, `--remote`, `--no-html`
(`./render.sh --help`).

The render machine has no LaTeX and no internet. `render.sh` compiles new formulas here first and
sends `media/Tex` along. Its one-time setup lives in `C:\Users\user\thesis_render`:

* `python\`: standalone CPython 3.12 for Windows (python-build-standalone, `install_only`);
  3.12 because `moderngl`/`glcontext` have no Windows wheels for 3.14.
* `venv\`: `python\python.exe -m venv venv`, then
  `venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements.txt`.
  The wheels are downloaded on Linux with
  `pip download -d wheels --platform win_amd64 --python-version 3.12 --only-binary=:all: -r requirements.txt -c requirements.lock.txt`
  (minus `pexpect`/`ptyprocess`), plus `colorama` and a wheel of `srt` (`pip wheel srt`).
* ssh key login: this machine's public key in `C:\ProgramData\ssh\administrators_authorized_keys`
  (`user` is an administrator), readable only by SYSTEM and Administrators.

Text uses Noto Serif (DejaVu Serif for →≈≤≥), pinned in `helpers.py` so both machines produce
the same slides.

## Present

```bash
# native presenter (arrow keys / space: next, left: previous, q: quit, f: fullscreen)
manim-slides present $(cat scenes.txt) --full-screen

# self-contained HTML (reveal.js) - works offline, open in any browser
manim-slides convert $(cat scenes.txt) presentation.html --one-file --offline
```

`scenes.txt` contains the main talk followed by the backup slides
(`BackupLyapunov`, `BackupControllers`, `BackupDirect` and `BackupIterative` (comparisons of the direct and
iterative point-distribution methods, thesis appendix), `BackupParameters` (Ch. 6 simulation parameters)).
To present only the main talk, drop those names.

**Shortened talk:** the sphere-only decentralized stage (thesis Ch. 5, scene `Decentralized`, and its
backup `BackupDecentralized`) is left out of `scenes.txt`; both classes are still in `slides.py`. The
generalized stage (Ch. 6, scenes `GeneralizedMesh` and `GeneralizedMPC`), which is decentralized as
well, is presented as "Decentralized" (sections 9 and 10), and the roadmap and conclusions show two
stages. To restore: put `Decentralized` back after `CentralizedResults` (and `BackupDecentralized`
before `BackupParameters`) and revert the titles, roadmap, conclusions and section bar.

## Language

The slide text is written in Portuguese (pt-PT) directly in `slides.py`. Text inside the thesis figures and in the flight videos is in English, and numbers keep the
decimal point of the figures.

## Notes / provenance

* All numbers on the slides come from the thesis text, tables and figures
  (Ch. 1, 4, 5, 6, 7 and the appendix).
* The animations of the Fibonacci lattice, Lyapunov controller (5 agents), Lloyd's algorithm
  (2D and on the sphere), mesh controller and Bernstein MPC are **illustrative
  re-implementations** of the thesis methods, written for the slides. The spherical Lloyd
  animation uses a sample-based Voronoi partition, not the convex-hull/half-plane code; its live
  CV value is illustrative and the quoted thesis values are in the table next to it.
* The 3D replay in `Results` plays simulation data at 2x speed: the Ch. 6 straight-boundary
  scenario (circle at 0.25 rad/s, 12 agents, 4 lost at 10 s) re-simulated with
  `Python/flight_sims/swarm_MPC_voronoi`, whose cone has its rim on the wall
  (`tools/make_generalized_replay.py`). The V16 run behind the thesis figures used a cone whose
  outer ring curled back past the wall (goals up to 3.9 m outside), so it is not replayed; the
  plots on the next slide are still the thesis figures. The mesh drawn behind the agents is
  recomputed from the true target position with the thesis mesh-controller equations.

Math on the slides: any `$...$` segment of a `Text` string is rendered with LaTeX inline
(`helpers.mixed_text`); `MathTex` for whole formulas. The thesis title and subtitle are kept in
English. The control architectures shown before the simulation results of each chapter
(`arch_diagram` in `slides.py`) are drawn on the slides, not taken from the thesis figures: the
centralized one merges the Ch. 4 sub-diagrams (path planning, Mellinger controller, target
estimation) and the Fibonacci/Munkres central node into one diagram. The mesh-controller animation and the `Results` replay both use the thesis cone (rim on
the wall).

## Structure

16 main scenes (54 key presses: the mesh controller stops at the cone and at the sphere, both
replays of `Results` stop at every phase) + 6 backup scenes (9 key presses). Each scene ends with a short fade-out that
advances automatically (`auto_next`) into the next scene, so no blank "extra" key press is needed.
Speaker notes from `SPEAKER_NOTES.md` (`#### [Scene.k]` blocks) are attached to the slides at
render time: press `S` in the HTML export to open the reveal.js speaker view.
After editing the notes, re-render the affected scene(s) and re-run `convert`.

## Left for the author

* The last sub-slide of `CentralizedResults` and of `Results` plays the camera + simulation overlay of a Crazyflie/OptiTrack flight
  (`Results/swarm_geometric_Fibonacci_sim_target_trajectory_20261113_182427` and
  `Results/swarm_MPC_voronoi_sim_target_trajectory_20261113_144433`, whole flights, 36 s and 44 s: the main controller phase in real time, take off, approach, return and landing at 3x, see `assets/videos/make_clips.sh`;
  the next key press skips the rest of a video). These flights are not in the thesis text, and the `Conclusions` slide
  still lists "simulation only" as a limitation: adjust it if you want the two to agree.
* The motivation headlines (RTP, Público) and logos are reused from the APCA presentation;
  the DFS numbers (161 in 2024, 225 in 2025) are from thesis Ch. 1.
* Rehearse with a timer: the budget is in `SPEAKER_NOTES.md`; animations play automatically on
  each key press, so the timing mostly depends on speech.
