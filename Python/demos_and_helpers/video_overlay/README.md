# Video overlay: real flight vs simulation, seen from a tracked camera

Film a Crazyflie flight in the OptiTrack arena with a camera (phone or GoPro)
that carries its own OptiTrack rigid body. The pipeline then:

* draws the **real** (OptiTrack) and the **simulated** drone trajectories over the video,
  as seen by a virtual camera that follows the real one (`overlay.mp4`, for the talk and YouTube);
* builds a **stroboscopic "motion through time" image** from a tripod clip: the real
  drones at N instants are cut out of the video and pasted on one background, and the
  simulated drones of the same instants are drawn as ghosts (`strobe.png`, for the paper);
* measures the **real vs sim difference** (3D RMS / max per drone, and in pixels),
  to back the claim that the difference is small (`error.pdf`, `metrics.json`).

The simulation is `Crazy_Flies/flight_sim.py`: it runs the same flight code (formation,
MPC, setpoint loop) on the Crazyflie plant of `Python/plant.py`, starting from the state
logged at the start of the Formation phase. It writes `<log>_sim.csv` next to the flight log.

```
video_overlay/
  overlay_lib.py            frames, poses, camera model, logs, time alignment, video I/O
  record_mocap.py           NatNet recorder: drones + camera rigid body + sync events
  calibrate_intrinsics.py   checkerboard -> K + distortion
  sync.py                   video <-> mocap time (flash detection, motion refinement)
  calibrate_extrinsics.py   hand-eye offset rigid body -> optical centre (click / auto / solve)
  render_overlay.py         overlay video (real green, sim orange, target magenta)
  stroboscopic.py           motion-through-time composite + real-vs-sim error plot/metrics
  flight_overlay.py         hand-held phone video of swarm_take_off_hover_land vs its simulation
                            (phone rigid body logged by the flight code, phone_align.py calibration)
  flight_dashboard.py       dashboard.mp4 of flight_overlay.py: overlay + V16-style animated plots
  make_synthetic.py         synthetic sessions with ground truth (no footage needed)
  run_synthetic_test.py     end-to-end test on them
  test_output/              test results (see "Validation" below)
```

Python environment: the repository's root `.venv` (installed once from the root `requirements.txt`):

```bash
source <repo>/.venv/bin/activate     # <repo> = the repository root
```

## Frames and time bases

* World: OptiTrack converted to Z-up as the flight code does (`natnet_funcs.py`):
  `(x, y, z) = (Zm, Xm, Ym)`. Same frame as the `opti_*` columns of the flight logs.
  Positions you read in the Motive GUI are Y-up: convert them the same way.
* `T_wc = T_wb T_bc`: Motive gives the rigid body `T_wb`, the calibration gives the
  fixed hand-eye offset `T_bc` (rigid body -> OpenCV optical frame).
* `t_mocap = scale * t_video + offset` (`sync.json`). The recorder log and the flight
  log have different time origins: they are aligned automatically by matching the drone
  positions (`estimate_log_offset`, residual printed). Sim time 0 = start of the
  Formation phase of the flight log.

## Arena day protocol

### Equipment

* The camera. Phone: rear main lens (1x, not the ultra wide), or a GoPro in **Linear**
  lens mode. Fully charged, enough storage, a phone tripod clamp.
* A second camera on a **tripod** for the stroboscopic image (it also needs a rigid
  body, or put the tracked camera on the tripod for one flight).
* 4-5 OptiTrack markers (14 mm) per camera and a small rigid **marker plate / cage**:
  3D-printed or stiff cardboard/plywood plate fixed to the phone clamp, markers on short
  rods so the body of the phone does not hide them.
* Printed checkerboard, **9 x 6 inner corners** (10 x 7 squares), squares ~30 mm, glued
  flat on a rigid board (or shown full screen on a tablet). Measure the square side.
* A laptop running `record_mocap.py` (it can be the flight laptop) whose screen the
  camera can see: it is the sync flash.
* 6-10 surveyed points for the hand-eye calibration: loose markers or spare
  Crazyflies (rigid bodies), on the floor **and** on boxes / stands at different
  heights (0.3-1.5 m), spread in the camera view.
* Matte tape to cover shiny parts of the phone/clamp that Motive could see as markers.

### Camera settings (keep them identical for calibration and flights)

* Fixed resolution and frame rate: 1920x1080 at 30 fps (or 60). If the phone records
  variable frame rate, the pipeline uses the container timestamps, which is fine.
* **Video stabilization OFF** (EIS/HyperSmooth warps every frame: the intrinsics stop
  being valid). No zoom; lock focus and exposure (AE/AF lock), fast shutter if
  possible (1/250 s or faster) to limit motion blur and rolling-shutter skew.
* iPhone Pro: the LiDAR emits IR pulses that the OptiTrack cameras can see; cover it or
  check Motive for ghost markers while the phone is in the volume.

### 1. Camera rigid body (in Motive)

1. Fix the marker plate to the camera clamp so that **nothing can move** between the
   markers and the lens for the whole day (the hand-eye offset is calibrated once).
   Asymmetric layout, markers >= 4 cm apart, the constellation spread >= 10 cm, no
   marker in front of the lens.
2. Motive: select its markers, create the rigid body, give it a **unique ID** (e.g. 10,
   not 1-4: those are the drones), check that it streams (Data Streaming: rigid bodies
   on, up axis Y as for the flights).
3. Walk the camera through the volume and check that it stays tracked
   (`record_mocap.py` prints the number of valid camera frames).

### 2. Intrinsics (once per camera / video mode; can be done before the arena day)

Film the checkerboard ~1 min in the flight video mode: move it across the whole image
(corners included), tilted up to ~45 deg, near and far, slowly (no motion blur).

```bash
python calibrate_intrinsics.py --video calib.mp4 --board 9 6 --square 0.030 \
    --out sessions/day1/intrinsics.json --preview sessions/day1/undistort.png
```

RMS should be < 0.5 px. Copy `intrinsics.json` into every session directory of that camera.

### 3. Start the recording (every flight)

On the laptop (Motive network), next to the flight script:

```bash
python record_mocap.py --out sessions/day1_flight1 --camera-id 10 --drone-ids 1 2 3 4 --flash 4
```

* Start the camera recording first, then press **Space** in the full-screen black
  window and keep the laptop screen in the camera view while it flashes white 4 times
  at irregular intervals. Do it again **at the end** of the flight (two flash groups
  far apart also give the clock drift).
* No display / fallback: run without `--flash`, press Enter while clapping / firing the
  phone flash in the camera view (then `sync.py --event-kind key`), and refine with the
  motion of the takeoff (`sync.py --refine`, static camera).
* If the flight laptop runs the recorder too and Motive streams unicast to the flight
  script, start the recorder with `--multicast` and switch Motive to multicast, or run
  the recorder on a second computer (the logs are aligned afterwards anyway).

### 4. Hand-eye calibration shots (once per day, after the rigid body is fixed)

With the surveyed points in place and the recorder running:

1. Hold the camera still for ~2 s each at >= 3 clearly different positions and
   orientations (left / centre / right of the arena side, different heights), all
   points in view. Turning on the **phone torch** makes the retro-reflective markers
   glow: they become the brightest blobs (needed by `auto`).
2. Note the video times of these still shots.
3. Write `points.csv` (`name,x,y,z`, Z-up metres): positions from Motive (convert
   `x = Zm, y = Xm, z = Ym`), or use the drones sitting on their takeoff spots (`--drones`).
4. Remove the stands from the flight volume before flying.

### 5. Fly

Start the flight script as usual (`Crazy_Flies/swarm_MPC_voronoi_sim_target/main.py`).
Film the formation phase. Handheld: move slowly and keep all drones in view. Tripod:
frame the whole flight volume, do not touch the camera. Stop the recorder (Esc / Ctrl+C)
after the landing and the final flashes.

### 6. Processing

Put the video in the session directory as `video.mp4` and complete
`session.json` with the flight log and its simulation:

```bash
# simulation of the flight (same start state, same flight code) -> <log>_sim.csv
python <repo>/Crazy_Flie/Optitrack/flight_sim.py swarm_MPC_voronoi_sim_target \
    --log <repo>/Results/swarm_MPC_voronoi_sim_target_trajectory_YYYYMMDD_HHMMSS/trajectory_YYYYMMDD_HHMMSS.csv --no-show
```

```json
{
  "video": "video.mp4",
  "drones": "mocap_trajectory.csv", "camera": "mocap_camera.csv", "events": "mocap_events.csv",
  "camera_id": 10,
  "flight_log": "<repo>/Results/swarm_MPC_voronoi_sim_target_trajectory_YYYYMMDD_HHMMSS/trajectory_YYYYMMDD_HHMMSS.csv",
  "sim":        "<repo>/Results/swarm_MPC_voronoi_sim_target_trajectory_YYYYMMDD_HHMMSS/trajectory_YYYYMMDD_HHMMSS_sim.csv"
}
```

```bash
S=sessions/day1_flight1
python sync.py $S                                    # flashes -> sync.json (check the residuals: a few ms)
python calibrate_extrinsics.py click $S --times 4.0 9.5 15.2 --points points.csv   # click the points
python calibrate_extrinsics.py solve $S              # -> extrinsics.json (RMS a few px)
python calibrate_extrinsics.py auto  $S --points points.csv --t0 3 --t1 20 --step 0.5   # more points
python calibrate_extrinsics.py solve $S              # final hand-eye offset
python render_overlay.py $S --grid --out overlay_check.mp4   # check: the grid must stick to the floor tape
python render_overlay.py $S --trail 3                # -> overlay.mp4 (talk / YouTube)
python stroboscopic.py $S --n 8                      # tripod session -> strobe/strobe.png, error.pdf, metrics.json
```

The hand-eye offset is a property of the camera + marker plate: once solved, copy
`extrinsics.json` to the other sessions of the same day (re-check with `--grid`).
A tripod camera with the rigid body still attached is processed the same way.

Troubleshooting:

* Overlay drifts in time (drones lead/lag their markers): check `sync.json` residuals;
  `sync.py --refine` (static camera) refines the offset from the drone motion.
* Overlay offset constant in the image: hand-eye offset wrong. More / better spread
  correspondences (several camera poses, points at different heights).
* Grid bends away from the floor tape at the image borders: redo the intrinsics with
  more views near the corners (or `--model rational` for wide lenses).
* Strobe cut-outs include people/shadows: lower `--gate`, raise `--thr`, or pick a
  window without people in view (`--t0 --t1`).

## Phone camera correction in Motive (one-time, alternative to the hand-eye step)

Instead of carrying a hand-eye offset in `extrinsics.json`, the phone rigid body can be
corrected once in Motive so that its streamed pose **is** the camera:
`+Z` forward (optical axis), `+Y` up, `+X` left in the rigid body's local Motive axes
(`+x` forward, `+y` left, `+z` up in the Z-up world). The virtual camera is then
`phone_align.camera_T_wc(pos, quat)` of the streamed pose.

```bash
# phone = rigid body 0, a Crazyflie on the floor = rigid body 1; phone torch ON, video recording
python phone_capture.py --out sessions/phone_calib          # Enter at each of 3-5 still stations, q to finish
python phone_align.py sessions/phone_calib --video phone.mp4 --first-time 5.2 --intrinsics intrinsics.json
#   click the markers on 1-2 stations -> Solve -> Snap -> Solve -> Save: prints the Motive correction
#   (translate the pivot by X Y Z mm in local axes, then rotate X, Y, Z deg)
# apply it in Motive, then check: the residual must be ~0 (a few mm / < 0.3 deg)
python phone_capture.py --out sessions/phone_check
python phone_align.py sessions/phone_check --video check.mp4 --first-time 3.0 --corrected
```

Stations: different positions **and** heights around the object (one station leaves the
translation undetermined; the solver says which DOF are not determined by the data).
Synthetic test (4 stations, 4 markers, 68 deg pinhole): T_bc recovered to 0.19 deg / 10 mm,
and the verification run after applying the printed correction gives 0.13 deg / 8 mm.

## Outputs and where they go

| Output | Use |
|---|---|
| `strobe/strobe.png` (annotated), `strobe_clean.png`, `strobe_overlay.png` | paper figure (Crazyflie Arena Experiment); clean version to annotate in the paper style |
| `strobe/error.pdf` + `metrics.json` (RMS, mean, p95, max per drone, per axis) | paper: quantifies "real vs sim difference is small" |
| `overlay.mp4` (H.264, 30/60 fps) | presentation and YouTube video; `--t0/--t1` for clips, `--width 1280` for slides |
| `overlay_metrics.csv` | per-frame 3D / pixel error, for plots synced to the video |

## Validation (synthetic end-to-end test)

No footage exists yet, so `make_synthetic.py` renders two sessions with known ground
truth from a real flight (`Results/swarm_MPC_voronoi_sim_target_trajectory_20261026_141807/trajectory_20261026_141807.csv`,
3 drones, and its simulation `_sim.csv`): a ray-cast arena (floor tape grid, netted
walls), the drones, 8 bright surveyed markers, screen flashes, lens distortion, sensor
noise and full-frame exposure. The recorder logs carry mocap noise (0.3 mm, 0.04 deg),
a different time origin than the flight log (7.3 s) and the video (1.0137 s, clock
30 ppm fast), and a non-trivial hand-eye offset. `run_synthetic_test.py` then runs every
tool above (the manual click step is replaced by true projections + 0.7 px noise) and
compares with the truth:

```bash
python run_synthetic_test.py          # --regen to re-render the synthetic sessions
```

Result of the last run (`test_output/report.json`, all 9 checks PASS):

| Stage | Recovered vs ground truth |
|---|---|
| Intrinsics (23 checkerboard views) | RMS 0.14 px; fx, fy, cx, cy within 1.6 px |
| Flash sync (4 flashes, automatic) | 4/4 matched, residuals < 1.3 ms, time error 1.2 ms (moving) / 0.15 ms (static) at mid video, 1/30 s frames |
| Motion-only sync (static, start 300 ms off) | 2.2 ms error, so the takeoff/flight motion alone is enough as a fallback |
| Hand-eye T_bc, moving camera (3 click frames + 240 auto points) | 0.09 deg, 6.6 mm |
| Hand-eye T_bc, static camera (one pose) | 0.10 deg, 10.3 mm (along the optical axis, weakly observable from one pose; harmless for that camera) |
| Whole chain: projected mocap drone vs true image position | mean 0.56 / 0.59 px, max 1.3 / 1.9 px (moving / static) |
| Real vs sim (this flight, Formation phase 20.7 s) | RMS 9.2 cm, max 13.1 cm (per drone 7.7-10.2 cm RMS) |

The last row is the actual flight of 26 Oct against its simulation. It is not a
synthetic number: it is what the paper figure would quantify for that flight.
Because the hand-eye offset is better determined from several camera poses, calibrate
it with the handheld procedure (step 4) even for the camera that later goes on the tripod.

Files in `test_output/`: `overlay_moving.mp4` (handheld, with the arena grid),
`overlay_static.mp4`, `strobe.png` / `strobe_clean.png` / `strobe_overlay.png`,
`strobe_error.pdf|png`, `strobe_metrics.json`, `intrinsics_undistort_preview.png`,
`report.json`. The full synthetic sessions (videos, logs, calibration files) are in
`test_output/synthetic/` (git-ignored, regenerate with `--regen`, ~4 min).

## Hand-held phone video of a single flight (`flight_overlay.py`): take off / hover / land, 8traj

The flight code (`Crazy_Flie/Optitrack/swarm_take_off_hover_land`) logs the phone rigid
body next to the flight log (`_camera.csv/.json`); the simulation is
`Python/flight_sims/swarm_take_off_hover_land/main.py --IRL_log` (`_sim.csv`). Put the phone video in
the run folder as `video.mp4`:

```bash
python flight_overlay.py <repo>/Results/swarm_take_off_hover_land_trajectory_X/trajectory_X.csv
#   --offset S      skip the automatic sync (video t = flight t + S)
#   --flip on|off   force the phone orientation (default auto)
#   --no-dashboard  skip the (slow, ~0.4 s/frame) dashboard
```

Outputs in `<run folder>/overlay/`: `overlay.mp4`, `sim_camera.mp4`, `side_by_side.mp4`,
`dashboard.mp4` (overlay video, 3D scene with the tracked camera and its frustum, zoomed top
view in cm, z(t) real / sim / setpoint and the real - sim error components with live |e| and
RMS, trails and curves growing with time as in `V16_generic_mesh/animation_display.py`),
`stills/`, `overlay_metrics.csv`, `sync.json`.

* Sync: image motion of the video (phase correlation) vs the image motion predicted from the
  phone rotation, both smoothed over 0.2 s. Check `sync.json`: correlation >= 0.5 and well
  above the next best peak.
* Phone held upside down compared with the calibration video (landscape the other way): the
  image is rotated by 180 deg about the optical axis, the predicted motion has the opposite
  sign and the overlay lands on the wrong side of the image. `--flip auto` tries both and keeps
  the one with the higher signed correlation (`"flip"` in `sync.json`). Flight of 10 Nov
  (`trajectory_20261110_161917`): flip on, offset 10.16 s, correlation 0.66 (next best 0.32;
  0.08 without the flip).

### Figure 8 (`Crazy_Flie/Optitrack/8traj.py`)

The flight script logs the phone rigid body too (`CAMERA_RIGID_BODY_ID = 0`, `camera_logger.py`
next to it, from the moment NatNet starts: do the pan / tilt sync gestures before the take off)
and an `id` column. Simulation and overlay:

```bash
python <repo>/Python/flight_sims/8traj/main.py --IRL_log <repo>/Results/8traj_trajectory_X/trajectory_X.csv --no-show
python flight_overlay.py <repo>/Results/8traj_trajectory_X/trajectory_X.csv
```

`main.py --IRL_log` replays the setpoint stream of `8traj.py` (constants read from the script) on the
V8 model, writes `_sim.csv`, `simulation/` and `comparison/` (position, top view, difference, metrics.json).
The dashboard switches to x(t), y(t), z(t) panels and a top view in metres when the drone moves
more than 0.3 m horizontally. The phases (and the stills, one per phase) are those of the log.
