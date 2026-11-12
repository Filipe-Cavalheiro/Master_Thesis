# Crazyflie real flights

Real-flight code of the thesis, for Crazyflie 2.x drones controlled from a PC over Crazyradio (`cflib`).

| Folder / file | Content |
|---|---|
| [`Optitrack/`](Optitrack/README.md) | **current** flight code: position from OptiTrack (Motive, NatNet) streamed to the onboard Kalman filter. Swarm folders, standalone scripts, two-stage Ctrl+C, flight logs, analysis and flight simulation tools |
| `Lighthouse/` | older version of the same flights, with the Lighthouse positioning system (`light_house_conf.yaml`: base-station calibration). Figure 8, box, rectangle, Fibonacci and Voronoi swarms, a collision-avoidance test. Kept for reference: no flight logging, no Ctrl+C handler |
| `natnet_test.py` | checks the connection to Motive (edit `LOCAL_IP` / `MOTIVE_IP`) |
| `cache/` | Crazyflie TOC cache (parameter / log tables; speeds up connecting) |

## Setup

The Python environment is the repository's root `.venv` (`source .venv/bin/activate` from the repository root; see the root README): it includes `cflib` and `natnet`.

```bash
source .venv/bin/activate     # from the repository root
cd Crazy_Flie/Optitrack/...   # then python <script>
```

The Crazyradio needs the usual USB permissions (udev rules; see the Bitcraze `cflib` installation guide).

Flight logs are written to the repository `Results/` folder, one folder per run (see [`Results/README.md`](../Results/README.md)).
