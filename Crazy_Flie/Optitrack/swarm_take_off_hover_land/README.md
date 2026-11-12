# Swarm take off / hover / land

Take off to `TAKEOFF_HEIGHT` above each takeoff point, hover `HOVER_TIME` (8 s), return above the takeoff points and land. Run it first with new drones, batteries or controller settings, before the Voronoi formation.

Self-contained (the logs go to `Results/` two levels up from `Crazy_Flie/Optitrack`). The Crazyflie/OptiTrack code and settings are in `Structs.py` (URIs, Motive IPs, arena, heights, battery/yaw limits), `cf_funcs.py`, `natnet_funcs.py` and `trajectory_logger.py`. Logs go to the repository `Results/` folder, one folder per run (`Results/<this folder>_trajectory_<date>/`) and the Crazyflie TOC cache to `cache/`, both in this folder.

Requires `cflib`, `natnet`, `numpy` and `scipy`.

```bash
python main.py
```

The pose of the filming phone (Motive rigid body `CAMERA_RIGID_BODY_ID`, `Structs.py`; `None` to
disable) is logged from the same NatNet frames to `<log>_camera.csv/.json` next to the trajectory
log (`camera_logger.py`), for the video overlay
(`Python/demos_and_helpers/video_overlay/flight_overlay.py`).
