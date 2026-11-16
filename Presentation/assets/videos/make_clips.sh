#!/usr/bin/env bash
# Cut the real-flight clips shown at the end of the CentralizedResults and Results scenes from the
# run folders in ../Results (flight_overlay.py: camera + simulation overlay, target, formation mesh).
# Usage (from Presentation/):  bash assets/videos/make_clips.sh
#   whole overlay.mp4 (take off to landing), real time, 30 fps, 1600 px wide, no audio
set -euo pipefail
R=../Results
OUT=assets/videos

clip() {  # src  dst
    ffmpeg -v error -y -i "$1" -an \
        -vf "fps=30,scale=1600:-2:flags=lanczos" \
        -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT/$2"
}

# Fibonacci lattice + geometric controller, 3 Crazyflies
clip "$R/swarm_geometric_Fibonacci_sim_target_trajectory_20261113_182427/overlay/overlay.mp4" flight_geometric.mp4
# Geodesic CVT + MPC, 4 Crazyflies (CF02 crashes in the Return phase)
clip "$R/swarm_MPC_voronoi_sim_target_trajectory_20261113_144433/overlay/overlay.mp4" flight_mpc.mp4
