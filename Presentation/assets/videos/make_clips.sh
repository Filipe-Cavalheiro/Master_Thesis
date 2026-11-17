#!/usr/bin/env bash
# Cut the real-flight clips shown at the end of the CentralizedResults and Results scenes from the
# run folders in ../Results (flight_overlay.py: camera + simulation overlay, target, formation mesh).
# Usage (from Presentation/):  bash assets/videos/make_clips.sh
#   whole overlay.mp4 (take off to landing), 30 fps, 1600 px wide, no audio. Only the main
#   controller phase plays in real time; take off, approach, return and landing play at FAST x,
#   with a "FASTx" badge in the upper-right corner.
set -euo pipefail
R=../Results
OUT=assets/videos
FAST=3
FONT=$(fc-match -f '%{file}' "Noto Sans:bold")

# start and end [s] of the main phase inside overlay.mp4, from overlay/overlay_metrics.csv
# (video_t is camera-video time; overlay.mp4 starts flight_overlay.py's --margin = 2 s before
# the first flight phase)
main_phase() {  # run_dir phase_name
    python3 - "$1/overlay/overlay_metrics.csv" "$2" <<'EOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["phase"]]
t0 = float(rows[0]["video_t"]) - 2.0
t = [float(r["video_t"]) - t0 for r in rows if r["phase"] == sys.argv[2]]
print(f"{min(t):.2f} {max(t):.2f}")
EOF
}

clip() {  # run_dir  main_phase_name  dst
    read -r a b < <(main_phase "$1" "$2")
    echo "$3: real time from $a s to $b s of overlay.mp4, ${FAST}x outside"
    local badge="drawtext=fontfile=$FONT:text='${FAST}x':fontsize=44:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10:x=w-tw-24:y=20"
    local base="fps=30,scale=1600:-2:flags=lanczos"
    ffmpeg -v error -y -i "$1/overlay/overlay.mp4" -an -filter_complex "
        [0:v]$base,split=3[v1][v2][v3];
        [v1]trim=end=$a,setpts=(PTS-STARTPTS)/$FAST,fps=30,$badge[s1];
        [v2]trim=start=$a:end=$b,setpts=PTS-STARTPTS[s2];
        [v3]trim=start=$b,setpts=(PTS-STARTPTS)/$FAST,fps=30,$badge[s3];
        [s1][s2][s3]concat=n=3:v=1:a=0[out]" \
        -map "[out]" -c:v libx264 -crf 18 -pix_fmt yuv420p "$OUT/$3"
}

# Fibonacci lattice + geometric controller, 3 Crazyflies
clip "$R/swarm_geometric_Fibonacci_sim_target_trajectory_20261113_182427" "Main Controller" flight_geometric.mp4
# Geodesic CVT + MPC, 4 Crazyflies (CF02 crashes in the Return phase)
clip "$R/swarm_MPC_voronoi_sim_target_trajectory_20261113_144433" "Formation" flight_mpc.mp4
