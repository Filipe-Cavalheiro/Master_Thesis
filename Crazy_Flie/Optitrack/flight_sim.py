"""
Simulation of a Crazyflie flight, to compare with its log, or a dry run of
a flight code before flying it. The code is in the Python simulation V17
(Python/flight_sims/V17_Lisbon_aeroport/Python/flight_replay.py, see there);
this script only runs it for any swarm folder of this directory.

    python flight_sim.py swarm_MPC_voronoi                       # newest run of swarm_MPC_voronoi (either target)
    python flight_sim.py swarm_MPC_voronoi --target real         # newest target-drone run
    python flight_sim.py swarm_voronoi_sim_target --log ../../Results/swarm_voronoi_sim_target_trajectory_20261026_110924/trajectory_20261026_110924.csv
    python flight_sim.py swarm_MPC_voronoi_sim_target --dry-run --traj 5 --drones 4
    python flight_sim.py swarm_MPC_voronoi --no-show             # save the figures only

For swarm_MPC_voronoi the same runs are python main.py --IRL_log [LOG] /
--arena in the V17 folder.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Python" / "flight_sims" / "V17_Lisbon_aeroport" / "Python"))

from flight_replay import main  # noqa: E402

if __name__ == "__main__":
    main()
