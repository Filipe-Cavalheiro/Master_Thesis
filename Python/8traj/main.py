import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from Structs import Param, SimulationData
from Animation_display import create_figures
from helper_funcs import *
from time_alignment import (
    run_simulation, load_real_reference, find_best_time_offset, plot_alignment,
)

# =============================================================================
# Alignment settings
# =============================================================================
AUTO_ALIGN = True
ALIGN_SOURCE = "opti"            # compare against "opti" (OptiTrack) or "est" (Kalman)
ALIGN_SEARCH_RANGE = (-1.0, 1.0) # s, trajectory time offsets to try
FALLBACK_TRAJ_OFFSET = -0.5      # s, used if AUTO_ALIGN is off or no real data exists

# =============================================================================
# Simulation initialization
# =============================================================================
param = Param()

(position, velocity, attitude, ang_vel,
 real_world_time_offset) = get_initial_real_offset(param)
init_state = (position, velocity, attitude, ang_vel)

param.sim.print_parameters()

# =============================================================================
# Time alignment against the real flight (if available)
# =============================================================================
traj_offset = FALLBACK_TRAJ_OFFSET
real = load_real_reference(param, real_world_time_offset, source=ALIGN_SOURCE)

if AUTO_ALIGN and real is not None:
    result = find_best_time_offset(
        param, init_state, real, search_range=ALIGN_SEARCH_RANGE
    )
    traj_offset = result["offset"]
    if param.display.verbose:
        plot_alignment(param, init_state, real, result)
else:
    print(f"[align] skipped; using trajectory offset {traj_offset:+.3f} s")

# =============================================================================
# Main simulation (with the chosen offset)
# =============================================================================
agent_true_hist = run_simulation(param, init_state, traj_offset, verbose=True)

# =============================================================================
# Results
# =============================================================================
data = {
    "agent_true_hist": agent_true_hist,
    "removed_initial_pos": False,
    "colors": plt.cm.tab10(np.linspace(0, 1, param.sim.nD))
}

if param.display.store_data == True:
    print("Saving simulation data (.pkl)")
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, "simulation_results.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

if param.display.verbose:
    sim_data = SimulationData(**data)
    create_figures(sim_data, param)