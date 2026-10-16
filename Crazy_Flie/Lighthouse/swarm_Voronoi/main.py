
import threading
import os

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm

from cf_funcs import *

if __name__ == "__main__":
    os.system('clear')
    cflib.crtp.init_drivers()

    active_uris = discover_active_uris()

    if not active_uris:
        raise RuntimeError("No drones found")

    param = Param(number_of_drones=len(active_uris), traj=2)

    # init drones
    # last drone is target
    drone_states = {
        uri: DroneState()
        for uri in active_uris
    }

    target_lock = threading.Lock()
    stop_event = threading.Event()
    target_uri = active_uris[-1]
    target_state = drone_states[target_uri]
    # --------------------------------------------------------
    # Voronoi thread
    # --------------------------------------------------------
    voronoi_thread = threading.Thread(
        target = voronoi_worker,
        args = (target_state, target_lock, param, stop_event),
        daemon = True
    )

    # --------------------------------------------------------
    # Crazyflies
    # --------------------------------------------------------
    factory = CachedCfFactory(rw_cache="./cache")

    with Swarm(active_uris, factory=factory) as swarm:
        swarm.reset_estimators()

        print("Lights On")
        swarm.parallel_safe(light_check)
        
        swarm.parallel_safe(
            init_position_getting,
            args_dict={
                uri: (state,)
                for uri, state in drone_states.items()
            }
        )
        time.sleep(1)
        voronoi_thread.start()

        swarm.parallel_safe(arm)
        
        swarm.parallel_safe(
            take_off,
            args_dict={
                uri: (state, 
                      param)
                for uri, state in drone_states.items()
            }
        )

        swarm.parallel_safe(
            controller,
            args_dict={
                uri: (
                    active_uris,
                    state,
                    param
                )
                for uri, state in drone_states.items()
            }
        )

        stop_event.set()

        swarm.parallel_safe(
            land,
            args_dict={
                uri: (state, 
                      param)
                for uri, state in drone_states.items()
            }
        )

        swarm.parallel_safe(
            stop_logs,
            args_dict={
                uri: (state,)
                for uri, state in drone_states.items()
            }
        )