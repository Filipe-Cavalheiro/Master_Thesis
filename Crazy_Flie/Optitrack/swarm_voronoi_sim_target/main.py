"""
Voronoi formation around a virtual target, with OptiTrack: the
lighthouse_swarm_voronoi_sim_target mission (target on a circle, formation
mesh + CVT, every drone following its site) with the OptiTrack external
position and the safety sequence of swarm_take_off_hover_land. Take off,
fly to the formation slots, follow the formation for FORMATION_TIME, return
above the takeoff points and land.
"""

import sys
import time
import threading
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from natnet import NatNetClient

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm

from cf_funcs import *
from natnet_funcs import *
from formation import Formation
from trajectory_logger import TrajectoryLogger
import ctrl_c

def main():
    subprocess.run("clear")
    print("\n==============================================")
    print(" Crazyflie swarm Voronoi formation + OptiTrack")
    print("==============================================\n")
    print(f"  Candidate URIs : {CANDIDATE_URIS}")
    print(f"  Rigid bodies   : {RIGID_BODY_IDS}")
    print(f"  Motive / local : {MOTIVE_IP} / {LOCAL_IP}\n")

    cflib.crtp.init_drivers()

    factory = CachedCfFactory(rw_cache=CACHE_DIR)

    active_uris = discover_active_uris()
    if not active_uris:
        raise RuntimeError("No drones found")

    drone_states = {uri: DroneState() for uri in active_uris}

    # Formation at the initial target position, converged before arming
    param = Param(number_of_drones=len(active_uris), traj=TARGET_TRAJ)
    formation = Formation(param)
    formation.converge()

    stop_event = threading.Event()
    formation_stop = threading.Event()
    position_lock = threading.Lock()
    abort_event = threading.Event()
    ctrl_c.install(abort_event)   # 1st Ctrl+C: return home + land, 2nd: motors off
    extpos_threads = []
    formation_threads = []

    client = NatNetClient(
        server_ip_address=MOTIVE_IP,
        local_ip_address=LOCAL_IP,
        use_multicast=False,
    )
    client.on_data_description_received_event.handlers.append(on_data_description)
    client.on_data_frame_received_event.handlers.append(
        lambda frame: on_data_frame(frame, drone_states, position_lock)
    )

    logger = TrajectoryLogger(drone_states, position_lock,
                              sample_period=0.02,
                              target_fn=formation.get_target)

    with client:
        client.request_modeldef()
        client.run_async()

        if not wait_for_tracking(5.0, drone_states, position_lock, active_uris):
            print("ABORT: no valid OptiTrack tracking.")
            return

        print("\nConnecting to Crazyflies...")
        with Swarm(active_uris, factory=factory) as swarm:
            print("Swarm connected.\n")
            stop_event.clear()
            abort_event.clear()
            armed = False

            try:
                swarm.parallel_safe(
                    start_extpos,
                    args_dict={
                        uri: (state, extpos_threads, stop_event, abort_event, position_lock)
                        for uri, state in drone_states.items()
                    },
                )
                swarm.parallel_safe(set_onboard_controller)
                swarm.reset_estimators()
                swarm.parallel_safe(capture_home,
                    args_dict={uri: (state, position_lock) for uri, state in drone_states.items()}
                )

                if not all_tracking_is_valid(drone_states, position_lock) or abort_event.is_set():
                    print("ERROR: tracking lost during estimator settling.")
                    return

                swarm.parallel_safe(check_battery)

                print("Lights On")
                ctrl_c.flying([scf.cf for scf in swarm._cfs.values()])
                swarm.parallel_safe(arm)
                armed = True
                print("Armed.")

                swarm.parallel_safe(warning_lights)

                swarm.parallel_safe(logger.start_logging)
                logger.begin()
                logger.mark_state("Take off")

                swarm.parallel_safe(take_off,
                    args_dict={uri: (state, position_lock, abort_event) for uri, state in drone_states.items()}
                )

                if not abort_event.is_set():
                    # Drone i of active_uris follows desired position i
                    positions = np.array([get_position(drone_states[u], position_lock) for u in active_uris])
                    slots = formation.assign_slots(positions)
                    dist = np.max(np.linalg.norm(slots - positions, axis=1))
                    duration = max(1.5 * dist / APPROACH_SPEED, 2.0)   # smoothstep peak = 1.5 x mean speed

                    logger.mark_state("Approach")
                    swarm.parallel_safe(go_to_slot,
                        args_dict={uri: (drone_states[uri], slots[i], duration, position_lock, abort_event)
                                   for i, uri in enumerate(active_uris)}
                    )

                if not abort_event.is_set():
                    logger.mark_state("Formation")
                    formation_threads = formation.start_threads(formation_stop)
                    swarm.parallel_safe(controller,
                        args_dict={uri: (i, drone_states[uri], formation, drone_states, position_lock, abort_event)
                                   for i, uri in enumerate(active_uris)}
                    )

            except KeyboardInterrupt:
                print("\nKeyboard interrupt received.")
                abort_event.set()
            except Exception as e:
                print(f"\nERROR DURING FLIGHT: {e}" + (f" ({e.__cause__!r})" if e.__cause__ else ""))
                abort_event.set()
            finally:
                # no join here: a CVT update in progress would delay the
                # return setpoints past the commander watchdog (0.5 s)
                formation_stop.set()

                # Always (also after an abort): return above a takeoff point,
                # land vertically, disarm
                if armed:
                    time.sleep(0.3)
                    try:
                        try:
                            logger.mark_state("Return")
                            duration = assign_land_spots(drone_states, position_lock)
                            swarm.parallel_safe(return_origin,
                                args_dict={uri: (state, position_lock, abort_event, duration)
                                           for uri, state in drone_states.items()}
                            )
                        except Exception as e:
                            print(f"ERROR DURING RETURN: {e}: landing in place.")
                        logger.mark_state("Land")
                        swarm.parallel_safe(land,
                            args_dict={uri: (state, position_lock, abort_event) for uri, state in drone_states.items()}
                        )
                        time.sleep(1.0)
                    finally:
                        logger.mark_state("EOF")
                        logger.close()
                        swarm.parallel_safe(logger.stop_logging)
                        print(f"Trajectory saved to: {logger.output_path}")

                        swarm.parallel_safe(stop_flight)
                        ctrl_c.landed()
                        time.sleep(1.0)

                stop_event.set()
                for t in extpos_threads + formation_threads:
                    t.join(timeout=1.0)

            if not abort_event.is_set():
                print("\nVoronoi formation flight completed successfully.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("\nProgram terminated.")
