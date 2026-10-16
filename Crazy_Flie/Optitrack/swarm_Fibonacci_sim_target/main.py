"""
Implementation of the centralized 
fibonacci sphere + lyapunov controller (path planner).
"""

import threading
import subprocess
import numpy as np

from natnet import NatNetClient

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm

from cf_funcs import *
from natnet_funcs import *
from trajectory_logger import TrajectoryLogger

def main():
    print("\n==============================================")
    print(" Crazyflie swarm + OptiTrack (Kalman, ext. position)")
    print("==============================================\n")
    print(f"  Candidate URIs : {CANDIDATE_URIS}")
    print(f"  Rigid bodies   : {RIGID_BODY_IDS}")
    print(f"  Motive / local : {MOTIVE_IP} / {LOCAL_IP}\n")

    subprocess.run("clear")
    cflib.crtp.init_drivers()

    factory = CachedCfFactory(rw_cache='./cache')

    # Discover existing drones
    active_uris = discover_active_uris()
    if not active_uris:
        raise RuntimeError("No drones found")

    param = Param(number_of_drones=len(active_uris), traj=2)

    # init drones
    drone_states = {uri: DroneState(param)for uri in active_uris}
    target = DroneState(param)

    target_lock = threading.Lock()
    chaser_list_ref_lock = threading.Lock()
    stop_event = threading.Event()
    position_lock = threading.Lock()
    abort_event = threading.Event()      # aborts the flight phases
    extpos_threads = []

    client = NatNetClient(
        server_ip_address=MOTIVE_IP,
        local_ip_address=LOCAL_IP,
        use_multicast=False,
    )
    client.on_data_description_received_event.handlers.append(on_data_description)
    client.on_data_frame_received_event.handlers.append(
        lambda frame: on_data_frame(
            frame,
            drone_states,
            position_lock
        )
    )

    # --------------------------------------------------------
    # Target thread
    # --------------------------------------------------------
    target_thread = threading.Thread(
        target=update_virtual_target,
        args=(target, target_lock, param, stop_event),
        daemon=True
    )

    fib_ref_points = get_fibonacci_points(param.nD, 
                                            param.sphere_radius, 
                                            np.zeros([3,1]))
    min_safe_dist = get_min_safe_distance(fib_ref_points, param) * 0.5

    q_ref_list = []
    for drone in drone_states.values():
        q_s = drone.position - target.position
        norm = np.linalg.norm(q_s)
        if norm > 1e-6:
            q_ref_list.append(q_s / norm)
        else:
            # choose a fallback direction
            q_ref_list.append(np.array([[1.0], [0.0], [0.0]]))

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------
    logger = TrajectoryLogger(drone_states, position_lock,
                            output_dir="Optitrack/logs",
                            sample_period=0.02)
    
    # --------------------------------------------------------
    # Crazyflies
    # --------------------------------------------------------
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
                        uri: (
                            state,
                            extpos_threads,
                            stop_event,
                            abort_event,
                            position_lock,
                        )
                        for uri, state in drone_states.items()
                    },
                )
                swarm.reset_estimators()
                swarm.parallel_safe(capture_home,
                    args_dict={
                        uri: (state, 
                            position_lock)
                        for uri, state in drone_states.items()
                    }
                )

                if not all_tracking_is_valid(drone_states, position_lock) or abort_event.is_set():
                    print("ERROR: tracking lost during estimator settling.")
                    return

                print("Lights On")
                swarm.parallel_safe(arm)
                armed = True
                print("Armed.")
        
                swarm.parallel_safe(warning_lights)


                swarm.parallel_safe(logger.start_logging)   # per-drone log blocks
                logger.begin()                              # single sampling thread + CSV
                logger.mark_state("Take off")

                swarm.parallel_safe(
                    take_off,
                    args_dict={
                        uri: (state, position_lock, abort_event, 
                            param)
                        for uri, state in drone_states.items()
                    }
                )

                logger.mark_state("Main Controller")
                # start the moving target only once everyone is on the sphere
                if not abort_event.is_set():
                    target_thread.start()
                    swarm.parallel_safe(
                        controller,
                        args_dict={
                            uri: (
                                target_lock,
                                target,
                                chaser_list_ref_lock,
                                q_ref_list, 
                                fib_ref_points, 
                                min_safe_dist, 
                                active_uris,
                                state,
                                param
                            )
                            for uri, state in drone_states.items()
                        }
                    )

                logger.mark_state("Return to (local) origin")
                swarm.parallel_safe(return_origin,
                    args_dict={
                        uri: (
                            state, 
                            position_lock, 
                            abort_event,
                            param
                        )
                        for uri, state in drone_states.items()
                    }
                )

            except KeyboardInterrupt:
                print("\nKeyboard interrupt received.")
                abort_event.set()
            except Exception as e:
                print(f"\nERROR DURING FLIGHT: {e}")
                abort_event.set()
            finally:
                # Always land (per drone, as long as it is still tracked) and disarm.
                if armed:
                    time.sleep(0.3)   # let aborted phases exit
                    try:
                        logger.mark_state("Land")
                        swarm.parallel_safe(land,
                            args_dict={
                                uri: (
                                    state, 
                                    position_lock, 
                                    abort_event,
                                )
                                for uri, state in drone_states.items()
                            }
                        )
                        time.sleep(1.0)
                    finally:
                        logger.mark_state("EOF")
                        logger.close()                              # stop writing first
                        swarm.parallel_safe(logger.stop_logging)    # then tear down log blocks (link still open)
                        print(f"Trajectory saved to: {logger.output_path}")

                        swarm.parallel_safe(stop_flight)
                        time.sleep(1.0)

                stop_event.set()
                for t in extpos_threads:
                    t.join(timeout=1.0)
                if target_thread.is_alive():
                    target_thread.join(timeout=1.0)

            if not abort_event.is_set():
                print("\nFlight test completed successfully.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("\nProgram terminated.")