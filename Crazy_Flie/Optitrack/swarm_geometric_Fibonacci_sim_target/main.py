"""
Implementation of the centralized 
fibonacci sphere + lyapunov controller (path planner).
"""

import time
import atexit
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
import ctrl_c
from camera_logger import CameraLogger

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
    ctrl_c.install(abort_event)   # 1st Ctrl+C: return home + land, 2nd: motors off
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

    # guidance state (q_ref, radius): set by init_guidance once the homes are known
    q_ref_list = []
    firmware_params = {}

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------
    logger = TrajectoryLogger(drone_states, position_lock,
                            sample_period=LOG_SAMPLE_PERIOD)
    set_setpoint_logger(logger)     # every setpoint sent goes to <log>_setpoints.csv

    # pose of the filming camera (phone), from the same NatNet frames (no second
    # client), from now on: the sync gestures before the take off are recorded too
    cam_log = None
    if CAMERA_RIGID_BODY_ID is not None:
        cam_log = CameraLogger(logger.output_path, CAMERA_RIGID_BODY_ID)
        client.on_data_frame_received_event.handlers.append(cam_log.on_frame)
        atexit.register(cam_log.close)
    
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
                # Kalman estimator + Mellinger before the reset (the reset is of the Kalman filter)
                swarm.parallel_safe(set_mellinger)
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

                # read back what the firmware has (raises, so no arming, unless Kalman + Mellinger)
                swarm.parallel_safe(read_firmware_params,
                    args_dict={uri: (firmware_params,) for uri in active_uris})

                # Guidance state from the takeoff points (not from the
                # positions before the OptiTrack stream started)
                q_ref_list[:] = init_guidance(drone_states, active_uris, param)
                logger.write_meta(flight_metadata(param, active_uris, drone_states, firmware_params,
                                                  fib_ref_points, min_safe_dist, q_ref_list))
                print(f"Flight setup saved to: {logger.meta_path}")

                print("Lights On")
                ctrl_c.flying([scf.cf for scf in swarm._cfs.values()])
                swarm.parallel_safe(arm)
                armed = True
                print("Armed.")
        
                swarm.parallel_safe(warning_lights)


                swarm.parallel_safe(logger.start_logging)   # per-drone log blocks
                logger.begin()                              # single sampling thread + CSV
                if cam_log is not None:
                    cam_log.mark_flight_start(logger.start_time)
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
                                param,
                                abort_event,
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
                # Always (also after an abort / 1st Ctrl+C): return above the
                # local home, land (per drone, as long as it is still tracked), disarm.
                if armed:
                    time.sleep(0.3)   # let aborted phases exit
                    try:
                        try:
                            logger.mark_state("Return to (local) origin")
                            swarm.parallel_safe(return_origin,
                                args_dict={
                                    uri: (state, position_lock, abort_event, param)
                                    for uri, state in drone_states.items()
                                }
                            )
                        except Exception as e:
                            print(f"ERROR DURING RETURN: {e}: landing in place.")
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
                        ctrl_c.landed()
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