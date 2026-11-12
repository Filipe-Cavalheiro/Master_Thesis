"""
Swarm take off, hover above the takeoff points, return and land: the check
to run first with new drones, batteries or controller settings, before the
Voronoi formation. Self-contained: the Crazyflie/OptiTrack code it needs is
in this folder (no formation, no planner, no target).
"""

import sys
import time
import threading
import subprocess
import atexit
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from natnet import NatNetClient                        

import cflib.crtp                                      
from cflib.crazyflie.swarm import CachedCfFactory      
from cflib.crazyflie.swarm import Swarm                

from cf_funcs import *                                 
from natnet_funcs import *                             
from trajectory_logger import TrajectoryLogger
import ctrl_c
from camera_logger import CameraLogger         

HOVER_TIME = 8.0     # s, hover above the takeoff points

def hover(scf, state: DroneState, position_lock: threading.Lock, abort_event: threading.Event):
    """Hold above the takeoff point for HOVER_TIME."""
    hx, hy, _ = state.local_origin
    return ramp_to(scf, [hx, hy, TAKEOFF_HEIGHT], HOVER_TIME, "hover", state, position_lock, abort_event)


def assign_land_spots(drone_states: dict[str, DroneState], position_lock: threading.Lock):
    """
    Landing spot of every drone: one of the takeoff points (at takeoff
    height), minimizing the sum of squared distances. Returns the common
    return duration.
    """
    uris = list(drone_states)
    positions = np.array([get_position(drone_states[u], position_lock) for u in uris])
    homes = np.array([[*drone_states[u].local_origin[:2], TAKEOFF_HEIGHT] for u in uris])
    cost = np.sum((positions[:, None, :] - homes[None, :, :]) ** 2, axis=2)
    _, home_of = linear_sum_assignment(cost)
    for u, h in zip(uris, home_of):
        drone_states[u].land_spot = homes[h]
    dist = np.max(np.linalg.norm(positions - homes[home_of], axis=1))
    return max(1.5 * dist / RETURN_SPEED, 2.0)          # smoothstep peak = 1.5 x mean speed


def main():
    print("\n==============================================")
    print(" Crazyflie swarm take off / hover / land + OptiTrack")
    print("==============================================\n")
    print(f"  Candidate URIs : {CANDIDATE_URIS}")
    print(f"  Rigid bodies   : {RIGID_BODY_IDS}")
    print(f"  Motive / local : {MOTIVE_IP} / {LOCAL_IP}\n")

    subprocess.run("clear")
    cflib.crtp.init_drivers()

    factory = CachedCfFactory(rw_cache=CACHE_DIR)

    active_uris = discover_active_uris()
    if not active_uris:
        raise RuntimeError("No drones found")

    drone_states = {uri: DroneState() for uri in active_uris}

    stop_event = threading.Event()
    position_lock = threading.Lock()
    abort_event = threading.Event()
    ctrl_c.install(abort_event)   # 1st Ctrl+C: return home + land, 2nd: motors off
    extpos_threads = []

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
                              sample_period=0.02)

    # pose of the filming camera (phone), from the same NatNet frames (no second
    # client), from now on: the sync gestures before the take off are recorded too
    cam_log = None
    if CAMERA_RIGID_BODY_ID is not None:
        cam_log = CameraLogger(logger.output_path, CAMERA_RIGID_BODY_ID)
        client.on_data_frame_received_event.handlers.append(cam_log.on_frame)
        atexit.register(cam_log.close)

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
                print("test\n")
                swarm.parallel_safe(capture_home,
                    args_dict={uri: (state, position_lock) for uri, state in drone_states.items()}
                )
                print("test 2\n")

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
                if cam_log is not None:
                    cam_log.mark_flight_start(logger.start_time)
                logger.mark_state("Take off")

                swarm.parallel_safe(take_off,
                    args_dict={uri: (state, position_lock, abort_event) for uri, state in drone_states.items()}
                )

                if not abort_event.is_set():
                    logger.mark_state("Hover")
                    swarm.parallel_safe(hover,
                        args_dict={uri: (state, position_lock, abort_event) for uri, state in drone_states.items()}
                    )

            except KeyboardInterrupt:
                print("\nKeyboard interrupt received.")
                abort_event.set()
            except Exception as e:
                print(f"\nERROR DURING FLIGHT: {e}" + (f" ({e.__cause__!r})" if e.__cause__ else ""))
                abort_event.set()
            finally:
                # Always (also after an abort): return above a takeoff point,
                # land vertically, disarm
                if armed:
                    time.sleep(0.3)
                    try:
                        try:
                            logger.mark_state("Return to takeoff points")
                            return_time = assign_land_spots(drone_states, position_lock)
                            swarm.parallel_safe(return_origin,
                                args_dict={
                                    uri: (state, position_lock, abort_event, return_time)
                                    for uri, state in drone_states.items()
                                }
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
                for t in extpos_threads:
                    t.join(timeout=1.0)

            if not abort_event.is_set():
                print("\nHover test completed successfully.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("\nProgram terminated.")
