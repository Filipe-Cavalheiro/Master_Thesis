"""
Voronoi formation around a target with the MPC of the Python simulation,
with OptiTrack: the drones reach their Voronoi goals through the
distributed MPC (mpc.py: Bernstein horizon, ADMM, inter-drone and geofence
constraints, velocity/acceleration limits). Take off, fly to the formation
slots, follow the formation for FORMATION_TIME, return above the takeoff
points and land.

    python main.py

The target is set by REAL_TARGET_DRONE_ID (Structs.py): -1 for a simulated
(virtual) target, the formation follows the path; a rigid-body ID: that
Crazyflie flies the path (to the start of it in the approach), the formation
follows its OptiTrack position and velocity and the chasers' MPC avoids it
as the other drones.

Log: Results/swarm_MPC_voronoi_<sim|real>_target_trajectory_<date>/.
"""

import argparse
import os
import sys
import time
import threading
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
# the target comes from REAL_TARGET_DRONE_ID only (the replay overrides are not for flights)
os.environ.pop("CF_TARGET", None)
os.environ.pop("CF_TARGET_DRONE_ID", None)

from natnet import NatNetClient

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm

from cf_funcs import *
from natnet_funcs import *
from formation import Formation, target_reference
from mpc import SwarmMPC
from trajectory_logger import TrajectoryLogger
import ctrl_c
import atexit
from camera_logger import CameraLogger

def main():
    subprocess.run("clear")
    print("\n==============================================")
    print(" Crazyflie swarm Voronoi formation + MPC + OptiTrack")
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
    position_lock = threading.Lock()

    # The target drone (if any) and the chasers
    if TARGET_URI is not None and TARGET_URI not in active_uris:
        raise RuntimeError(f"Target drone {TARGET_URI} (TARGET_DRONE_ID = {TARGET_DRONE_ID}) not found")
    chaser_uris = [u for u in active_uris if u != TARGET_URI]
    print(f"  Target         : {TARGET_URI or 'virtual'}\n  Chasers        : {chaser_uris}\n")

    # Formation at the initial target position, converged before arming
    param = Param(number_of_drones=len(chaser_uris), traj=TARGET_TRAJ)
    formation = Formation(param, drone_states.get(TARGET_URI), position_lock)
    formation.converge()
    mpc = SwarmMPC(len(chaser_uris), with_target=TARGET_URI is not None)
    target_start = target_reference(param, 0.0)[0]

    stop_event = threading.Event()
    formation_stop = threading.Event()
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
                              run_name=f"{HERE.name}_{TARGET_MODE}_target",   # Results/swarm_MPC_voronoi_sim_target_trajectory_<date>/
                              sample_period=0.02,
                              target_fn=formation.get_target)
    set_setpoint_logger(logger)     # every setpoint sent goes to <log>_setpoints.csv

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
                swarm.parallel_safe(capture_home,
                    args_dict={uri: (state, position_lock) for uri, state in drone_states.items()}
                )

                if not all_tracking_is_valid(drone_states, position_lock) or abort_event.is_set():
                    print("ERROR: tracking lost during estimator settling.")
                    return

                swarm.parallel_safe(check_battery)

                # what the firmware flies with (raises unless Kalman + Mellinger: no arming)
                firmware_params = {}
                swarm.parallel_safe(read_firmware_params,
                    args_dict={uri: (firmware_params,) for uri in active_uris})

                # Chaser i of chaser_uris follows desired position i: slots
                # assigned from the takeoff points (vertical takeoff), then
                # moved to where the MPC holds them apart (the goals can be stacked)
                homes = np.array([[*drone_states[u].local_origin[:2], TAKEOFF_HEIGHT] for u in chaser_uris])
                goals = formation.assign_slots(homes)
                slots = mpc.settle(goals, goals, target=target_start if TARGET_URI else None)
                print("Formation slots:\n" + "\n".join(
                    f"  {u}: {np.round(s_, 2)}" for u, s_ in zip(chaser_uris, slots)))
                if TARGET_URI is not None:
                    print(f"  {TARGET_URI} (target): {np.round(target_start, 2)}")
                meta_path = write_meta(logger.output_path, flight_metadata(
                    param, drone_states, chaser_uris, firmware_params, goals, slots, target_start))
                print(f"Flight setup saved to: {meta_path}")

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
                    # the chasers to their slots, the target to the start of its path
                    dest = {u: slots[i] for i, u in enumerate(chaser_uris)}
                    if TARGET_URI is not None:
                        dest[TARGET_URI] = target_start
                    dist = max(np.linalg.norm(dest[u] - get_position(drone_states[u], position_lock))
                               for u in active_uris)
                    duration = max(1.5 * dist / APPROACH_SPEED, 2.0)   # smoothstep peak = 1.5 x mean speed

                    logger.mark_state("Approach")
                    swarm.parallel_safe(go_to_slot,
                        args_dict={uri: (drone_states[uri], dest[uri], duration, position_lock, abort_event)
                                   for uri in active_uris}
                    )

                if not abort_event.is_set():
                    logger.mark_state("Formation")
                    formation_threads = formation.start_threads(formation_stop)
                    states = [drone_states[u] for u in chaser_uris]
                    mpc.plan_from(states, formation, position_lock)      # first plan before the controllers
                    formation_threads.append(
                        mpc.start_thread(states, formation, position_lock, formation_stop, abort_event))
                    roles = {uri: ("chaser", (i, drone_states[uri], mpc, drone_states, position_lock, abort_event))
                             for i, uri in enumerate(chaser_uris)}
                    if TARGET_URI is not None:
                        roles[TARGET_URI] = ("target", (drone_states[TARGET_URI], formation, drone_states,
                                                        position_lock, abort_event))
                    swarm.parallel_safe(formation_controller, args_dict=roles)

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
                print("\nVoronoi formation + MPC flight completed successfully.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        print("\nProgram terminated.")
