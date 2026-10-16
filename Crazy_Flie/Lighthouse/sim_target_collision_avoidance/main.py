import threading
import os

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from cf_funcs import *

if __name__ == "__main__":
    os.system("clear")
    cflib.crtp.init_drivers()

    active_uris = discover_active_uris()

    if not active_uris:
        raise RuntimeError("No drones found")

    # Use only the first discovered drone
    uri = active_uris[0]

    param = Param(number_of_drones=1, traj=2)

    drone_state = DroneState()
    target = DroneState()

    stop_event = threading.Event()
    target_lock = threading.Lock()
    desired_lock = threading.Lock()

    desired_positions = {
        "position": None
    }

    # --------------------------------------------------------
    # Target thread
    # --------------------------------------------------------
    target_thread = threading.Thread(
        target=update_virtual_target,
        args=(target, target_lock, param, stop_event),
        daemon=True,
    )

    desired_thread = threading.Thread(
        target=update_desired_positions,
        args=(
            drone_state,
            target_lock,
            target,
            desired_lock,
            desired_positions,
            param,
            stop_event,
        ),
        daemon=True,
    )

    cf = Crazyflie(rw_cache="./cache")

    with SyncCrazyflie(uri, cf=cf) as scf:
        print("Lights On")
        light_check(scf)

        target_thread.start()

        # --------------------------------------------------------
        # Desired position thread
        # --------------------------------------------------------
        desired_thread.start()

        init_position_getting(scf, drone_state)

        arm(scf)

        take_off(scf, drone_state, param)

        controller(
            scf,
            drone_state,
            desired_lock,
            desired_positions,
            param,
        )

        stop_event.set()

        land(scf, drone_state, param)

        stop_logs(scf, drone_state)