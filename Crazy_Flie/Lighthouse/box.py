import logging
import sys
import time
from threading import Event
import os
import sys

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig


# === Constants ===
# URI of the Crazyflie radio link.
URI = uri_helper.uri_from_env(default='radio://0/10/2M/E7E7E7E702')

# Default takeoff altitude in meters.
DEFAULT_HEIGHT = 0.5

# Maximum allowed X/Y-axis displacement before reversing direction.
BOX_LIMIT = 2.3

VERBOSE = True

# === Global vars ===
# Event used to signal successful detection of the Lighthouse deck.
deck_attached_event = Event()

# Current estimated position [x, y] in meters.
position_estimate = [0, 0, 0]
attitude_estimate = [0]

logging.basicConfig(level=logging.ERROR)

def param_deck_check(_, value_str):
    """
    Callback function triggered when the Lighthouse deck parameter is updated.

    Args:
        _ : str
            Parameter name (unused in this callback).
        value_str : str
            String representation of the deck status.
            '1' indicates the Lighthouse deck is attached,
            '0' indicates it is not attached.

    Side Effects:
        - Sets the global `deck_attached_event` when the deck is detected.
        - Prints deck attachment status to the console.
    """
    value = int(value_str)
    if value:
        deck_attached_event.set()
        print('Lighthouse Deck is attached!')
    else:
        print('Lighthouse Deck is NOT attached!')

def move_linear_simple(scf):
    """
    Executes a simple forward-and-back movement sequence.

    Args:
        scf (SyncCrazyflie):
            Active Crazyflie connection object.

    Behavior:
        - Takes off to DEFAULT_HEIGHT.
        - Moves forward 1 meter.
        - Returns backward 1 meter to approximately
          the starting position.
        - Lands automatically when MotionCommander exits.
    """
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(1)
        mc.forward(1)
        time.sleep(1)
        mc.back(1)
        time.sleep(1)

def move_front_turn_back(scf):
    """
    Moves forward, turns around, returns to the starting area,
    and restores the original heading.

    Args:
        scf (SyncCrazyflie):
            Active Crazyflie connection object.

    Behavior:
        - Moves forward 1 meter.
        - Rotates 180 degrees.
        - Moves forward 1 meter (returning toward the start).
        - Rotates another 180 degrees to face the original direction.
        - Lands automatically when MotionCommander exits.
    """
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(1)
        mc.forward(1)
        time.sleep(1)
        mc.turn_left(180)
        time.sleep(1)
        mc.forward(1)
        time.sleep(1)
        mc.turn_left(180)
        time.sleep(1)

def log_pos_callback(timestamp, data, logconf):
    """
    Callback function for position logging updates.

    Args:
        timestamp (int):
            Time when the log sample was generated (milliseconds).
        data (dict):
            Dictionary containing logged variables.
            Expected keys:
                - 'stateEstimate.x' : Estimated X position (meters)
                - 'stateEstimate.y' : Estimated Y position (meters)
        logconf (LogConfig):
            Log configuration object that generated this callback.

    Side Effects:
        - Updates the global `position_estimate` array.
        - Prints the received log data.
    """
    if VERBOSE:
        print(data)
        
    global position_estimate, attitude_estimate
    position_estimate[0] = data['stateEstimate.x']
    position_estimate[1] = data['stateEstimate.y']
    position_estimate[2] = data['stateEstimate.z']
    attitude_estimate[0] = data['stateEstimate.yaw']

def move_box_limit(scf):
    """
    Continuously moves the Crazyflie back and forth between
    predefined X-axis boundaries.

    Args:
        scf (SyncCrazyflie):
            Active Crazyflie connection object.

    Behavior:
        - Takes off to DEFAULT_HEIGHT.
        - Starts moving forward.
        - Reverses direction when the estimated X position
          exceeds BOX_LIMIT or goes below -BOX_LIMIT.
        - Runs indefinitely until externally interrupted.
    """
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        body_x_cmd = 0.2
        body_y_cmd = 0.2
        max_vel = 0.2
        
        for i in range(100):
            if position_estimate[0] > BOX_LIMIT:
                body_x_cmd = -max_vel
            elif position_estimate[0] < -BOX_LIMIT:
                body_x_cmd = max_vel

            if position_estimate[1] > BOX_LIMIT:
                body_y_cmd = -max_vel
            elif position_estimate[1] < -BOX_LIMIT:
                body_y_cmd = max_vel

            mc.start_linear_motion(body_x_cmd, body_y_cmd, 0)
            time.sleep(0.1)
        
        mc.stop()
    print("Returning home...")
    return_home(scf)
    print("Landing...")

def return_home(scf):
    cf = scf.cf
    cf.commander.send_position_setpoint(0,0,DEFAULT_HEIGHT,0)
    time.sleep(5)
    
if __name__ == '__main__':
    os.system('clear')

    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.add_update_callback(group="deck", name="bcLighthouse4",
                               cb=param_deck_check)
        time.sleep(1)

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        logconf.add_variable('stateEstimate.z', 'float')
        logconf.add_variable('stateEstimate.yaw', 'float')
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        if not deck_attached_event.wait(timeout=5):
            print('No Lighthouse deck detected!')
            sys.exit(1)

        logconf.start()
        # return_home(scf)
        # Arm the Crazyflie
        scf.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

        move_box_limit(scf)

        logconf.stop()
        