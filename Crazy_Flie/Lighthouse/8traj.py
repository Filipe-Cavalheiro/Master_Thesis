import logging
import sys
import time
from threading import Event
import math 
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

def param_deck_flow(_, value_str):
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

def setup_flight(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

def take_off(scf):
    print("Take off")
    cf = scf.cf
    for _ in range(40):
        cf.commander.send_position_setpoint(0, 0, DEFAULT_HEIGHT, 0)
        time.sleep(0.1)

def move_8_traj(scf):
    cf = scf.cf
    dt = 0.05
    A = 1.5
    B = 1.0
    w = 0.8

    for i in range(500):
        x = A * math.sin(w * i * dt)
        y = B * math.sin(2 * w * i * dt)
        cf.commander.send_position_setpoint(x,\
                                            y,\
                                            DEFAULT_HEIGHT,\
                                            0)
        time.sleep(dt)
    
def return_home(scf):
    print("Returning home...")
    cf = scf.cf
    for _ in range(50):
        cf.commander.send_position_setpoint(0,0,DEFAULT_HEIGHT,0)
        time.sleep(0.1)
    print("Landing...")
    for z in [0.4, 0.3, 0.2, 0.1, 0.0]:
        cf.commander.send_position_setpoint(0, 0, z, 0)
        time.sleep(0.6)
    cf.commander.send_notify_setpoint_stop()

if __name__ == '__main__':
    os.system('clear')

    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.add_update_callback(group="deck", name="bcLighthouse4",
                               cb=param_deck_flow)
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

        setup_flight(scf)

        logconf.start()

        take_off(scf)

        move_8_traj(scf)

        return_home(scf)
        
        logconf.stop()


        