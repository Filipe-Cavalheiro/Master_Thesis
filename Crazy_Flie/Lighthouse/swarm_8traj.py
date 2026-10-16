import time
from dataclasses import dataclass, field
from typing import List
import math

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

@dataclass
class DroneState:
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    attitude: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # roll,pitch,yaw

# CONSTS
DEFAULT_HEIGHT = 0.5    # Default takeoff altitude (m)
DEFAULT_SPEED = 0.8    # Default speed (m/s)
URIS = [
    'radio://0/10/2M/E7E7E7E701',
    'radio://0/10/2M/E7E7E7E703',
    # 'radio://0/10/2M/E7E7E7E704'
    # Add more URIs if you want more copters in the swarm
    # URIs in a swarm using the same radio must also be on the same channel!!!
]

# Global vars
drone_states = {
    uri: DroneState()
    for uri in URIS
}

def get_position_callback(uri, data):
    state = drone_states[uri]

    state.position[0] = data['stateEstimate.x']
    state.position[1] = data['stateEstimate.y']
    state.position[2] = data['stateEstimate.z']

def init_position_getting(scf: SyncCrazyflie):
    log_conf = LogConfig(name='Position', period_in_ms=500)
    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    log_conf.add_variable('stateEstimate.z', 'float')
    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(lambda _timestamp,
                    data,
                    _logconf: get_position_callback(scf.cf.link_uri, data))
    log_conf.start()

def activate_led_bit_mask(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 255)

def deactivate_led_bit_mask(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 0)

def light_check(scf: SyncCrazyflie):
    activate_led_bit_mask(scf)
    time.sleep(2)
    deactivate_led_bit_mask(scf)

def arm(scf):
    scf.cf.supervisor.send_arming_request(True)
    time.sleep(1.0)

def take_off(scf, state: DroneState):
    cf = scf.cf
    while state.position[2] < DEFAULT_HEIGHT*0.9:
        cf.commander.send_position_setpoint(state.position[0], state.position[1], DEFAULT_HEIGHT, 0)
        time.sleep(0.1)

def land(scf: SyncCrazyflie, state: DroneState):
    cf = scf.cf
    x, y = state.position[0], state.position[1]

    while state.position[2] > 0.05:
        cf.commander.send_position_setpoint(x, y, 0.1, 0)
        time.sleep(0.1)

    cf.commander.send_position_setpoint(x, y, 0.0, 0)
    time.sleep(0.3)

    cf.commander.send_notify_setpoint_stop()
    
def move_8_traj(scf: SyncCrazyflie):
    cf = scf.cf
    dt = 0.05
    A = 2.0
    B = 1.5

    start_time = time.time()
    end_time = start_time + 12

    uri = scf.cf.link_uri
    iD = URIS.index(uri)

    while time.time() < end_time:
        t = time.time() - start_time

        match iD:
            case 0:
                x = A * math.sin(DEFAULT_SPEED * t)
                y = B * math.sin(2 * DEFAULT_SPEED * t)
                h = DEFAULT_HEIGHT

            case 1:
                x = B * math.cos(DEFAULT_SPEED * t)
                y = A * math.cos(2 * DEFAULT_SPEED * t)
                h = DEFAULT_HEIGHT + 0.5

            case 2:
                x = A * math.sin(DEFAULT_SPEED * t)
                y = B * math.sin(2 * DEFAULT_SPEED * t)
                h = DEFAULT_HEIGHT + 1.0

            case _:
                x = 0
                y = 0
                h = DEFAULT_HEIGHT

        cf.commander.send_position_setpoint(x, y, h, 0)
        time.sleep(dt)

if __name__ == '__main__':
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    
    with Swarm(URIS, factory=factory) as swarm:
        swarm.parallel_safe(light_check)
        swarm.reset_estimators()

        # store vars in Ground Control memory
        swarm.parallel_safe(init_position_getting)

        # arm vehicles
        swarm.parallel_safe(arm)

        print("Take off")
        swarm.parallel_safe(take_off, 
                            args_dict={uri: (state,) for uri, state in drone_states.items()})
        swarm.parallel_safe(move_8_traj)
        print("Sleepy time")
        swarm.parallel_safe(land, 
                            args_dict={uri: (state,) for uri, state in drone_states.items()})