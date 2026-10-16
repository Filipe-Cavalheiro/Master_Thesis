import time

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

DEFAULT_HEIGHT = 0.5    # Default takeoff altitude (m)
DEFAULT_VELOCITY = 0.2  # Default takeoff altitude (m/s)
URIS = [
    'radio://0/10/2M/E7E7E7E701',
    'radio://0/10/2M/E7E7E7E703',
    'radio://0/10/2M/E7E7E7E704'
    # Add more URIs if you want more copters in the swarm
    # URIs in a swarm using the same radio must also be on the same channel!!!
]

def activate_led_bit_mask(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 255)

def deactivate_led_bit_mask(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 0)

def light_check(scf: SyncCrazyflie):
    print("light on!")
    activate_led_bit_mask(scf)
    time.sleep(2)
    print("light off!")
    deactivate_led_bit_mask(scf)

def arm(scf):
    scf.cf.platform.send_arming_request(True)
    time.sleep(1.0)

def take_off(scf: SyncCrazyflie):
    commander= scf.cf.high_level_commander

    commander.takeoff(1.0, 2.0)
    time.sleep(3)

def land(scf: SyncCrazyflie):
    commander= scf.cf.high_level_commander

    commander.land(0.0, 2.0)
    time.sleep(2)

    commander.stop()

def run_square_sequence(scf: SyncCrazyflie):
    box_size = 1
    flight_time = 2

    commander= scf.cf.high_level_commander

    commander.go_to(box_size, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(0, box_size, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(-box_size, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(0, -box_size, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

# The layout of the positions (1m apart from each other):
#   <------ 1 m ----->
#                    CF4
#            ^           ^
#            |X          |
#            |           |
#      Y<----+           1 m
#                        |
#                        |
#                        v
#   CF1              CF3     .
x0, y0 = +1.0, +1.0
x1, y1 = -1.0, -1.0

#    x   y   z  time
sequence0 = [
    (x1, y0, DEFAULT_HEIGHT, 3.0),
    (x1, y1, DEFAULT_HEIGHT, 3.0),
    (x0, y1, DEFAULT_HEIGHT, 3.0),
    (x0, y0, DEFAULT_HEIGHT, 3.0),
    (x1, y0, DEFAULT_HEIGHT, 3.0)
]

sequence1 = [
    (x1, y1, DEFAULT_HEIGHT, 3.0),
    (x0, y1, DEFAULT_HEIGHT, 3.0),
    (x0, y0, DEFAULT_HEIGHT, 3.0),
    (x1, y0, DEFAULT_HEIGHT, 3.0),
    (x1, y1, DEFAULT_HEIGHT, 3.0)
]
sequence2 = [
    (x0, y1, DEFAULT_HEIGHT, 3.0),
    (x0, y0, DEFAULT_HEIGHT, 3.0),
    (x1, y0, DEFAULT_HEIGHT, 3.0),
    (x1, y1, DEFAULT_HEIGHT, 3.0),
    (x0, y1, DEFAULT_HEIGHT, 3.0)
]

seq_args = {
    URIS[0]: [sequence0],
    URIS[1]: [sequence1],
    URIS[2]: [sequence2]
}

def run_sequence(scf: SyncCrazyflie, sequence):
    cf = scf.cf

    for arguments in sequence:
        commander = scf.cf.high_level_commander

        x, y, z = arguments[0], arguments[1], arguments[2]
        duration = arguments[3]

        print('Setting position {} to cf {}'.format((x, y, z), cf.link_uri))
        commander.go_to(x, y, z, 0, duration, relative=False)
        time.sleep(duration)

if __name__ == '__main__':
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(URIS, factory=factory) as swarm:
        swarm.parallel_safe(light_check)
        swarm.reset_estimators()

        print("Take off")
        swarm.parallel_safe(take_off)
        print("Square Up Gang")
        # swarm.parallel_safe(run_square_sequence)
        swarm.parallel_safe(run_sequence, args_dict=seq_args)
        print("Sleepy time")
        swarm.parallel_safe(land)
