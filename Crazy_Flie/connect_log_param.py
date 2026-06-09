import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

def simple_connect():
    print("Yeah, I'm connected! :D")
    time.sleep(3)
    print("Now I will disconnect :'(")

def simple_log(scf, logconf):
    # description: Logs the state of the cf synchronously
    # scf: Sync CrazyFlie
    # logconf: logging configuration
    with SyncLogger(scf, logconf) as logger:
        for log_entry in logger:

            timestamp = log_entry[0]
            data = log_entry[1]
            logconf_name = log_entry[2]

            print('[%d][%s]: %s' % (timestamp, logconf_name, data))

def log_stab_callback(timestamp, data, logconf):
    # description: async func callback for logging
    # timestamp: current since prog start
    # data: log data (available data at TOC table)
    # logconf: logging configuration
    print('[%d][%s]: %s' % (timestamp, logconf.name, data))

def simple_log_async(scf, logconf):
    # description: Logs the state of the cf asynchronously
    # scf: Sync CrazyFlie
    # logconf: logging configuration
    cf = scf.cf
    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(log_stab_callback)
    logconf.start()
    time.sleep(5)
    logconf.stop()

def param_stab_est_callback(name, value):
    print(f'{name} = {value}')

def simple_param_async(scf, groupstr, namestr):
    cf = scf.cf
    full_name = f"{groupstr}.{namestr}"

    # Register callback first
    # update callback only runs after an update of variable
    cf.param.add_update_callback(
        group=groupstr,
        name=namestr,
        cb=param_stab_est_callback
    )

    # Set to 1
    print("Setting estimator to 1")
    cf.param.set_value(full_name, "1")

    time.sleep(5)

    # Set to 2
    print("Setting estimator to 2")
    cf.param.set_value(full_name, "2")

    # Keep program alive to receive updates
    while True:
        time.sleep(0.1)

# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(
    default='radio://0/10/2M/E7E7E7E701'
)
logging.basicConfig(level=logging.ERROR)

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    # Define logging variables
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')

    group = "stabilizer"
    group_var = "estimator"

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        # simple_connect()
            
        # simple_log(scf, lg_stab)

        # simple_log_async(scf, lg_stab)

        simple_param_async(scf, group, group_var)
        
