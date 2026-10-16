import time
import threading
import numpy as np

from Structs import *
from helper_funcs import *

from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie

def uri_is_alive(uri, timeout=0.5):
    """
    Try connecting briefly to check if a Crazyflie exists.
    """
    try:
        cf = Crazyflie(rw_cache='./cache')

        with SyncCrazyflie(uri, cf=cf) as scf:
            # If we get here, connection succeeded
            return True

    except Exception:
        return False

def discover_active_uris():
    active = []
    for uri in CANDIDATE_URIS:
        print(f"Probing {uri} ...")
        if uri_is_alive(uri):
            print(f"  ✔ Found: {uri}")
            active.append(uri)
        else:
            print(f"  ✖ Not found: {uri}")
    return active

def stop_logs(scf, state: DroneState):
    state.log_conf.stop()
    
def get_position_callback(data, state: DroneState):
    state.position[0] = data['stateEstimate.x']
    state.position[1] = data['stateEstimate.y']
    state.position[2] = data['stateEstimate.z']

def init_position_getting(scf: SyncCrazyflie, state: DroneState):
    log_conf = LogConfig(name='Position', period_in_ms=500)
    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    log_conf.add_variable('stateEstimate.z', 'float')
    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(lambda _timestamp,
                    data,
                    _logconf: get_position_callback(data, state))
    log_conf.start()

    state.log_conf = log_conf
    
def light_check(scf: SyncCrazyflie):
    scf.cf.param.set_value('led.bitmask', 255)
    time.sleep(2)
    scf.cf.param.set_value('led.bitmask', 0)

def arm(scf: SyncCrazyflie):
    scf.cf.supervisor.send_arming_request(True)
    time.sleep(1.0)

def take_off(scf, state: DroneState, param: Param):
    cf = scf.cf
    deadline = time.time() + param.deadline_time
    while state.position[2] < param.default_height*0.9:
        if time.time() > deadline:
            land(scf, state, param)
            raise RuntimeError("Takeoff timeout")
        cf.commander.send_position_setpoint(state.position[0], state.position[1], param.default_height, 0)
        time.sleep(param.dTi)

def land(scf: SyncCrazyflie, state: DroneState, param: Param):
    cf = scf.cf
    x, y = state.position[0], state.position[1]

    deadline = time.time() + param.deadline_time
    while state.position[2] > 0.12 and time.time() < deadline:
        cf.commander.send_position_setpoint(x, y, 0.1, 0)
        time.sleep(param.dTi)

    cf.commander.send_position_setpoint(x, y, 0.0, 0)
    time.sleep(0.3)

    cf.commander.send_notify_setpoint_stop()

def update_virtual_target(
    target: DroneState,
    lock: threading.Lock,
    param: Param,
    stop_event: threading.Event
):
    start_time = time.time()

    while not stop_event.is_set():
        t = time.time() - start_time
        pos = param.ref_target_traj["pos"](t)
        with lock:
            target.position[:] = pos

        time.sleep(param.dTi)

# ============================================================
# Shared desired positions
# ============================================================
def controller(
    scf: SyncCrazyflie,
    state: DroneState,
    desired_lock: threading.Lock,
    desired_positions, 
    param: Param
):  
    """
    Runs continuously on each CF.
    """
    cf = scf.cf
    end_time = time.time() + param.Tend
    while time.time() < end_time:
        with desired_lock:
            dp = desired_positions["position"]
        if dp is None:
            print("Warning: dp is undefined")
            # hover in place
            cf.commander.send_position_setpoint(
                0,
                0,
                param.default_height,
                0.0
            )
            time.sleep(param.dTi)
            continue    

        cf.commander.send_position_setpoint(
            float(dp[0]),
            float(dp[1]),
            float(dp[2]),
            0.0
        )

        time.sleep(param.dTi)
  
# ============================================================
# update_desired_positions thread
# ============================================================
def update_desired_positions(
    drone_states: DroneState,
    target_lock: threading.Lock,
    target: DroneState,
    desired_lock: threading.Lock,
    desired_positions, 
    param: Param,
    stop_event: threading.Event
):
    # ========================================================
    # Initialization
    # ========================================================
    qp = initialize_qp(drone_states, target.position, param)

    W0, W1, W2 = qp["W0"], qp["W1"], qp["W2"]
    Q, q = qp["Q"], qp["q"]
    A_eq, A_obs = qp["A_eq"], qp["A_obs"]
    G, h = qp["G"], qp["h"]
    b_obs = qp["b_obs"]
    lambda_obs = qp["lambda_obs"]
    lambda_ineq = qp["lambda_ineq"]
    s = qp["s"]
    x = qp["x"]
    dim = qp["dim"]
    n_ctrl = qp["n_ctrl"]
    wN = W0[-1].reshape([-1,1])

    while not stop_event.is_set():
        with target_lock:
            goal = target.position.copy()

        # Update goal position
        for axis in range(3):
            idx = np.arange(axis, dim, 3)
            q[idx] = -param.w_goal * goal[axis] * wN

        # ----------------------------------------------------
        # Read current drone state
        # ----------------------------------------------------

        # Equality constraint
        b_eq = np.hstack((
            drone_states.position,
            drone_states.velocity,
            drone_states.acceleration
        )).reshape(-1, 1)

        # Reset multipliers
        lambda_obs[:] = 0
        lambda_ineq[:] = 0

        # ====================================================
        # Augmented Lagrangian iterations
        # ====================================================
        for iteration in range(param.max_iter):
            # ----------------------------------------------
            # Step 1: Solve QP
            # ----------------------------------------------
            A_bar = (
                Q
                + param.rho_obs * A_obs.T @ A_obs
                + param.rho_ineq * G.T @ G
            )

            b_bar = (
                - q
                - A_obs.T @ lambda_obs
                + param.rho_obs * A_obs.T @ b_obs
                - G.T @ lambda_ineq
                + param.rho_ineq * G.T @ (h - s)
            )

            KKT = np.block([
                [A_bar, A_eq.T],
                [A_eq, np.zeros((A_eq.shape[0], A_eq.shape[0]))]
            ])

            rhs = np.concatenate([
                b_bar,
                b_eq
            ])

            solution = np.linalg.solve(KKT, rhs)

            n = A_bar.shape[0]
            x = solution[:n]

            # ----------------------------------------------
            # Step 2: Recover trajectory
            # ----------------------------------------------

            C = x.flatten().reshape(n_ctrl, 3)
            traj = W0 @ C

            # ----------------------------------------------
            # Step 3: Update obstacle constraints
            # ----------------------------------------------
            b_obs = build_b_obs(param.H, param.obstacle_radius, param.obstacle_center, traj)

            # ----------------------------------------------
            # Step 4: Update slack variables
            # ----------------------------------------------
            s = np.maximum(
                0,
                h - G @ x - lambda_ineq / param.rho_ineq
            )

            # ----------------------------------------------
            # Step 5: Compute residuals
            # ----------------------------------------------
            residual_eq = A_eq @ x - b_eq
            residual_obs = A_obs @ x - b_obs
            residual_ineq = G @ x - h + s

            # ----------------------------------------------
            # Step 6: Update multipliers
            # ----------------------------------------------
            lambda_obs += param.rho_obs * residual_obs
            lambda_ineq += param.rho_ineq * residual_ineq

            if (
                np.linalg.norm(residual_eq)
                + np.linalg.norm(residual_obs)
                + np.linalg.norm(residual_ineq)
                < 1e-6
            ):
                break
                
        C = x.flatten().reshape(n_ctrl,3)
        traj = W0 @ C

        # ====================================================
        # Publish desired position
        # ====================================================
        with desired_lock:
            desired_positions["position"] = traj[-1].copy()

        time.sleep(param.dTi)