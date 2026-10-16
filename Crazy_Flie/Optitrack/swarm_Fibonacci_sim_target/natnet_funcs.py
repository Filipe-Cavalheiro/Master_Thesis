from natnet import DataDescriptions, DataFrame, NatNetClient
import math
import time
import numpy as np
import threading

from Structs import ID_TO_URI, DroneState

# ============================================================
# NATNET
# ============================================================

def on_data_description(desc: DataDescriptions):
    print("Received NatNet data descriptions.")

def quaternion_to_euler(x, y, z, w):
    """Quaternion (x, y, z, w) -> (roll, pitch, yaw) in radians."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw

def on_data_frame(
    frame: DataFrame,
    drone_states: dict[str, DroneState],
    position_lock: threading.Lock,
):
    """Store the latest pose of every configured rigid body."""
    for rb in frame.rigid_bodies:
        uri = ID_TO_URI.get(rb.id_num)
        if uri is None or not rb.tracking_valid:
            continue

        mx, my, mz = rb.pos
        qx, qy, qz, qw = rb.rot

        # Motive (Y-up) -> Crazyflie (Z-up): (x, y, z) = (Zm, Xm, Ym).
        roll, pitch, yaw = quaternion_to_euler(qz, qx, qy, qw)

        with position_lock:
            st = drone_states.get(uri)
            if st is None:
                continue

            st.position = np.array([mz, mx, my])
            st.attitude = np.array([roll, pitch, yaw])
            st.last_update = time.monotonic()