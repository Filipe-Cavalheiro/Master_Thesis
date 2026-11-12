"""
Two-stage Ctrl+C for the Crazyflie flights.

    1st Ctrl+C (in flight): ABORT -> the abort event is set: the flight
                            phases stop, the drones return above their local
                            home and land (the script's normal abort path).
    2nd Ctrl+C:             KILL  -> stop setpoint + disarm sent to every
                            drone at once: motors off, the drones fall.

Outside a flight (before flying() / after landed()) Ctrl+C raises
KeyboardInterrupt as usual.

After a kill every commander setpoint (position, velocity, full state, ...)
from any thread is dropped, so no flight loop still running can spin the
motors up again.

Usage:
    import ctrl_c
    ctrl_c.install(abort_event)          # once, at start-up
    ctrl_c.flying([scf.cf, ...])         # right before arming
    ...                                  # loops: stop on abort_event / ctrl_c.killed()
    ctrl_c.landed()                      # after stop_flight / disarm
"""

import functools
import signal
import threading
import time

from cflib.crazyflie.commander import Commander

_kill_event = threading.Event()
_abort_event = None
_cfs = []
_flying = False
_presses = 0

# Commander methods that (re)start the motors: dropped after a kill
_SETPOINT_METHODS = (
    "send_setpoint",
    "send_setpoint_manual",
    "send_velocity_world_setpoint",
    "send_zdistance_setpoint",
    "send_hover_setpoint",
    "send_full_state_setpoint",
    "send_position_setpoint",
)


def _guard(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if _kill_event.is_set():
            return None
        return method(self, *args, **kwargs)
    wrapper._ctrl_c_guarded = True
    return wrapper


for _name in _SETPOINT_METHODS:
    _m = getattr(Commander, _name, None)
    if _m is not None and not getattr(_m, "_ctrl_c_guarded", False):
        setattr(Commander, _name, _guard(_m))


def install(abort_event: threading.Event):
    """Installs the SIGINT handler (main thread only); abort_event is set on the 1st Ctrl+C."""
    global _abort_event
    _abort_event = abort_event
    signal.signal(signal.SIGINT, _on_sigint)


def flying(cfs):
    """From now on Ctrl+C means abort / kill for these Crazyflies (call right before arming)."""
    global _flying, _presses
    _cfs[:] = list(cfs)
    _presses = 0
    _kill_event.clear()
    _flying = True


def landed():
    """Back to the default Ctrl+C (KeyboardInterrupt)."""
    global _flying
    _flying = False


def killed() -> bool:
    return _kill_event.is_set()


def kill():
    """Motors off on every drone, now (also usable from code)."""
    _kill_event.set()
    if _abort_event is not None:
        _abort_event.set()
    _stop_motors()
    # repeat for a moment: a setpoint already queued by a flight thread
    # just before the kill would otherwise re-enable the motors
    threading.Thread(target=_stop_motors_repeat, daemon=True).start()


def _stop_motors():
    for cf in list(_cfs):
        try:
            cf.commander.send_stop_setpoint()
            cf.supervisor.send_arming_request(False)
        except Exception as e:
            print(f"KILL: could not stop {getattr(cf, 'link_uri', cf)}: {e}", flush=True)


def _stop_motors_repeat(n=5, period=0.1):
    for _ in range(n):
        time.sleep(period)
        _stop_motors()


def _on_sigint(signum, frame):
    global _presses
    if not _flying:
        raise KeyboardInterrupt
    _presses += 1
    if _presses == 1:
        print("\n[Ctrl+C] ABORT: returning to the local home and landing. "
              "Press Ctrl+C again to cut the motors.", flush=True)
        if _abort_event is not None:
            _abort_event.set()
    else:
        print("\n[Ctrl+C x2] KILL: motors off, the drones will fall.", flush=True)
        kill()
