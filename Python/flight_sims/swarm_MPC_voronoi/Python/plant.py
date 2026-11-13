import copy
from collections import deque

import numpy as np

from actuate_control import actuate_control, actuate_passive
from Structs_sim import Param
from Structs_agent import DroneState

class PythonPlant:
    """
    Drone "plant" of the Python simulation: the controller output is
    integrated with the 3D quadrotor model right away (actuate_control).

    main() only talks to the drones through this interface, so another plant
    (e.g. Gazebo/sim/gazebo_plant.py) can replace the dynamics while the rest
    of the simulation stays the same:

        start(chasers, target)   once, after initialization
        actuate(agent)           active drone / target: apply the controller
        actuate_passive(agent)   dropped drone: motors off
        advance(chasers, target) end of the step: propagate the dynamics
        stop()                   end of the simulation
    """

    def __init__(self, param: Param):
        self.param = param

    def start(self, chaser_list: list[DroneState], target: DroneState):
        pass

    def actuate(self, agent: DroneState):
        return actuate_control(agent, self.param)

    def actuate_passive(self, agent: DroneState):
        return actuate_passive(agent, self.param)

    def advance(self, chaser_list: list[DroneState], target: DroneState):
        pass

    def stop(self):
        pass


class CrazyfliePlant(PythonPlant):
    """
    Crazyflie flying on its onboard controller, as in the OptiTrack flights:
    the setpoint (agent.desired position/velocity/acceleration) reaches the
    onboard Mellinger after `delay` (radio + Kalman), the position loop runs
    with the firmware gains (param.controller.Mellinger, set by
    experiment_arena.CRAZYFLIE_ONBOARD) at `rate` Hz, and the drone hovers
    `z_offset` below its setpoint (thrust mapping / integrator in the firmware).

    setpoint_mode "position" drops the setpoint velocity and acceleration
    (cf.commander.send_position_setpoint: the firmware Mellinger then damps
    towards zero velocity, ~0.5 s lag); "full_state" keeps them.
    """

    def __init__(self, param: Param, onboard: dict | None = None, setpoint_mode="full_state"):
        super().__init__(param)
        if onboard is None:
            from experiment_arena import CRAZYFLIE_ONBOARD
            onboard = CRAZYFLIE_ONBOARD
        self.setpoint_mode = setpoint_mode
        self.z_offset = onboard["z_offset"]

        # inner steps of the onboard controller within one simulation step
        self.substeps = max(1, round(param.sim.dTi * onboard["rate"]))
        self.inner = copy.copy(param)
        self.inner.sim = copy.copy(param.sim)
        self.inner.sim.dTi = param.sim.dTi / self.substeps
        # setpoints in flight, per drone (id(agent))
        self.delay_steps = max(0, round(onboard["delay"] / param.sim.dTi))
        self.queues: dict[int, deque] = {}
        self.target_id = None

    def start(self, chaser_list: list[DroneState], target: DroneState):
        # a virtual target (as swarm_MPC_voronoi_sim_target) keeps the plain
        # Python model; a target drone (param.sim.target_is_drone, as
        # swarm_MPC_voronoi) is a Crazyflie like the chasers
        if not getattr(self.param.sim, "target_is_drone", False):
            self.target_id = id(target)

    def actuate(self, agent: DroneState):
        if id(agent) == self.target_id:
            return actuate_control(agent, self.param)
        d = agent.desired
        sp = (d.position.copy(), d.velocity.copy(), d.acceleration.copy())
        if self.setpoint_mode == "position":
            sp = (sp[0], np.zeros((3, 1)), np.zeros((3, 1)))

        queue = self.queues.setdefault(id(agent), deque())
        queue.append(sp)
        if len(queue) > self.delay_steps + 1:
            queue.popleft()
        p, v, a = queue[0]

        # onboard setpoint: offset in z (the desired values of the agent are
        # restored after, the planner reads them)
        d.position, d.velocity, d.acceleration = p + np.array([[0.0], [0.0], [-self.z_offset]]), v, a
        for _ in range(self.substeps):
            actuate_control(agent, self.inner)
        d.position, d.velocity, d.acceleration = sp
        return agent

    def actuate_passive(self, agent: DroneState):
        for _ in range(self.substeps):
            actuate_passive(agent, self.inner)
        return agent
