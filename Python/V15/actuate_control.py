import numpy as np
from scipy.integrate import solve_ivp

from drone_mellinger_ctrl import drone_mellinger_ctrl
from drone_3dfull_dyn import drone_3dfull_dyn
from helper_funcs import rot_integrate
from Structs import DroneState, Param

def actuate_control(agent: DroneState, param: Param):
    if param.controller.selected_controller == "Mellinger":
        pos = agent.position
        vel = agent.velocity
        attitude = agent.attitude.reshape((3, 3))
        ang_vel = agent.ang_vel

        # Reference yaw rate
        dpsi_d = (agent.desired.psi - agent.desired.old_psi) / param.sim.dTi
        agent.desired.old_psi = agent.desired.psi

        # Mellinger controller
        thrust, torque = drone_mellinger_ctrl(
            agent,
            dpsi_d,
            param
        )

        # Continuous-time drone dynamics
        dot_p, dot_v, dot_om = drone_3dfull_dyn(
            vel,
            attitude,
            ang_vel,
            thrust,
            torque,
            param,
        )

        # Euler discretization
        agent.position = pos + param.sim.dTi * dot_p
        agent.velocity = vel + param.sim.dTi * dot_v

        Rp = rot_integrate(
            attitude,
            ang_vel,
            param.sim.dTi,
        )

        agent.attitude = Rp.reshape(-1,1)

        agent.ang_vel = (
            ang_vel + param.sim.dTi * dot_om
        )

    elif param.controller.selected_controller == "Lee":

        X = struct_to_pack(k, agent)

        sol = solve_ivp(
            lambda t, XR: eom(
                t,
                XR,
                desired,
                Sim,
                param,
            ),
            [param.t[k], param.t[k + 1]],
            X,
        )

        Xtmp = sol.y[:, -1]

        agent = pack_to_struct(
            k + 1,
            agent,
            Xtmp,
        )

    else:
        raise ValueError(
            f"Unknown controller type: {param.controller}"
        )

    return agent