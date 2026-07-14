import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from bernstein_funcs import *
from Structs import *
from helper_funcs import *
from obstacle_avoidance_funcs import *
from visualise import visualise

# This program has objective have 2 drone restricted in vel and acc not collide
# with each other the knowledge of both position horizons

os.system('cls' if os.name=='nt' else 'clear')

# ============================================================
# Problem
# ============================================================
param = Param()

Nsim = param.sim.Nsim
dTi = param.sim.dTi

start = np.array([
    [-5,  5],
    [-5,  5],
    [-5,  5]
], dtype=float)

goal = np.array([
    [ 5, -5],
    [ 5.7, -4.7],
    [ 5.7,  -4.7]
], dtype=float)

chaser_list: list[DroneState] = []
for iD in range(param.sim.nD):
    drone = DroneState(param)
    drone.position = start[:, iD].reshape(-1,1)
    drone.desired.position = goal[:,iD].reshape(-1,1)
    chaser_list.append(drone)

obs_vars = init_obstacle_avoidance(chaser_list, param)
Q = obs_vars["Q"]
A_eq = obs_vars["A_eq"]
A_obs = obs_vars["A_obs"]
G = obs_vars["G"]
h = obs_vars["h"]
W0 = obs_vars["W0"]
W1 = obs_vars["W1"]
W2 = obs_vars["W2"]
n_ctrl = obs_vars["n_ctrl"]
dim = obs_vars["dim"]

wN = W0[-1].reshape(-1,1)
x = np.zeros(dim)

for iD, chaser_i in enumerate(chaser_list):
    # update goal
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        chaser_i.constraints.q[idx] = -param.w_goal * goal[axis, iD] * wN
    chaser_i.constraints.s = np.zeros(G.shape[0]).reshape(-1,1)
    chaser_i.constraints.lambda_obs = np.zeros([A_obs.shape[0],1])
    chaser_i.constraints.lambda_ineq = np.zeros([G.shape[0],1])

    # get 1st trajectory
    solve_unconstraint_motion(obs_vars, chaser_i, param)

# Logs
executed_path = [[] for _ in range(param.sim.nD)]
velocity_history = [[] for _ in range(param.sim.nD)]
acc_history = [[] for _ in range(param.sim.nD)]
horizon_history = [[] for _ in range(param.sim.nD)]
d_history = [[] for _ in range(param.sim.nD)]

for iD, chaser_i in enumerate(chaser_list):
    executed_path[iD].append(chaser_i.position.flatten())
    velocity_history[iD].append(chaser_i.velocity.flatten())
    acc_history[iD].append(chaser_i.acceleration.flatten())
    

# pre initialize for optimization
b_eq = np.zeros([9,1])
A_bar = np.zeros([33,33])
b_bar = np.zeros([33,1])
KKT = np.zeros([42,42])
rhs = np.zeros([42,1])
solution = np.zeros([42,1])

for step in range(Nsim):
    t = step*param.sim.dTi
    print(f"Current iter: {step}")

    # =====================================================
    # Solve AL subproblem
    # =====================================================
    for iteration in range(100):
        for iD, agent in enumerate(chaser_list):
            # =====================================================
            # Equality constraint becomes current position
            # =====================================================
            b_eq = np.vstack([agent.position, agent.velocity, agent.acceleration]).reshape(-1,1)

            # Step 1 - solve trajectory
            A_bar = (
                Q
                + param.rho_obs * A_obs.T @ A_obs
                + param.rho_ineq * G.T @ G
            )

            b_bar = (
                - agent.constraints.q
                - A_obs.T @ agent.constraints.lambda_obs
                + param.rho_obs * A_obs.T @ agent.constraints.b_obs
                - G.T @ agent.constraints.lambda_ineq
                + param.rho_ineq * G.T @ (h - agent.constraints.s)
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
            x = solution[:33]
            agent.constraints.C = x.flatten().reshape(n_ctrl,3)
            # Step 2 - Update alpha, beta and d
            other_chasers = chaser_list[:iD] + chaser_list[iD+1:]
            if len(other_chasers) != 0:
                other_horizons = []
                for other_chaser in other_chasers:
                    other_horizons.append(W0 @ other_chaser.constraints.C)
                other_horizons = np.vstack(other_horizons)
                traj_horizon = W0 @ agent.constraints.C
                for k in range(param.horizon):
                    delta = np.linalg.inv(agent.constraints.obstacle_radius) @ (traj_horizon[k, :] - other_horizons[k,:]).reshape(-1,1)
                    d, alpha, beta = cart2pol_3D(delta)
                    #d_history[iD].append(d)
                    d = max(d, 1.0)
                    b = other_horizons[k,:].reshape(3,1) + agent.constraints.obstacle_radius @ (d * omega(alpha, beta))
                    rows = np.arange(k, 3 * param.horizon, param.horizon)
                    agent.constraints.b_obs[rows,:] = b

            # Step 3 - Update slack term
            agent.constraints.s = np.maximum(0,h - G @ x - agent.constraints.lambda_ineq/param.rho_ineq)

            # calculate residuals
            residual_eq  = A_eq @ x - b_eq
            residual_ineq = G @ x - h + agent.constraints.s
            residual_obs = A_obs @ x - agent.constraints.b_obs

            # Step 4 - Update lambdas
            agent.constraints.lambda_ineq += param.rho_ineq * residual_ineq
            agent.constraints.lambda_obs += param.rho_obs * residual_obs
                
    for iD, chaser_i in enumerate(chaser_list):
        chaser_i.constraints.lambda_obs = np.zeros([A_obs.shape[0],1])
        chaser_i.constraints.lambda_ineq = np.zeros([G.shape[0],1])
        C = chaser_i.constraints.C
        traj = W0 @ C
        vel = W1 @ C
        acc = W2 @ C

        # executing next step velocity
        chaser_i.acceleration = acc[1]
        chaser_i.velocity = vel[1]
        chaser_i.position = traj[1]

        executed_path[iD].append(chaser_i.position.flatten())
        velocity_history[iD].append(chaser_i.velocity.flatten())
        acc_history[iD].append(chaser_i.acceleration.flatten())
        horizon_history[iD].append(traj.copy())

        for axis in range(3):
            idx = np.arange(axis, dim, 3)
            chaser_i.constraints.q[idx] = -param.w_goal * goal[axis, iD] * wN

executed_path = np.array(executed_path)
velocity_history = np.array(velocity_history)
acc_history = np.array(acc_history)
horizon_history = np.array(horizon_history)
#d_history = np.array(d_history)

results = {
    "executed_path": executed_path,
    "velocity_history": velocity_history,
    "acc_history": acc_history,
    "horizon_history": horizon_history,
    #"d_history": d_history,
    "start": start,
    "goal": goal
}

if param.display.store_data == True:
    with open("simulation_results.pkl", "wb") as f:
        pickle.dump(results, f)

if param.display.verbose == True:
    visualise(results, param)