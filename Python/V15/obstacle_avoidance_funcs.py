import numpy as np
from Structs import Param, DroneState
from bernstein_funcs import bernstein_basis_matrix
from helper_funcs import build_fixed_b_obs, build_horizon_b_obs

def init_obstacle_avoidance(chaser_list: list[DroneState], param: Param):
    """
    Initializes necessary variables for the QP with constrains being the 
    constraints vel, acc and obstacle avoidance

    Parameters
    ----------
    chaser_list : List of class DroneState
        DronesState has all valuable and drone unique data
    param: Class
        Global variables for simulation 
    
    Returns
    -------
        - Q: smoothness factor matrix 
        - A_eq : equality constraints in pos vel and acc, basically current 
        values must be taken into account when optimizing

        - A_obs: equality constraints on position (obstacle avoidance)
        - G: inequality constraints on vel and acc
        - h: min/max values of the inequality constraints (G)
        - W0: bernstein polynomial matrix basis for position
        - W1: bernstein polynomial matrix basis for velocity (1st derivative)
        - W2: bernstein polynomial matrix basis for acceleration 
            (2st derivative)
        - n_ctrl: polynomial degree of bernstein polynomial + 1 
            (0th degree included)
        - dim: dimensions of the array polynomial given 3 axis (3*n_ctrl)
    """

    # ===========================================================
    # Variable layout
    #
    # x =
    #   [
    #       c0x,c0y,c0z,
    #       c1x,c1y,c1z,
    #       ...
    #       cnx,cny,cnz
    #   ]
    # the path is defined via bernstein polynomials instead of [x,y,z]
    # being the number of elements of the matrix R^[n_ctrl x 3]
    # ============================================================
    n_degree = param.n_degree               # bernstein polynomial degree
    H = param.horizon
    n_ctrl = n_degree + 1
    dim = 3 * n_ctrl

    # bernstein basis matrix construction
    # obtained from the Paper:
    # On the Derivatives of Bernstein Polynomials: 
    # An Application for the Solution of High Even-Order Differential Equations
    t = np.linspace(0, 1, H)
    W0 = bernstein_basis_matrix(t, n_degree)

    B = bernstein_basis_matrix(t, n_degree-1)
    D1 = np.zeros((n_degree, n_degree+1))
    for i in range(n_degree-1):
        D1[i,i]   = -1
        D1[i,i+1] = 1
    W1 = n_degree * (B @ D1)

    B = bernstein_basis_matrix(t, n_degree-2)
    D2 = np.zeros((n_degree-1, n_degree+1))
    for i in range(n_degree-1):
        D2[i,i]   = 1
        D2[i,i+1] = -2
        D2[i,i+2] = 1
    W2 = n_degree * (n_degree-1) * (B @ D2)

    # ============================================================
    # Smoothness Hessian
    #
    # sum ||\ddot p(k)||²
    # the smoothness hessian is defined as a cost function that reduces high value 
    # accelerations
    # ============================================================
    Q1 = param.w_smooth * W2.T @ W2

    Q = np.zeros((dim, dim))
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        Q[np.ix_(idx, idx)] = Q1

    # ============================================================
    # Goal cost
    #
    # w_goal ||p_N - goal||²
    # The goal cost is the cost associated with the distance between the agent 
    # and the goal position
    # ============================================================
    wN = W0[-1].reshape(-1,1)

    Qg = param.w_goal * np.outer(wN, wN)
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        Q[np.ix_(idx, idx)] += Qg
    for iD in range(param.sim.nD):
        for axis in range(3):
            idx = np.arange(axis, dim, 3)
            chaser_list[iD].constraints.q[idx] = -param.w_goal * chaser_list[iD].desired.position[axis] * wN

    # ============================================================
    # Equality constraints
    # ============================================================

    # [p(0),v(0),acc(0)] = start
    # This equality constraint defines that the system must have at horizon 1
    # the current pos, vel and acc in account
    # A_eq =
    # [ W0(0)  0      0    ]
    # [ 0      W0(0)  0    ]  position
    # [ 0      0      W0(0)]
    # ----------------------
    # [ W1(0)  0      0    ]
    # [ 0      W1(0)  0    ]  velocity
    # [ 0      0      W1(0)]
    # ----------------------
    # [ W2(0)  0      0    ]
    # [ 0      W2(0)  0    ]  acceleration
    # [ 0      0      W2(0)]
    A_eq = np.zeros((9, dim))
    for axis in range(3):
        cols = np.arange(axis, dim, 3)
        A_eq[axis, cols]     = W0[0]   # position
        A_eq[axis+3, cols]   = W1[0]   # velocity
        A_eq[axis+6, cols]   = W2[0]   # acceleration

    # This equality constraint defines that the system must along the entire horizon
    # respect the enforcement on collision avoidance.
    # A_obs =
    # [ W0  0   0 ] x
    # [ 0   W0  0 ] y  position
    # [ 0   0   W0] z
    A_obs_blk = np.zeros((3*param.horizon, dim))
    for axis in range(3):
        cols = np.arange(axis, dim, 3)
        A_obs_blk[param.horizon*axis:param.horizon*(axis+1), cols] = W0   # position
        
    # ============================================================
    # Inequality constraints
    #
    # G x <= h
    # The inequality is given as a restriction in velocity
    # the inequality can be introduced directly the slack term 's' in
    # Gx-h+s = 0. The constrain is given as a cube around the agent 
    # ============================================================

    # G_tilde =
    # [ W1  0   0 ]  x
    # [ 0   W1  0 ]  y
    # [ 0   0   W1]  z
    # Velocity constraints
    G_vel = np.zeros((3 * H, dim))
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        rows = np.arange(axis * H, (axis + 1) * H)
        G_vel[np.ix_(rows, idx)] = W1

    # Acceleration constraints
    G_acc = np.zeros((3 * H, dim))
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        rows = np.arange(axis * H, (axis + 1) * H)
        G_acc[np.ix_(rows, idx)] = W2

    # Assemble G
    G = np.vstack((
        G_vel,
        -G_vel,
        G_acc,
        -G_acc,
    ))

    # Assemble h
    h = np.vstack((
        np.ones((6 * H, 1)) * param.v_max,
        np.ones((6 * H, 1)) * param.a_max,
    ))

    # ============================================================
    # Augmented Lagrangian
    # ============================================================
    for iD in range(param.sim.nD):
        chaser_list[iD].constraints.s = np.zeros(G.shape[0]).reshape(-1,1)
        chaser_list[iD].constraints.lambda_obs = np.zeros([A_obs_blk.shape[0],1])
        chaser_list[iD].constraints.lambda_ineq = np.zeros([G.shape[0],1])

    return {
        "Q": Q,
        "A_eq": A_eq,
        "A_obs_blk": A_obs_blk,
        "G": G,
        "h": h,
        "W0": W0,
        "W1": W1,
        "W2": W2,
        "n_ctrl": n_ctrl,
        "dim": dim,
    }

def solve_unconstrained_motion(obs_vars, agent: DroneState, param: Param):
    """
    Solve the unconstrained motion, useful for initialization of desired 
    trajectory

    Parameters
    ----------
    obs_vars: Dictionary of variables associated with the QP and constraints
    agent : class DroneState
        DronesState has all valuable and drone unique data
    param: Class
        Global variables for simulation 
    
    Updates
    -------
        agent.constraints.C - bernoulli coefficients
        agent.constraints.s - slack term
    """

    Q = obs_vars["Q"]
    A_eq = obs_vars["A_eq"]
    G = obs_vars["G"]
    h = obs_vars["h"]
    n_ctrl = obs_vars["n_ctrl"]

    for _ in range(param.max_iter):
        # =====================================================
        # Equality constraint becomes current position
        # =====================================================
        b_eq = np.vstack([agent.position, agent.velocity, agent.acceleration]).reshape(-1,1)

        # Step 1 - solve trajectory
        A_bar = (
            Q
        )

        b_bar = (
            - agent.constraints.q
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

        # Step 2 - Update alpha, beta and d
        agent.constraints.C = x.flatten().reshape(n_ctrl,3)

        # Step 3 - Update slack term
        agent.constraints.s = np.maximum(0,h - G @ x - agent.constraints.lambda_ineq/param.rho_ineq)

        # calculate residuals
        residual_eq  = A_eq @ x - b_eq

        if (
            np.linalg.norm(residual_eq)
            < 1e-6
        ):
            break


def solve_constraint_motion_horizon(chaser_list: list[DroneState], iD, obs_vars, param: Param):
    """
    Solve the unconstrained motion, useful for initialization of desired 
    trajectory

    Parameters
    ----------
    obs_vars: Dictionary of variables associated with the QP and constraints
    agent : class DroneState
        DronesState has all valuable and drone unique data
    param: Class
        Global variables for simulation 
    
    Updates
    -------
        agent.constraints.C - bernoulli coefficients
        agent.constraints.s - slack term
    """

    Q = obs_vars["Q"]
    A_eq = obs_vars["A_eq"]
    A_obs = obs_vars["A_obs"]
    G = obs_vars["G"]
    h = obs_vars["h"]
    W0 = obs_vars["W0"]
    n_ctrl = obs_vars["n_ctrl"]

    agent = chaser_list[iD]
    # =====================================================
    # Equality constraint becomes current position
    # =====================================================
    b_eq = np.hstack([agent.position, agent.velocity, agent.acceleration]).reshape(-1,1)

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

    n = A_bar.shape[0]
    x = solution[:n]

    # Step 2 - Update alpha, beta and d
    agent.constraints.C = x.flatten().reshape(n_ctrl,3)

    other_chasers = chaser_list[:iD] + chaser_list[iD+1:]
    if len(other_chasers) != 0:
        other_horizons = []
        for other_chaser in other_chasers:
            other_horizons.append(W0 @ other_chaser.constraints.C)
        other_horizons = np.vstack(other_horizons)
        agent.constraints.b_obs = build_horizon_b_obs(
            param.horizon,
            agent.constraints.obstacle_radius,
            other_horizons,
            W0 @ agent.constraints.C
        )

    # Step 3 - Update slack term
    agent.constraints.s = np.maximum(0,h - G @ x - agent.constraints.lambda_ineq/param.rho_ineq)

    # calculate residuals
    residual_eq  = A_eq @ x - b_eq
    residual_ineq = G @ x - h + agent.constraints.s
    residual_obs = A_obs @ x - agent.constraints.b_obs

    # Step 4 - Update lambdas
    agent.constraints.lambda_ineq += param.rho_ineq * residual_ineq
    agent.constraints.lambda_obs += param.rho_obs * residual_obs

    # if (
    #     np.linalg.norm(residual_eq)
    #     + np.linalg.norm(residual_ineq)
    #     + np.linalg.norm(residual_obs)
    #     < 1e-6
    # ):
    #     break


def solve_constraint_motion_fixed(chaser_list: list[DroneState], iD, obs_vars, param: Param):
    """
    Solve the unconstrained motion, useful for initialization of desired 
    trajectory

    Parameters
    ----------
    obs_vars: Dictionary of variables associated with the QP and constraints
    agent : class DroneState
        DronesState has all valuable and drone unique data
    param: Class
        Global variables for simulation 
    
    Updates
    -------
        agent.constraints.C - bernoulli coefficients
        agent.constraints.s - slack term
    """

    Q = obs_vars["Q"]
    A_eq = obs_vars["A_eq"]
    A_obs = obs_vars["A_obs"]
    G = obs_vars["G"]
    h = obs_vars["h"]
    W0 = obs_vars["W0"]
    n_ctrl = obs_vars["n_ctrl"]

    agent = chaser_list[iD]
    # =====================================================
    # Equality constraint becomes current position
    # =====================================================
    b_eq = np.hstack([agent.position, agent.velocity, agent.acceleration]).reshape(-1,1)

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

    n = A_bar.shape[0]
    x = solution[:n]

    # Step 2 - Update alpha, beta and d
    agent.constraints.C = x.flatten().reshape(n_ctrl,3)

    traj = W0 @ agent.constraints.C
    agent.constraints.b_obs = build_fixed_b_obs(param.horizon, agent.constraints.obstacle_radius, agent.constraints.obstacle_center, traj)


    # Step 3 - Update slack term
    agent.constraints.s = np.maximum(0,h - G @ x - agent.constraints.lambda_ineq/param.rho_ineq)

    # calculate residuals
    residual_eq  = A_eq @ x - b_eq
    residual_ineq = G @ x - h + agent.constraints.s
    residual_obs = A_obs @ x - agent.constraints.b_obs

    # Step 4 - Update lambdas
    agent.constraints.lambda_ineq += param.rho_ineq * residual_ineq
    agent.constraints.lambda_obs += param.rho_obs * residual_obs

    return (
        np.linalg.norm(residual_eq)
        + np.linalg.norm(residual_ineq)
        + np.linalg.norm(residual_obs)
    )