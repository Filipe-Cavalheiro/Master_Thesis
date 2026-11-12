import numpy as np
from scipy.linalg import lu_factor, lu_solve
from Structs_sim import Param
from Structs_agent import DroneState
from bernstein_funcs import bernstein_basis_matrix
from helper_funcs import build_fixed_b_obs, build_horizon_b_obs
from environment import project_out

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
        - G: vel and acc samples of the horizon (G x), bounded in norm
            via project_vel_acc
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
    n_degree = param.sim.n_degree               # bernstein polynomial degree
    H = param.sim.horizon
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
    for i in range(n_degree):
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
    # Q =
    # [ n_ctrl x n_ctrl  0                0              ] x
    # [ 0                n_ctrl x n_ctrl  0              ] y position
    # [ 0                0                n_ctrl x n_ctrl] z
    # ============================================================
    Q = np.zeros((dim, dim))
    Qs = param.sim.w_smooth * W2.T @ W2
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        Q[np.ix_(idx, idx)] = Qs

    # ============================================================
    # Goal cost
    #
    # sum{H}{k=1} w_goal ||p_k - goal||²
    # The goal cost is the cost associated with the distance between the agent 
    # and the goal position
    # ============================================================
    Qg = param.sim.w_goal * (W0.T @ W0)
    goal_vec = W0.T @ np.ones((H,1))
    
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        Q[np.ix_(idx, idx)] += Qg

    # ============================================================
    # Desired velocity cost
    #
    # sum{H}{k=1} w_vel ||v_k - v_ref,k||²
    # ============================================================
    if param.sim.w_vel > 0:
        Qv = param.sim.w_vel * (W1.T @ W1)
        for axis in range(3):
            idx = np.arange(axis, dim, 3)
            Q[np.ix_(idx, idx)] += Qv
    for iD in range(param.sim.nD):
        for axis in range(3):
            idx = np.arange(axis, dim, 3)
            chaser_list[iD].constraints.q[idx] = -param.sim.w_goal * chaser_list[iD].desired.position[axis] * goal_vec

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
    A_obs_blk = np.zeros((3*param.sim.horizon, dim))
    for axis in range(3):
        cols = np.arange(axis, dim, 3)
        A_obs_blk[param.sim.horizon*axis:param.sim.horizon*(axis+1), cols] = W0   # position
        
    # ============================================================
    # Velocity / acceleration constraints (spherical)
    #
    # ||v[k]|| <= v_max, ||a[k]|| <= a_max
    # Following AMSwarm (arXiv:2303.04856) the norm constraints are written
    # in polar form  G x = s,  s[k] = d[k] w(alpha[k], beta[k]),  d[k] <= r.
    # The s update is a projection onto the ball (project_vel_acc), so the
    # bound holds on the norm.
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

    # Assemble G, rows laid out as [vx(H), vy(H), vz(H), ax(H), ay(H), az(H)]
    G = np.vstack((
        G_vel,
        G_acc,
    ))

    # ============================================================
    # Augmented Lagrangian
    # ============================================================
    for iD in range(param.sim.nD):
        chaser_list[iD].constraints.s = np.zeros(G.shape[0]).reshape(-1,1)
        chaser_list[iD].constraints.lambda_obs = np.zeros([A_obs_blk.shape[0],1])
        chaser_list[iD].constraints.lambda_ineq = np.zeros([G.shape[0],1])

    return {
        "t_h": t,
        "Q": Q,
        "A_eq": A_eq,
        "A_obs_blk": A_obs_blk,
        "G": G,
        "W0": W0,
        "W1": W1,
        "W2": W2,
        "n_ctrl": n_ctrl,
        "dim": dim,
    }

def project_vel_acc(Gx, param: Param):
    """
    Projection of the stacked samples [v(3H); a(3H)] onto the balls
    ||v[k]|| <= v_max and ||a[k]|| <= a_max (polar form of AMSwarm: keeps the
    direction w(alpha, beta) and clips the radius d).
    returns: (6*H, 1) with the same layout as G x
    """
    va = Gx.reshape(2, 3, param.sim.horizon)                 # [vel/acc, axis, k]
    d = np.linalg.norm(va, axis=1, keepdims=True)            # (2, 1, H)
    r = np.array([param.sim.v_max, param.sim.a_max]).reshape(2, 1, 1)
    scale = np.minimum(1.0, r / np.maximum(d, 1e-12))
    return (va * scale).reshape(-1, 1)


def bernstein_eval(C, tau, n_degree):
    """
    Position, velocity and acceleration (each (3,)) of the plan with control
    points C (n_ctrl, 3) at the normalized time tau in [0, 1] of the 1 s
    horizon (same bases as init_obstacle_avoidance).
    """
    t = np.array([tau])
    W0 = bernstein_basis_matrix(t, n_degree)
    D1 = np.zeros((n_degree, n_degree+1))
    for i in range(n_degree):
        D1[i,i]   = -1
        D1[i,i+1] = 1
    W1 = n_degree * (bernstein_basis_matrix(t, n_degree-1) @ D1)
    D2 = np.zeros((n_degree-1, n_degree+1))
    for i in range(n_degree-1):
        D2[i,i]   = 1
        D2[i,i+1] = -2
        D2[i,i+2] = 1
    W2 = n_degree * (n_degree-1) * (bernstein_basis_matrix(t, n_degree-2) @ D2)
    return (W0 @ C)[0], (W1 @ C)[0], (W2 @ C)[0]


def clip_norm(v, r):
    """v with its norm clipped to r."""
    n = np.linalg.norm(v)
    return v * min(1.0, r / max(n, 1e-12))


def mpc_setpoint(agent: DroneState, param: Param):
    """
    Setpoint (p, v, a), each (3, 1), of drones that track their setpoints
    tightly (param.sim.mpc_tight_tracking, as the Crazyflies): the plan one
    control period ahead (instead of the second horizon sample, 0.25 s
    ahead), feedforward clipped to 1.2 v_max / 1.5 a_max, as the flight code
    (Crazy_Flies/swarm_MPC_voronoi_sim_target/mpc.py).
    """
    p, v, a = bernstein_eval(agent.constraints.C, param.sim.dTi, param.sim.n_degree)
    v = clip_norm(v, 1.2 * param.sim.v_max)
    a = clip_norm(a, 1.5 * param.sim.a_max)
    return p.reshape(-1, 1), v.reshape(-1, 1), a.reshape(-1, 1)


def set_goal_cost(agent: DroneState, obs_vars, param: Param):
    """
    Linear term q of the goal (and desired velocity) objective of an agent.

    Reference over the horizon (t in [0, 1] s):
        r(t) = goal + v_g t + (p_now - goal) exp(-t / tau)
        v(t) = v_g - (p_now - goal) / tau * exp(-t / tau)
    with v_g the goal velocity (from the target velocity). The horizon is
    pulled towards an exponential approach to the goal starting at the
    current position, instead of straight to the goal, so the planned
    trajectory (and the controller setpoint) does not jump far ahead of the
    drone. tau is raised to |p_now - goal| / v_max when needed, so the
    approach never asks for more than v_max. tau = 0 and w_vel = 0 give the
    plain goal objective.
    """
    W0, W1, dim = obs_vars["W0"], obs_vars["W1"], obs_vars["dim"]
    t = obs_vars["t_h"].reshape(-1, 1)                      # (H, 1)
    goal = agent.desired.goal.reshape(1, 3)
    tau = param.sim.goal_ref_tau
    use_vel = param.sim.w_vel > 0
    v_g = agent.desired.goal_velocity.reshape(1, 3) if use_vel else np.zeros((1, 3))

    ref_pos = goal + t * v_g
    ref_vel = np.repeat(v_g, len(t), axis=0)
    if tau > 0:
        err0 = agent.position.reshape(1, 3) - goal
        # the approach speed |err0| / tau must not exceed v_max: far from the
        # goal (e.g. a goal the drone cannot reach) the approach is slower
        tau = max(tau, np.linalg.norm(err0) / param.sim.v_max)
        decay = np.exp(-t / tau)                             # (H, 1)
        ref_pos = ref_pos + decay * err0
        ref_vel = ref_vel - decay / tau * err0

    q = np.zeros((dim, 1))
    for axis in range(3):
        idx = np.arange(axis, dim, 3)
        q[idx] = -param.sim.w_goal * W0.T @ ref_pos[:, axis:axis+1]
        if use_vel:
            q[idx] += -param.sim.w_vel * W1.T @ ref_vel[:, axis:axis+1]
    agent.constraints.q = q


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
    n_ctrl = obs_vars["n_ctrl"]

    for _ in range(param.sim.max_iter):
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
        agent.constraints.s = project_vel_acc(G @ x + agent.constraints.lambda_ineq/param.sim.rho_ineq, param)

        # calculate residuals
        residual_eq  = A_eq @ x - b_eq

        if (
            np.linalg.norm(residual_eq)
            < 1e-6
        ):
            break

def solve_constraint_motion_horizon(chaser_list: list[DroneState], iD, obs_vars, param: Param):
    """
    solve the constrained motion horizon with the obstacles of the existence of other agents in account
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
        + param.sim.rho_obs * A_obs.T @ A_obs
        + param.sim.rho_ineq * G.T @ G
    )

    b_bar = (
        - agent.constraints.q
        - A_obs.T @ agent.constraints.lambda_obs
        + param.sim.rho_obs * A_obs.T @ agent.constraints.b_obs
        - G.T @ agent.constraints.lambda_ineq
        + param.sim.rho_ineq * G.T @ agent.constraints.s
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
            param.sim.horizon,
            agent.constraints.obstacle_radius,
            other_horizons,
            W0 @ agent.constraints.C
        )

    # Step 3 - Update slack term
    agent.constraints.s = project_vel_acc(G @ x + agent.constraints.lambda_ineq/param.sim.rho_ineq, param)

    # calculate residuals
    residual_eq  = A_eq @ x - b_eq
    residual_ineq = G @ x - agent.constraints.s
    residual_obs = A_obs @ x - agent.constraints.b_obs

    # Step 4 - Update lambdas
    agent.constraints.lambda_ineq += param.sim.rho_ineq * residual_ineq
    agent.constraints.lambda_obs += param.sim.rho_obs * residual_obs

    # if (
    #     np.linalg.norm(residual_eq)
    #     + np.linalg.norm(residual_ineq)
    #     + np.linalg.norm(residual_obs)
    #     < 1e-6
    # ):
    #     break


def solve_constraint_motion_fixed(chaser_list: list[DroneState], iD, obs_vars, param: Param):
    """
    solve the constraint motion with an fixed in place obstacle
    
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
        + param.sim.rho_obs * A_obs.T @ A_obs
        + param.sim.rho_ineq * G.T @ G
    )

    b_bar = (
        - agent.constraints.q
        - A_obs.T @ agent.constraints.lambda_obs
        + param.sim.rho_obs * A_obs.T @ agent.constraints.b_obs
        - G.T @ agent.constraints.lambda_ineq
        + param.sim.rho_ineq * G.T @ agent.constraints.s
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
    agent.constraints.b_obs = build_fixed_b_obs(param.sim.horizon, agent.constraints.obstacle_radius, agent.constraints.obstacle_center, traj)


    # Step 3 - Update slack term
    agent.constraints.s = project_vel_acc(G @ x + agent.constraints.lambda_ineq/param.sim.rho_ineq, param)

    # calculate residuals
    residual_eq  = A_eq @ x - b_eq
    residual_ineq = G @ x - agent.constraints.s
    residual_obs = A_obs @ x - agent.constraints.b_obs

    # Step 4 - Update lambdas
    agent.constraints.lambda_ineq += param.sim.rho_ineq * residual_ineq
    agent.constraints.lambda_obs += param.sim.rho_obs * residual_obs

    return (
        np.linalg.norm(residual_eq)
        + np.linalg.norm(residual_ineq)
        + np.linalg.norm(residual_obs)
    )


def build_b_obs(traj, others, r_vec):
    """
    Vectorized obstacle projection points.

    traj   : (H, 3)    own horizon
    others : (n, H, 3) horizons of the other agents
    returns: (n*3*H, 1), per obstacle laid out as [x(H), y(H), z(H)]
    """
    delta = (traj[None, :, :] - others) / r_vec          # (n, H, 3)
    d = np.linalg.norm(delta, axis=2, keepdims=True)     # (n, H, 1)
    safe = d > 1e-9
    direction = np.where(safe, delta / np.where(safe, d, 1.0), 0.0)
    # if the two agents coincide, push along +x to avoid a zero direction
    direction[~safe[..., 0], :] = np.array([1.0, 0.0, 0.0])
    b = others + r_vec * np.maximum(d, 1.0) * direction  # (n, H, 3)
    return b.transpose(0, 2, 1).reshape(-1, 1)


def build_b_obs_cylinder(traj, others, r_xy, r_z, side):
    """
    Obstacle projection points out of vertical cylinders (radius r_xy,
    half-height r_z) around the other agents, as the flight code
    (Crazy_Flies/swarm_MPC_voronoi_sim_target/mpc.py, project_others): the
    shortest way out, so two stacked drones are pushed sideways (the
    ellipsoid pushes them vertically, which a formation whose vertical
    extent is below 2 r_z cannot give, and they stay in each other's
    downwash).

    traj   : (H, 3)    own horizon
    others : (n, H, 3) horizons of the other agents
    side   : (n,) +1/-1, direction along y when exactly above/below
    returns: (n*3*H, 1), per obstacle laid out as [x(H), y(H), z(H)]
    """
    pts = np.broadcast_to(traj, others.shape)
    delta = pts - others
    dh = np.linalg.norm(delta[..., :2], axis=2)
    dz = delta[..., 2]
    inside = (dh < r_xy) & (np.abs(dz) < r_z)
    horizontal = inside & (r_xy - dh <= r_z - np.abs(dz))
    vertical = inside & ~horizontal

    fallback = np.zeros_like(delta[..., :2])
    fallback[..., 1] = np.asarray(side)[:, None]
    hdir = np.where((dh > 1e-6)[..., None], delta[..., :2] / np.maximum(dh, 1e-6)[..., None], fallback)

    b = np.array(pts, copy=True)
    b[..., :2] = np.where(horizontal[..., None], others[..., :2] + r_xy * hdir, b[..., :2])
    b[..., 2] = np.where(vertical, others[..., 2] + np.where(dz >= 0, 1.0, -1.0) * r_z, b[..., 2])
    return b.transpose(0, 2, 1).reshape(-1, 1)


def build_b_air(traj, airspace, obstacles=None, clearance=0.0):
    """
    Airspace / environment projection points for the own horizon (H, 3): the
    horizon pushed out of the red zone, below the orange ceiling and away from
    the obstacles the drone senses (environment.py).
    returns: (3*H, 1) laid out as [x(H), y(H), z(H)]
    """
    traj = airspace.project(traj)
    if obstacles is not None and len(obstacles):
        traj = project_out(traj, obstacles, clearance)
    return traj.T.reshape(-1, 1)


class AdmmMPC:
    """
    Distributed MPC solved with ADMM (other agents + restricted airspace as
    projection constraints, vel/acc norm limits as projections onto balls).

    A_obs of every agent is just A_obs_blk tiled once per obstacle plus once
    for the restricted airspace, so the matrix
    Q + rho_obs*A_obs^T A_obs + rho_ineq*G^T G  (and hence the KKT matrix) is
    CONSTANT as long as the number of obstacles does not change. It is
    factorized once (and again only when drones are dropped, see rebuild) and
    only the right-hand side is rebuilt every iteration.
    """

    def __init__(self, obs_vars, chaser_list: list[DroneState], param: Param):
        self.param = param
        self.chaser_list = chaser_list
        self.airspace = param.airspace
        self.Q = obs_vars["Q"]
        self.A_eq = obs_vars["A_eq"]
        self.A_obs_blk = obs_vars["A_obs_blk"]
        self.G = obs_vars["G"]
        self.W0 = obs_vars["W0"]
        self.n_ctrl = obs_vars["n_ctrl"]
        self.dim = obs_vars["dim"]
        self.GT = self.G.T.copy()
        self.GTG = self.GT @ self.G
        self.obstacle_radius_vec = np.diag(chaser_list[0].constraints.obstacle_radius).copy()  # (3,)
        self.mpc_iters = getattr(param.sim, "mpc_iters", 15)   # ADMM iterations per step
        # target drone avoided as the other agents (param.sim.target_is_drone):
        # its horizon is a constant velocity prediction from target_state
        self.n_target = int(getattr(param.sim, "target_is_drone", False))
        self.target_state = None       # (position, velocity) of the target, set by main every step

        # horizon (position) of every agent, shape (nD, H, 3)
        self.horizons = np.zeros((param.sim.nD, param.sim.horizon, 3))
        for iD, agent in enumerate(chaser_list):
            self.horizons[iD] = self.W0 @ agent.constraints.C

        self.A_full = None
        self.A_full_T = None
        self.lu = None
        self.others = {}

    def b_total(self, iD, v=None):
        """
        b_obs of agent iD: [other agents ..., target drone, airspace], the
        projection of its horizon, or of v (= A_obs x + lambda_obs / rho_obs,
        scaled ADMM form, one (3H, 1) block per obstacle) when given.
        """
        others = self.others[int(iD)]
        H = self.param.sim.horizon
        n_blocks = len(others) + self.n_target + 1
        if v is None:
            pts = np.repeat(self.horizons[iD][None], n_blocks, axis=0)
        else:
            pts = v.reshape(n_blocks, 3, H).transpose(0, 2, 1)   # (n_blocks, H, 3)
        parts = []
        cylinder = getattr(self.param.sim, "obstacle_shape", "ellipsoid") == "cylinder"
        r = self.obstacle_radius_vec
        for k, j in enumerate(others):
            if cylinder:
                parts.append(build_b_obs_cylinder(pts[k], self.horizons[j][None], r[0], r[2],
                                                  [1.0 if j > iD else -1.0]))
            else:
                parts.append(build_b_obs(pts[k], self.horizons[j][None], r))
        if self.n_target:
            tp, tv = (np.asarray(x, dtype=float).reshape(1, 3) for x in self.target_state)
            t_h = np.linspace(0, 1, H).reshape(-1, 1)          # the 1 s horizon
            target_h = (tp + t_h * tv)[None]
            if cylinder:
                parts.append(build_b_obs_cylinder(pts[-2], target_h, r[0], r[2], [1.0]))
            else:
                parts.append(build_b_obs(pts[-2], target_h, r))
        parts.append(build_b_air(pts[-1], self.airspace,
                                 self.chaser_list[iD].perceived_obstacles, self.param.sim.env_clearance))
        return np.vstack(parts)

    def rebuild(self, active_ids):
        """
        (Re)build everything that depends on the number of obstacles:
        tiled A_obs, LU factorization of the KKT matrix, neighbor index lists,
        and reset of b_obs / lambda_obs of every active agent.
        """
        param = self.param
        n_obs = len(active_ids) - 1
        # one block per other agent (+ the target drone) + one block for the restricted airspace
        A_full = np.tile(self.A_obs_blk, (n_obs + self.n_target + 1, 1))
        A_full_T = A_full.T.copy()

        A_bar = (
            self.Q
            + param.sim.rho_obs * (A_full_T @ A_full)
            + param.sim.rho_ineq * self.GTG
        )
        KKT = np.block([
            [A_bar, self.A_eq.T],
            [self.A_eq, np.zeros((self.A_eq.shape[0], self.A_eq.shape[0]))]
        ])

        self.A_full = A_full
        self.A_full_T = A_full_T
        self.lu = lu_factor(KKT)
        self.others = {
            int(i): np.array([int(j) for j in active_ids if j != i], dtype=int)
            for i in active_ids
        }

        for i in active_ids:
            c = self.chaser_list[i].constraints
            c.A_obs = A_full
            c.b_obs = self.b_total(i)
            c.lambda_obs = np.zeros((A_full.shape[0], 1))

    def solve(self, active_ids):
        """ADMM iterations of one time step for the active agents."""
        param = self.param
        A_full, A_full_T, GT, G = self.A_full, self.A_full_T, self.GT, self.G

        # Start state of every plan. Default: the measured position and
        # velocity, zero acceleration. With mpc_tight_tracking (as the flight
        # code, Crazy_Flies/swarm_MPC_voronoi_sim_target/mpc.py): the previous
        # plan one control period later while the drone is within
        # replan_reset_dist of it, else the measured state with the velocity
        # clipped to 0.95 v_max; acceleration of the plan / last setpoint.
        # ADMM multipliers: carried over from the previous step (warm start)
        # when param.sim.admm_warm_multipliers (default: True in the Lisbon
        # simulation, False with mpc_tight_tracking, as the flight code,
        # where they wind up while the drones execute the bent plans), else
        # reset to zero at every step.
        tight = getattr(param.sim, "mpc_tight_tracking", False)
        warm = getattr(param.sim, "admm_warm_multipliers", None)
        warm = (not tight) if warm is None else warm
        b_eqs = {}
        for iD in active_ids:
            agent = self.chaser_list[iD]
            c = agent.constraints
            if not warm:
                c.lambda_obs[:] = 0.0
                c.lambda_ineq[:] = 0.0
            if not tight:
                b_eqs[iD] = np.vstack([agent.position, agent.velocity, agent.acceleration]).reshape(-1,1)
                continue
            p0 = agent.position.ravel()
            v0 = clip_norm(agent.velocity.ravel(), 0.95 * param.sim.v_max)
            a0 = clip_norm(agent.desired.acceleration.ravel(), 0.95 * param.sim.a_max)
            if c.C is not None:
                p, v, a = bernstein_eval(c.C, param.sim.dTi, param.sim.n_degree)
                a0 = clip_norm(a, 0.95 * param.sim.a_max)
                if np.linalg.norm(p - p0) < param.sim.replan_reset_dist:
                    p0, v0 = p, clip_norm(v, 0.95 * param.sim.v_max)
            b_eqs[iD] = np.concatenate([p0, v0, a0]).reshape(-1, 1)

        active_agents = list(active_ids)
        for _ in range(self.mpc_iters):
            for iD in active_agents[:]:
                agent = self.chaser_list[iD]
                c = agent.constraints
                # =====================================================
                # Equality constraint becomes current position
                # =====================================================
                b_eq = b_eqs[iD]

                # Step 1 - solve trajectory (only the RHS changes, KKT is factorized)
                b_bar = (
                    - c.q
                    - A_full_T @ c.lambda_obs
                    + param.sim.rho_obs * A_full_T @ c.b_obs
                    - GT @ c.lambda_ineq
                    + param.sim.rho_ineq * GT @ c.s
                )

                solution = lu_solve(self.lu, np.concatenate([b_bar, b_eq]))
                x = solution[:self.dim]                  # (dim, 1)
                c.C = x.reshape(self.n_ctrl, 3)

                # Step 2 - Update b_obs (other agents + airspace, vectorized):
                # standard scaled ADMM projection b_o = Pi(A_o z + lambda_o / rho_o),
                # so lambda_o decays once the constraint is no longer active
                # (the unscaled Pi(A_o z) used before froze lambda_o at its
                # last value and biased the plans away from the constraints)
                self.horizons[iD] = self.W0 @ c.C
                c.b_obs = self.b_total(iD, A_full @ x + c.lambda_obs / param.sim.rho_obs)

                # Step 3 - Update vel/acc targets (projection onto the balls)
                c.s = project_vel_acc(G @ x + c.lambda_ineq/param.sim.rho_ineq, param)

                # calculate residuals
                residual_eq   = self.A_eq @ x - b_eq
                residual_ineq = G @ x - c.s
                residual_obs  = A_full @ x - c.b_obs

                # Step 4 - Update lambdas (kept across iterations; across time steps if warm)
                c.lambda_ineq += param.sim.rho_ineq * residual_ineq
                c.lambda_obs  += param.sim.rho_obs * residual_obs

                eq_norm = np.linalg.norm(residual_eq)
                ineq_norm = np.linalg.norm(residual_ineq)
                obs_norm = np.linalg.norm(residual_obs)

                if (
                    eq_norm + ineq_norm + obs_norm < 1e-2
                ):
                    active_agents.remove(iD)
