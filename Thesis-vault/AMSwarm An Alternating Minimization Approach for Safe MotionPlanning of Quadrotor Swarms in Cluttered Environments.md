
## I A - Optimization Problem Formulation
pos - goal + quadratic acceleration regularization to encourage smooth trajectories. (1a)
max and min pos vel and acc (1b) - (1e)
barrier function (1f) 

The barrier function Eq(2) constrains **how quickly the safety measure is allowed to decrease**, not directly the physical velocity. 
- It enforces **forward invariance** of the safe set.
- In other words, if the robots are currently safe, they cannot approach each other faster than the barrier dynamics allow.
For example assume that theta is Identity making so that the barrier functions actuates if the position between agent and barrier is less then 1 meter, the following values for the horizon:

t0 = 3.5 m
t1 = 3 m
t2 = 2 m
t3 = 1.5 m
t4 = 1.1 m

All satisfy the Eq (1f) but the reduction of distance between agent is barrier might be so fast that it no longer upholds the Eq(2) this basically makes so that the agent reacts to a possible future collision earlier without having to increase the horizon to a value that might take a lot of computation time.

## I B - Trajectory Parameterize

The parameterize of the trajectory based on the positions / velocity's / accelerations of the agent in future N positions (horizon)  cannot guarantee safety between discretization points, passing this points to the Bernstein polynomial makes so that they are defined via a continuous function  that has its coefficients convex. The optimization only enforces constraints at the discrete prediction nodes. Between those nodes the polynomial trajectory could, in principle, violate the constraints.

## I C - Challenges in Solving the Optimization Problem

The trajectory without any actuation of the BF is convex but the barrier function is not making so that the problem becomes  a non-convex quadratically constrained quadratic program (QCQP) the proposed solution to this is to pass the Eq 1 from Cartesian coordinates to Polar coordinates

Each sub-problem becomes either a convex quadratic program or admits a closed-form solution, making the overall optimization significantly faster than solving the original nonlinear problem. Eq(2) does not help since it is non-convex

## II A - Constraints Reformulation

The main addition to the state of the art is made via a constraint reformulation made by deriving the collision avoidance constraints in polar coordinates.

No new optimization technique is introduced here; this section simply reformulates the geometry of the collision constraints so that they can later be separated into simpler optimization variables.

## II B - Reformulated Problem

There are a lot of things that are not explicitly explained 

$$Q = \omega_G W_g^\top W_g + \omega_s \ddot W^\top \ddot W$$
$$q^\top = -2 \omega_g p_d^\top \omega_g$$
Since the optimization predicts an entire trajectory, (p_d) is not a single goal position. Instead, it is a matrix containing the desired position for every prediction step, with dimensions $3 \times N,$ where (N) is the prediction horizon.

## II C - Relaxation and Solution by Alternating Minimization

Equation (10) is obtained by introducing an augmented Lagrangian for the reformulated optimization problem.

The augmented Lagrangian combines

- Lagrange multipliers, which enforce equality constraints,
- quadratic penalty terms, which improve convergence.

Although the formulation closely resembles the standard ADMM augmented Lagrangian, the paper applies it within an alternating minimization framework rather than a classical ADMM algorithm.

One thing I think is worth adding to your notes is **why they introduce the auxiliary variables** $\zeta_2$​ and $\zeta_3$​.

That's arguably the central idea of the paper.

Specifically,

- $\zeta_1$​ contains the trajectory variables,
- $\zeta_2$​ represents collision directions,
- $\zeta_3$​ represents collision distances.

Separating these variables removes the nonlinear coupling and allows the optimization problem to be solved through alternating minimization. By decoupling the nonlinear components, each subproblem becomes convex and can be solved efficiently. Without this reformulation, the problem would require Sequential Convex Programming (SCP), which repeatedly linearizes the nonlinear constraints and solves multiple optimization problems, resulting in significantly higher computational cost.

The quadratic optimization problem is therefore written as

$$
\bar{A} = Q + \rho_1 A^\top A + \rho_2 G^\top G
$$

and

$$
\bar{b}
=
-q
+
\underbrace{\rho_1 A^\top b - A^\top \lambda}_{\text{Equality constraints}}
+
\underbrace{\rho_2 G^\top (h-s) - G^\top \mu}_{\text{Inequality constraints}}.
$$

The slack variable \(s\) is introduced to transform inequality constraints into equality constraints. Instead of enforcing
$$
Gx \leq h,
$$

the optimization considers
$$
Gx + s = h,
\qquad s \geq 0,
$$
where \(s\) measures the amount by which the inequality is not active. This formulation enables the use of equality-constrained optimization techniques while still respecting the original inequalities.

### Equality Constraint Matrix

The matrix \(A\) encodes the initial-state constraints of the trajectory optimization. It ensures that the generated polynomial trajectory starts from the agent's current state by enforcing the initial position, velocity, and acceleration.

Specifically,
$$
[p(0),\,v(0),\,a(0)] = [p_{\mathrm{start}},\,v_{\mathrm{start}},\,a_{\mathrm{start}}]
$$

where the polynomial basis functions evaluated at \(t=0\) are used to constrain the optimization variables.

The resulting constraint matrix has the block structure
$$
A =
\begin{bmatrix}
W_0(0) & 0 & 0\\
0 & W_0(0) & 0\\
0 & 0 & W_0(0)\\
\hline
W_1(0) & 0 & 0\\
0 & W_1(0) & 0\\
0 & 0 & W_1(0)\\
\hline
W_2(0) & 0 & 0\\
0 & W_2(0) & 0\\
0 & 0 & W_2(0)
\end{bmatrix},
$$

where:

- $W_0(0)$ evaluates the polynomial basis at \(t=0\), constraining the initial position;
- $W_1(0)$ evaluates the first derivative of the basis, constraining the initial velocity;
- $W_2(0)$ evaluates the second derivative, constraining the initial acceleration.

### Inequality Constraint Matrix
There are three inequalities to take into consideration, the maximal velocity, the maximal acceleration and most importantly the collision avoidance or maximal distance. This three elements are part of the $G$ matrix

---
Velocity Constraints
The velocity at each time step is obtained through the Bernoulli basis matrix $W_1$, representing the first derivative basis
$$
v = W_1 \zeta_1  
$$

For each spatial axis $(x,y,z)$, the velocity is bounded by a maximum value $v_{\max}$,

$$
-v_{\max} \leq v_i(k) \leq v_{\max},  
\qquad  
i \in {x,y,z},  
\quad  
k=0,\ldots,H-1.  
$$
This constraint stand alone could be represented as
$$
G_{vel} \zeta_1 \le v_{max}
$$
$$
-G_{vel} \zeta_1 \le v_{max} 
$$
where $G_{vel}$ is the $W_ 1$

---
Acceleration Constraints
Similarly, acceleration is computed using the second-derivative mapping matrix (W_2),
$$a = W_2 \zeta_1$$

The acceleration magnitude along each axis is limited by

$$
-a_{\max} \leq a_i(k) \leq a_{\max},  
\qquad  
i \in {x,y,z},  
\quad  
k=0,\ldots,H-1.  
$$

Again, this is converted into two linear inequalities:

$$
a_i(k) \leq a_{\max} \Leftrightarrow W_2 \zeta_1 \leq a_{\max}
$$
$$
-a_i(k) \leq a_{\max} \Leftrightarrow  -W_2 \zeta_1 \leq a_{\max}
$$

Consequently, the matrices $G_{\mathrm{acc}}$ and $-G_{\mathrm{acc}}$ are added to the global inequality matrix.
 
--- 
Obstacle Avoidance Constraints
The obstacle avoidance constraint is obtained by linearizing the non-convex distance constraint around the current trajectory estimate.

Ideally, the drone should remain outside a safety sphere of radius (R),

[  
|p(k)-p_{\mathrm{obs}}| \geq R,  
]

where:

- (p(k)) is the predicted position at time step (k),
    
- (p_{\mathrm{obs}}) is the obstacle center.
    

This constraint is nonlinear because of the Euclidean norm.

To obtain a convex approximation, the constraint is linearized around the current trajectory point (p_k). The outward normal direction is computed as

# [  
n_k

\frac{p_k-p_{\mathrm{obs}}}  
{|p_k-p_{\mathrm{obs}}|}.  
]

In the implementation, the normal is approximated as

# [  
n_k

\frac{|p_k-p_{\mathrm{obs}}|}  
{||p_k-p_{\mathrm{obs}}||}.  
]

Using a first-order Taylor approximation, the nonlinear constraint becomes

[  
n_k^\top \bigl(p-p_k\bigr)  
+  
|p_k-p_{\mathrm{obs}}|  
\geq R.  
]

Rearranging,

[  
n_k^\top p  
\geq  
R-|p_k-p_{\mathrm{obs}}|  
+  
n_k^\top p_k.  
]

Expressing the position as

[  
p = W_0(k)x,  
]

yields a linear inequality in the optimization variables,

[  
-n_k^\top W_0(k)x  
\leq  
-(R-|p_k-p_{\mathrm{obs}}|).  
]

This corresponds exactly to the code:

```python
G_obs.append(-row)
h_obs.append(-(R - dist))
```

Each prediction step contributes one linearized obstacle constraint, creating a supporting hyperplane that pushes the optimized trajectory away from the obstacle.

---

## Final Constraint Matrix

All inequalities are stacked together as

$$
G=  
\begin{bmatrix}  
G_{\mathrm{vel}}\  
-G_{\mathrm{vel}}\  
G_{\mathrm{acc}}\  
-G_{\mathrm{acc}}\  
G_{\mathrm{obs}}  
\end{bmatrix},  
$$

with

$$
h=  
\begin{bmatrix}  
v_{\max}\  
v_{\max}\  
a_{\max}\  
a_{\max}\  
h_{\mathrm{obs}}  
\end{bmatrix}
$$

The resulting optimization problem therefore guarantees:

1. **Velocity limits** are respected at every prediction step.
2. **Acceleration limits** are respected at every prediction step.
3. **Obstacle avoidance** is enforced through a sequence of linearized safety constraints.

The obstacle constraints are the only constraints that depend on the current trajectory estimate, which is why they are updated at every iteration of the alternating minimization procedure.

## Alternating Minimization Steps

The trajectory variables ((\zeta_1)) are updated by solving a convex quadratic program.

This step closely resembles the quadratic programming update presented in Section 5.2 ("Linear and Quadratic Programming") of _Distributed Optimization via Alternating Direction Method of Multipliers_.

Step 2,3  are very straight forward 

Step 4 This is just the previously mentioned slack term in order to have inequalities

Step 5 is the update of the Lagrangian multiplier

There appears to be a missing factor of two in Algorithm 1. Differentiating the augmented Lagrangian yields. The correct equation does not have the $\frac \rho 2$ but only $\rho$ having the  $\lambda^{l+1}$  given by

$$
\lambda^{l+1} = \lambda^l - \frac{\partial }{\partial \zeta_1} \frac\rho 2\|A \zeta_1-b(\zeta_2,\zeta_3)\|^2 - \frac{\partial }{\partial \zeta_1} \frac\rho 2\|G -h+s^{l+1}\|^2
$$

$$
\lambda^{l+1} = \lambda^l - \rho A^\top (A \zeta_1-b(\zeta_2,\zeta_3)) -  \rho G^\top(G -h+s^{l+1})
$$

