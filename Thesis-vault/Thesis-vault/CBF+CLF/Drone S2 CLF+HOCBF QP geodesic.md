The following writing has as purpose to guide a point $q$ to a destination $q^\star$ with out collision with an obstacle $q_\text{obstacle}$ of a certain radius $r_{safe}$. The system evolves and is bound to the unit sphere $\mathbb{S}^2$.  
## General HOCBF definition:
As will be later shown via the equations the control barrier function is required to be of second order since the actuation term does not appear in the system in $\mathcal{L}_gh$.

h: barrier function
f: is system
g: actuation equations
v: velocity $v = \dot{q}$ 

The general higher order system can be defined as:
$$
\sup_{u \in U} \left[
\mathcal{L}_f^{m} h(x,t)
+ \mathcal{L}_g \mathcal{L}_f^{m-1} h(x,t)\, u
+ \frac{\partial^{m} h(x,t)}{\partial t^{m}}
+ O(h(x,t))
+ \alpha_m\!\left(\psi_{m-1}(x,t)\right)
\right] \ge 0
$$
being our case for $m=2$. 
Since the system is not linear it is not possible to separate the system in the 4 usual blocks having $\dot{x} = Ax+Bu$ the system is directly $\dot{x} = f(x)+ g(x)u$

$$
\dot{\begin{bmatrix}
q \\ \omega
\end{bmatrix}} =
\underbrace{
\begin{bmatrix}
S(\omega)q\\ 
J^{-1}(- c\omega - S(\omega)J\omega)
\end{bmatrix}}_{f(x)}
+ 
\underbrace{\begin{bmatrix}
0 \\ j^{-1}
\end{bmatrix}}_{g(x)}
(u+\tau)
$$

where q is the position, $\omega$ is angular velocity, J or $I_q$ is inertia (I switch between them sometimes), c is dampening factor, $\tau$ is an external factor.

The chosen barrier function is:
$$
h = sin(\theta) - q^\top q_{obstacle}
$$
where $r_{safe}$ is the radius to avoid, $q_{obstacle}$ is the centre of the obstacle and $\theta$ is calculated as:
$$
\theta = \pi r_{safe}
$$
This comes from the fact that from the literature for avoiding a undesired region $h \ge 0$ so from this the barrier function $h = sin(\theta) - q^\top q_{obstacle}$ is chosen since when $q$ get to close to the obstacle $q_\text{obstacle}$ the dot product between them increases leading to the negative term to increase if this term is greater than the $\sin(\theta)$ then $h$ would no longer be positive. The CBF will enforce that to never happen being at most $h = 0$. Ah and its $\sin$ not $\cos$ for a simple reason when the radius of the obstacle increases and theta also increases I want for the value to increase making so that $\sin$ makes more sense, it is also $\pi r_{safe}$ since when the radius is maximum at 1 then it is basically around the centre of the unit sphere so it makes sense for $sin(\theta)$ also be maximal.
# Calculating the CBF
$$
\dot{h}= -\dot{q}^\top q_{obstacle} = 
-(S(\omega)q)^\top q_{obstacle}
$$
It is a well known fact that $\dot{h} = \mathcal{L}_fh$ so having $\dot{h}$ calculated we can verify its correctness via the calculations of the [[Lie Derivatives]].
## First-order Derivatives
$$
\mathcal{L}_fh = \dot{h} = \nabla h \cdot f(x) = 
\begin{bmatrix} 
-q_{obstacle}&0
\end{bmatrix}
\begin{bmatrix}
S(\omega)q\\ 
J^{-1}(- c - S(\omega) J ) \omega
\end{bmatrix}
= -q_{obstacle}^\top (S(\omega)q)
$$
$$
\mathcal{L}_gh = \nabla h^\top g(x) = \begin{bmatrix} 
-q_{obstacle}&0
\end{bmatrix}
\begin{bmatrix}
0 \\ j^{-1}
\end{bmatrix} = 0
$$
As we can see $\dot{h} = \mathcal{L}_fh$ and since $\mathcal{L}_gh=0$ it is required to calculate the second order terms.
## Second-Order Derivatives
$$
\begin{align}
	&\ddot h = \mathcal{L}_f^2 h + \mathcal{L}_g \mathcal{L}_f h\, u &\boxed{\mathcal{L}_g^2 h = 0}\\
	&\mathcal{L}_f^2 h = ?
	& \mathcal{L}_g \mathcal{L}_f h = ?
\end{align}
$$
---
$$
\mathcal{L}^2_fh = \nabla (-S(\omega)q)^\top q_{obstacle})^\top f(x) = 
\begin{bmatrix} 
	\frac{\partial -(S(\omega)q)^\top q_{obstacle}}{\partial q} \\
	\frac{\partial -(S(\omega)q)^\top q_{obstacle}}{\partial \omega} 
\end{bmatrix}^\top f(x)
$$
### Calculate partial derivative of q
$$
	a = -q_{obstacle}^\top (S(\omega)q) 
	= -(S(\omega)q)^\top q_{obstacle}
	= q^\top S(\omega) q_{obstacle}
$$ 
$$
\frac{\partial a}{\partial q}
= S(\omega) q_{obstacle}
$$
There is a rule in [[Transpose derivatives]] that indicates that $\frac{\partial x^\top A}{\partial x} = A$.
### Calculate partial derivative of $\omega$
$$
-q_{obstacle}^\top (S(\omega)q) = -q_{obstacle} \cdot (\omega \times q)
$$
From [[Internal and external product]] using the Scalar triple product $a \cdot (b \times c) = b \cdot (c \times a) = c \cdot (a \times b)$	
$$
-q_{obstacle} \cdot (\omega \times q)) = \omega \cdot (q \times -q_{obstacle}))
$$
$$
\frac{\partial (\omega^\top (S(q)(-q_{obstacle})))}{\partial \omega} 
= S(q)(-q_{obstacle}) 

$$
since $a\times b$ = $-b\times a$
$$
S(q)(-q_{obstacle}) = S(q_{obstacle})(q)
$$

Chat GPT says that:
$$
\frac{\partial \left( -q_{obstacle}^\top S(\omega)q \right)}{\partial \omega} 
= -q_{obstacle}^\top S(q)
$$
This term calculated by Chat GPT has the exact same values as my calculations it is only transposed being my results 3x1 and his 1x3.

Having both the partial derivatives the Lie derivative can be given by
$$
\mathcal{L}^2_fh = \begin{bmatrix} 
S(\omega) q_{obstacle} &
(S(q_{obstacle}) q) 
\end{bmatrix}\cdot
\begin{bmatrix}
S(\omega)q \\ 
J^{-1}(- c\omega - S(\omega)J\omega)
\end{bmatrix}
$$
$$
\boxed{
\mathcal{L}^2_fh = (S(\omega) q_{obstacle})^\top (S(\omega)q) 
+ (S(q_{obstacle}) q)^\top (J^{-1}(-c\omega - S(\omega)J\omega))
}
$$
---
$$
\mathcal{L}_g \mathcal{L}_f h = \nabla
\left( -q_{obstacle}^\top (S(\omega)q) \right)^\top
g(x)
$$
$$
\mathcal{L}_g \mathcal{L}_f h = \begin{bmatrix} 
S(\omega) q_{obstacle} &
(S(q_{obstacle}) q)
\end{bmatrix} \cdot
\begin{bmatrix}
0 \\ j^{-1}
\end{bmatrix} 
= (S(q_{obstacle}) q)^\top J^{-1}
$$
$$
\boxed{ \mathcal{L}_g \mathcal{L}_f h= -q^\top S(q_\text{obstacle})J^{-1}}
$$

Small rant/break to understand what is to come: Ok here there is what I understand as a choice because there are two possible Quadratic Problems to solve, being those:
The Safety filter / CBF-QP
$$
\begin{align}
\min_{u} \quad & \|u - u_{\text{nominal}}\|^2 \\
\text{s.t.} \quad 
& \mathcal{L}_f^2 h(x) + \mathcal{L}_g \mathcal{L}_f h(x)\, u \ge -k_{1,\text{cbf}} \mathcal{L}_f h(x) - k_{0,\text{cbf}} h(x)
\end{align}
$$
Where we close the system with a $u_{\text{nominal}}$ being it in this case
$$
u_\text{nominal} = - K_r e_{q_i} - K_\omega e_{\omega_i} + S(\omega_i) I_q \omega_i + c \, \omega_i
$$
making so that the system state is 
$$
\begin{aligned}
\dot{q} &= S(\omega) \hat{q} \\
\dot{\omega} &= I_q^{-1}(- K_r e_{q} - K_\omega e_{\omega})
\end{aligned}
$$
and so the objective of the system is to create an actuation $u$ that is as close as possible to the nominal actuation $u_\text{nominal}$ while ensuring that the system does not collide via the CBF. Having the new maybe modified actuation $u$ that might just be the nominal depending if the CBF actuated its passed to the original system definition.
$$
\begin{align}
\dot{q} &= S(\omega) \hat{q} \\
\dot{\omega} &= I_q^{-1} (u - c \, \omega - S(\omega) I_q \omega)
\end{align}
$$
Meaning that the calculations end here! for the method below the CLF is still needed following that:
CLF+HOCBF - QP
we can chose the open loop version which is 
$$
\begin{align}
\min_{u} \quad & \|u\|^2 \\
\text{s.t.} \quad 
& \mathcal{L}_f V(x) + \mathcal{L}_g V(x)\, u \le -k_{\text{clf}} V(x) \\
& \mathcal{L}_f^2 h(x) + \mathcal{L}_g \mathcal{L}_f h(x)\, u \ge -k_{1,\text{cbf}} \mathcal{L}_f h(x) - k_{0,\text{cbf}} h(x)
\end{align}
$$
Being the state the same as the initially given state
$$
\begin{align}
\dot{q} &= S(\omega) \hat{q} \\
\dot{\omega} &= I_q^{-1} (u - c \, \omega - S(\omega) I_q \omega)
\end{align}
$$
With the objective of creating an actuation that goes toward the objective from the CLF and avoids obstacles from the CBF and is only minimized/zero at the objective. 

Now have I found indications saying which should I use? Nope...
But the papers I have seen usually at least one of this, the only thing that keep trying to use the first one for some reason is Chat GPT. Do I know which one is the best for my use case, I do not especially when trying both none of them seams to work.

The following section is the time and lie derivatives of V which only apply to the open loop case since it is the only one that depends on the CLF.
# Control Lyapunov function

Consider the Lyapunov function candidate
$$  
V(\hat{q}, \omega)
=
\frac{1}{2}\omega^\top I_q \omega +\Psi 
$$
Since $I_q$ is positive definite about $\hat{q}=\hat{q}^\star$, the function $V$ is positive definite about the equilibrium $(\hat{q}, \omega) = (\hat{q}^\star, 0)$.
## Time Derivative of $V$

The time derivative of $V$ along the system trajectories is
$$  
\dot{V}
=
\omega^\top I_q \dot{\omega} + \dot{\Psi}
$$

$$  
\dot{V} = \omega^\top ( u - c \omega - S(\omega) I_q \omega ) + e_q^\top \omega
$$

$$  
= \omega^\top u-\omega^\top c \omega - \omega^\top S(\omega) I_q \omega + e_q^\top\omega
$$
$$  
= \omega^\top u -\omega^\top c \omega + e_q^\top\omega, \quad \omega^\top S(\omega) \omega = 0
$$
$$
\boxed{
\dot{V} = \omega^\top u -\omega^\top c \omega + e_q^\top\omega
}
$$
Making so as will be calculated afterwards
$$
\begin{align}
\mathcal{L}_fV &= -\omega^\top c \omega + e_q^\top \omega\\
\mathcal{L}_gV &= \omega^\top 
\end{align}
$$
## Lie derivative of V
$$
\mathcal{L}_fV = \nabla V^\top f
$$
$$
\nabla V = 
\begin{bmatrix}
\frac{\partial \Psi}{\partial q}\\
J\omega
\end{bmatrix}
$$
$$
\mathcal{L}_fV  = 
\begin{bmatrix}
\frac{\partial \Psi}{\partial q}&
J\omega
\end{bmatrix}
\begin{bmatrix}
S(\omega)q\\ 
J^{-1}(-c\omega - S(\omega) I_q\omega)
\end{bmatrix}
$$
$$
\mathcal{L}_fV = \frac{\partial \Psi}{\partial q}^\top S(\omega)q + \omega^\top J^\top (-J^{-1} c \omega - J^{-1} S(\omega) I_q \omega)
$$

$$
\Psi = 2 - \sqrt{2 ( 1 + q^\top q^\star)}
$$
$$
\frac{\partial \Psi}{\partial q} = -\frac{q^\star}{\sqrt{2(1 + q^\top q^\star)}}
$$
$$
\mathcal{L}_fV =-\frac{{q^\star}^\top }{\sqrt{2(1 + q^\top q^\star)}}S(\omega)q + \omega^\top J^\top (-J^{-1}c\omega - J^{-1}S(\omega) I_q\omega)
$$
$$
-\frac{{q^\star}^\top S(\omega)q}{\sqrt{2(1 + q^\top q^\star)}} + \omega^\top J^\top (-J^{-1}c\omega - J^{-1}S(\omega) I_q\omega)
$$
From [[Internal and external product]] using the Scalar triple product $a \cdot (b \times c) = b \cdot (c \times a) = c \cdot (a \times b)$	

$$
-\frac{\omega^\top S(q)q^\star}{\sqrt{2(1 + q^\top q^\star)}} + \omega^\top J^\top (-J^{-1}c\omega - J^{-1}S(\omega) I_q\omega)
$$
$$
\omega^\top\frac{ S(q^\star)q}{\sqrt{2(1 + q^\top q^\star)}} + \omega^\top J^\top (-J^{-1}c\omega - J^{-1}S(\omega) I_q\omega)
$$
$$
\omega^\top e_q + \omega^\top J^\top (-J^{-1}c\omega - S(\omega) J^{-1}I_q\omega)
$$
$$
-\omega^\top J^\top J^{-1}c\omega - \omega^\top J^\top J^{-1}S(\omega) I_q\omega + e_q^\top\omega
$$
$$
-\omega^\top J^\top J^{-1}c\omega + e_q^\top\omega \,, \quad \omega^\top S(\omega) \omega = 0
$$
$$
\boxed{
\mathcal{L}_f V = -\omega^\top c\omega + e_q^\top\omega
}
$$

$$
\mathcal{L}_gV = \nabla V^\top g(x) = 
\begin{bmatrix}
\frac{\partial \Psi}{\partial q} & J \omega
\end{bmatrix}
\begin{bmatrix}
0 \\ j^{-1}
\end{bmatrix}
$$
$$
\boxed{
\mathcal{L}_gV 
= \omega^\top J^\top J^{-1} = \omega^\top
}
$$

