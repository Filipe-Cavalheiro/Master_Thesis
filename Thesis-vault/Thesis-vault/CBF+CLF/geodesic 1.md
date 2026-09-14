## General HOCBF definition:
h: barrier function
f: is system
g: actuation equations
v: velocity $v = \dot{q}$ 
$$
\sup_{u \in U} \left[
\mathcal{L}_f^{m} h(x,t)
+ \mathcal{L}_g \mathcal{L}_f^{m-1} h(x,t)\, u
+ \frac{\partial^{m} h(x,t)}{\partial t^{m}}
+ O(h(x,t))
+ \alpha_m\!\left(\psi_{m-1}(x,t)\right)
\right] \ge 0
$$
a control input u(t) should satisfy
$$
\begin{align}
\mathcal{L}^2_fh(x(t)) + \mathcal{L}_g\mathcal{L}_fh(x(t))u(t)+2h(x(t))\mathcal{L}_fh(x(t))+\\(\mathcal{L}_fh(x(t)))^2+2h^2(x(t))\mathcal{L}_fh(x(t))\mathcal{L}_fh(x(t))+h^4(x(t)) \ge 0
\end{align}
$$

System State definition of motion on the two-sphere:
$$
\dot{q} = S(\omega) q
$$
$$
J \dot \omega = u - c \omega- S(\omega)J \omega+\tau
$$
$$
\textbf{WRONG} \rightarrow \dot{\begin{bmatrix}
q \\ \omega
\end{bmatrix}} =
\underbrace{
\begin{bmatrix}
S(\omega) & 0\\ 
0 & J^{-1}(- c - S(\omega)J)
\end{bmatrix}}_{f(x)}
\begin{bmatrix}
q \\ \omega
\end{bmatrix} + 
\underbrace{\begin{bmatrix}
0 \\ j^{-1}
\end{bmatrix}}_{g(x)}
\begin{bmatrix}
0 \\ u+\tau
\end{bmatrix}
$$
Explanation: Since the system is not linear it is not possible to separate the system in the 4 usual blocks having $\dot{x} = Ax+Bu$ the system is directly $\dot{x} = f(x)+ g(x)u$

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
\end{bmatrix}}_{g(x)}(u+\tau)
$$

where q is the position, $\omega$ is angular velocity, J is inertia, c is dampening factor, $\tau$ is an external factor.

Barrier function:
$$
h = q^\top q_{obstacle} - cos(\theta)
$$
where r is the radius to avoid and $q_{obstacle}$ is the centre of the obstacle, and $\theta$ is calculated as:
$$
\theta = \arccos(1-\frac{r_{safe}^2}2);
$$
In order to calculate the second order CBF, $\nabla h$ $\dot{h}$ are necessary being
$$
\dot{h}= -\dot{q}^\top q_{obstacle} = 
-(S(\omega)q)^\top q_{obstacle}
$$
### First-order Derivatives
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

### Second-Order Derivatives

$$  
\ddot h = \mathcal{L}_f^2 h + \mathcal{L}_g \mathcal{L}_f h u  
$$

$$
\mathcal{L}^2_gh = 0
$$
---
$$
\mathcal{L}^2_fh = \nabla (-S(\omega)q)^\top q_{obstacle})^\top f(x) = 
\begin{bmatrix} 
	\frac{\partial -S(\omega)q)^\top q_{obstacle}}{\partial q} \\
	\frac{\partial -S(\omega)q)^\top q_{obstacle}}{\partial \omega} 
\end{bmatrix}^\top f(x)
$$
#### Calculate partial derivative of q
$$
	a = -q_{obstacle}^\top (S(\omega)q) 
	= (S(\omega)q)^\top q_{obstacle}
	= - q^\top S(\omega) q_{obstacle}
$$ 
$$
\frac{\partial a}{\partial q}
= - S(\omega) q_{obstacle}
$$
#### Calculate partial derivative of $\omega$
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

---
$$
\mathcal{L}^2_fh = \begin{bmatrix} 
-S(\omega) q_{obstacle} &
(S(q_{obstacle}) q) 
\end{bmatrix}\cdot
\begin{bmatrix}
S(\omega)q \\ 
J^{-1}(- c\omega - S(\omega)J\omega)
\end{bmatrix}
$$
$$
\mathcal{L}^2_fh = (-S(\omega) q_{obstacle})^\top (S(\omega)q) 
+ (S(q_{obstacle}) q)^\top (J^{-1}(-c\omega - S(\omega)J\omega))
$$
---
$$
\mathcal{L}_g \mathcal{L}_f h = \nabla
\left( -q_{obstacle}^\top (S(\omega)q) \right)^\top
g(x)
$$
$$
\mathcal{L}_g \mathcal{L}_f h = \begin{bmatrix} 
-S(\omega) q_{obstacle} &
-(S(q_{obstacle}) q)
\end{bmatrix}^\top 
\begin{bmatrix}
0 \\ j^{-1}
\end{bmatrix} 
= -(S(q_{obstacle}) q)^\top J^{-1}
$$
### Actuation
For class-$\mathcal{K}$ gains $\alpha_0,\alpha_1 > 0$:

$$  
\mathcal{L}_f^2 h  
+  
(\mathcal{L}_g \mathcal{L}_f h) u  
+  
k_1 \mathcal{L}_f h  
+  
k_0 h  
\ge 0  
$$

so: $Au \le b$ 
$$ 
A = \mathcal{L}_g \mathcal{L}_f h  \qquad
b = -\mathcal{L}^2_fh - k1*\dot{h} - k0*h
$$
Half plane projection:

$$
u =  
\begin{cases}  
u_{\text{nom}}, & \text{if } A u_{\text{nom}} \le b \\[8pt]  
u_{\text{nom}} - \dfrac{A u_{\text{nom}} - b}{A A^{\top}} A^{\top}, & \text{otherwise}  
\end{cases}  
$$
# Control Lyapunov function

Consider the Lyapunov function candidate
 $$  
V(\hat{q}, \omega)
=
\frac{1}{2}\omega^\top I_q \omega  
$$

Since $I_q$ is positive definite about $\hat{q}=\hat{q}^\star$, the function $V$ is positive definite about the equilibrium $(\hat{q}, \omega) = (\hat{q}^\star, 0)$.
## Time Derivative of $V$

The time derivative of $V$ along the system trajectories is
$$  
\dot{V}
=
\omega^\top I_q \dot{\omega}  
$$

$$  
\dot{V}  
= \omega^\top (- K_r e_q - K_\omega \omega) 
$$

$$  
= -\omega^\top K_r e_q - \omega^\top K_\omega \omega
$$

Since $K_\omega$ is positive definite, satisfying

$$  
\lambda_{\min}(K_\omega)  
\le x^\top K_\omega x  
\le \lambda_{\max}(K_\omega),  
\qquad \forall x \neq 0,  
$$

it follows that
$$  
\dot{V}  
\le  
-\lambda |\omega|^2  
\le  
0,  
$$

where $\lambda = \lambda_{\min}(K_\omega) > 0$.
Hence, $\dot{V}$ is negative semidefinite.

$$
\mathcal{L}_fV = \nabla V^\top f
$$
# $$
\nabla V = 
\begin{bmatrix}
0\\
\omega^\top J
\end{bmatrix}
$$
## First order Lie derivatives
$$
\mathcal{L}_fV  = 
\begin{bmatrix}
0& \omega^\top J
\end{bmatrix}
\begin{bmatrix}
S(\omega)q\\ 
J^{-1}(- c - S(\omega) J ) \omega
\end{bmatrix} 
$$
since 
$$
u_i
= - K_r e_{q_i} - K_\omega e_{\omega_i} + S(\omega_i) I_q \omega_i + c \, \omega_i
$$
$$
\begin{aligned}

\dot{\hat{q}}_i &= S(\omega_i)  \hat{q}_i \\

\dot{\omega}_i &= J^{-1}(- K_r e_{q_i} - K_\omega e_{\omega_i})

\end{aligned}
$$
$$
\mathcal{L}_fV  = 
\begin{bmatrix}
0 & \omega^\top J
\end{bmatrix}
\begin{bmatrix}
S(\omega)q\\ 
J^{-1}(- K_r e_{q_i} - K_\omega e_{\omega_i})
\end{bmatrix} 
$$
$$
\mathcal{L}_fV = (\nabla_qV)^\top S(\omega)q+(J \omega)^\top J^{-1}(- K_r e_{q_i} - K_\omega e_{\omega_i})
$$
Since
$$
(\nabla_qV)^\top S(\omega)q = K_re_q^\top \omega
$$
$$
\mathcal{L}_fV 
= (J \omega)^\top J^{-1}(- K_r e_{q_i} - K_\omega e_{\omega_i})
$$
$$
\mathcal{L}_fV = -\omega^\top K_r e_{q_i}-\omega^\top K_\omega \omega = \dot{V}
$$

$$
\mathcal{L}_gV = \nabla V^\top g(x) = 
\begin{bmatrix}
0 & \omega^\top J
\end{bmatrix}
\begin{bmatrix}
0 \\ j^{-1}
\end{bmatrix}
$$

$$\mathcal{L}_gV 
= (\omega^\top J)^\top J^{-1} = \omega^\top
$$ 

