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
obstacle = $q_o$
current position = $q$

$$
\frac{S(q_o)q}{\sqrt{s(1+q^\top q_o)}} \Leftrightarrow \\
$$

$$
\frac{\sin\theta}{\sqrt{2(1+\cos\theta)}}
=
\sqrt{\frac{\sin^2\theta}{2(1+\cos\theta)}}
$$

Using the identity:

$$
\sin^2\theta = 1 - \cos^2\theta
$$

we get:

$$
=
\sqrt{\frac{1 - \cos^2\theta}{2(1+\cos\theta)}}
$$

Factor:

$$
=
\sqrt{\frac{(1-\cos\theta)(1+\cos\theta)}{2(1+\cos\theta)}}
$$

Cancel $(1+\cos\theta)$:
$$
=
\sqrt{\frac{1-\cos\theta}{2}} = \sin(\frac{\theta}{2})
$$

# Actuation
For class-$\mathcal{K}$ gains $\alpha_0,\alpha_1 > 0$:

$$  
\mathcal{L}_f^2 h  
+  
\mathcal{L}_g \mathcal{L}_f h, u  
+  
\alpha_1 \mathcal{L}_f h  
+  
\alpha_0 h  
\ge 0  
$$

$$ 
A = -\nabla h^{\top}  \qquad
b = \mathcal{L}^2_fh + k1*\dot{h} + k0*h
$$
$$ 
A = \mathcal{L}_g \mathcal{L}_f h  \qquad
b = \mathcal{L}^2_fh + k1*\dot{h} + k0*h
$$
Half plane projection:

$$
u =  
\begin{cases}  
u_{\text{nom}}, & \text{if } A u_{\text{nom}} \le b \\[8pt]  
u_{\text{nom}} - \dfrac{A u_{\text{nom}} - b}{A A^{\top}} A^{\top}, & \text{otherwise}  
\end{cases}  
$$
