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
h = r^2-\|q-q_{obstacle}\|^2
$$
where r is the radius to avoid and $q_{obstacle}$ is the centre of the obstacle. 
In order to calculate the second order CBF, $\nabla h$ $\dot{h}$ are necessary being
$$
\dot{h}=2(q-q_{obstacle})^\top \frac{\partial (q-q_{obstacle})}{\partial t} = 2(q-q_{obstacle})^\top \dot{q} = 2(q-q_{obstacle})^\top S(\omega)q
$$
$$
\nabla h = 
\begin{bmatrix} 
\frac{\partial h}{\partial q}\\
\frac{\partial h}{\partial \omega}
\end{bmatrix}
= 
\begin{bmatrix} 
2(q-q_{obstacle})\\
0
\end{bmatrix}
$$

## First-order Derivatives
$$
\boxed{ 
\mathcal{L}_fh = \nabla h^\top \cdot f(x) = 
\begin{bmatrix} 
2(q-q_{obstacle})&0
\end{bmatrix}
\begin{bmatrix}
S(\omega)q\\ 
J^{-1}(- c - S(\omega) J ) \omega
\end{bmatrix}
= 2(q-q_{obstacle})^\top S(\omega) q
}
$$


$$
\boxed{ 
\mathcal{L}_gh = \nabla h^\top g(x) = \begin{bmatrix} 
2(q-q_{obstacle})&0
\end{bmatrix}
\begin{bmatrix}
0 \\ j^{-1}
\end{bmatrix} = 0
}
$$

## Second-Order Derivatives

$$  
\ddot h = \mathcal{L}_f^2 h + \mathcal{L}_g \mathcal{L}_f h u  
$$

$$
\boxed{ 
\mathcal{L}^2_gh = 0
}
$$
---
$$
\mathcal{L}^2_fh = \nabla (2(q-q_{obstacle})S(\omega)q)^\top f(x) = 
\begin{bmatrix} 
	\frac{\partial (2(q-q_{obstacle})S(\omega)q)}{\partial q} \\
	\frac{\partial (2(q-q_{obstacle})S(\omega)q)}{\partial \omega} 
\end{bmatrix}^\top f(x)
$$
### calculate partial derivative of q
$$
	a = -2(q-q_{obstacle})^\top S(\omega)q
$$
$$
\frac{\partial a}{\partial q} = \frac{\partial -2q^\top S(\omega) q+2q_{obstacle}^\top S(\omega)q}{\partial q}
$$
$$  
\text{Note:} \quad\frac{\partial}{\partial q}\left(q^\top A q \right) 
= (A + A^\top) q  
$$

$$
\frac{\partial}{\partial q} \left(-2 q^\top S(\omega) q\right)
= - 2 (S(\omega) + S(\omega)^\top) q  
$$

Since in the skew matrix $S^\top = -S$  is anti-symmetric, this simplifies to:

$$
\boxed{ 
-2 (S(\omega) - S(\omega)) q  = 0
}
$$
$$
2 q_{\text{obstacle}}^\top S(\omega) q  
$$

This is a scalar. Using the cross-product identity:

$$
q_{\text{obstacle}}^\top S(\omega) q  
= q_{\text{obstacle}}^\top (\omega \times q)  
= (q_{\text{obstacle}} \times \omega)^\top q  
$$

So we rewrite:

$$
\frac{\partial (2q_{obstacle}^\top S(\omega)q)}{\partial q}
$$

$$
f(q) = a^\top q  
$$
we know:
$$
\frac{\partial}{\partial q} (a^\top q) = a  
$$

Thus,
$$ 
\frac{\partial}{\partial q}  
\left( 2 (q_{\text{obstacle}} \times \omega)^\top q \right)
= 2 (q_{\text{obstacle}} \times \omega)  
$$
All three forms are equivalent.
$$
\frac{\partial (2q_{obstacle}^\top S(\omega)q)}{\partial q} = 2q_{obstacle}^\top S(\omega) = 2 S(\omega)^\top q_{obstacle} = 2 S(\omega) q_{obstacle}
$$
The final value for the partial derivative is
$$
\frac{\partial (-2q^\top S(\omega) q+2q_{obstacle}^\top S(\omega)q)}{\partial q} =  
2 S(\omega) q_{obstacle}
$$
### calculate partial derivative of $\omega$

$$
\frac{\partial a}{\partial \omega} = \frac{\partial (2q^\top S(\omega) q+2q_{obstacle}^\top S(\omega)q)}{\partial \omega}
$$
$$
\frac{\partial 2q^\top S(\omega) q}{\partial \omega} = 0 
$$
This comes from the fact that the dot product of a cross product is always zero, since the cross product creates a vector that is perpendicular to the first two the dot product is obligatory zero.
$$
2q_{obstacle}^\top S(\omega)q = 2q_{obstacle} \cdot (\omega \times q)
$$
From [[Internal and external product]] using the Scalar triple product $a \cdot (b \times c) = b \cdot (c \times a) = c \cdot (a \times b)$	
$$
2q_{obstacle}^\top (S(\omega) q)) = 2 \omega^\top (S(q) q_{obstacle}))
$$
$$
\frac{\partial (2 \omega^\top (S(q)  q_{obstacle}))}{\partial \omega} = 2(q \times q_{obstacle}) 
$$
The final value for the partial derivative is
$$
\frac{\partial (2q^\top S(\omega) q+2q_{obstacle}^\top S(\omega)q)}{\partial \omega} = 2(q \times q_{obstacle}) 
$$
---
$$
\mathcal{L}^2_fh = \begin{bmatrix} 
2 S(\omega) q_{obstacle}&2(q \times q_{obstacle}) 
\end{bmatrix}\cdot
\begin{bmatrix}
S(\omega)q\\ 
J^{-1}(- c\omega - S(\omega)J\omega)
\end{bmatrix}
$$
$$
\mathcal{L}^2_fh = (2 S(\omega) q_{obstacle})^\top S(\omega)q + 2(S(q) q_{obstacle})^\top J^{-1}(-c\omega -S(\omega)J\omega)
$$
---
$$
\mathcal{L}_g \mathcal{L}_f h = \nabla(2(q-q_{obstacle})^\top S(\omega) q)\cdot g(x)
$$
$$
\mathcal{L}_g \mathcal{L}_f h = \begin{bmatrix} 
2 S(\omega) q_{obstacle}&2(S(q) q_{obstacle}) 
\end{bmatrix}\cdot \begin{bmatrix}
0 \\ j^{-1}
\end{bmatrix} = 2(S(q) q_{obstacle})^\top J^{-1}
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
