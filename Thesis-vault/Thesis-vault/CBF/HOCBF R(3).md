$$
x = 
\begin{bmatrix} 
p\\ v
\end{bmatrix}=

\begin{bmatrix} 
p_x\\
p_y\\
p_z\\
v_x\\
v_y\\
v_z
\end{bmatrix}
$$

Dynamics:

$$
\dot{p} = v  \quad \dot{v} = u  
$$ 
So in control affine form:

$$
\dot{x} = f(x) + g(x)u  
$$

with

$$
f(x) =  
\begin{bmatrix}  
v \\  
0  
\end{bmatrix},  
\quad  
g(x) =  
\begin{bmatrix}  
0 \\  
I  
\end{bmatrix}  
$$

Barrier function $h$
$$
h(p) = \|p - p_{obs}\|^2 - r_{safe}^2  
$$
Gradient:
$$
\nabla_p h = 2(p - obs)  
$$
Time derivative:
$$
\dot{h} = 2(p - obs)^T v  
$$
## First order Lie derivatives
$$
L_f h = \nabla h^\top f = \dot{h} = 2(p - obs)^T v  
$$
## Second Order Lie derivatives

Differentiate again:
$$
\ddot{h} =  
2 v^T v  
+  
2 (p - obs)^T u  
$$
Now identify terms:

$$
\ddot{h} = L_f^2 h + L_g L_f h , u  
$$
$$
L_f^2 h = 2 v^T v  
$$
$$
L_g L_f h = 2 (p - obs)^T  
$$
## Actuation
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