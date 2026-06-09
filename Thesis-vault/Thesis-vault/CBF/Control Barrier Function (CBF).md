Based from @amesControlBarrierFunctions2019a and @amesControlBarrierFunction2017 and @reisControlBarrierFunction2021 
## Barrier to keep agent inside a set

This equations are made so that any actuation is made in a way that does not move the agent outside of the set/barrier function.

---

$$\begin{aligned}  
u(x) &= \arg\min_{u \in \mathbb{R}^m} \frac{1}{2} | u - k(x) |^2 \\  
\text{s.t. } & L_f h(x) + L_g h(x) u \ge -\alpha(h(x))  
\end{aligned}  $$

Where:

- k(x) is a **nominal control input** (e.g., from a controller ignoring safety constraints).
- h(x) is the **barrier function** defining the safe set $\mathcal{C} = { x : h(x) \ge 0 }$.
- $L_f h(x)$ and ($L_g h(x)$) are [[Lie Derivatives]] along the system dynamics.
- $\alpha(\cdot)$ is an **extended class-$\mathcal{K}$ function** (commonly linear: (\alpha(s) = \gamma s)) ensuring forward invariance.

$$
\begin{matrix}
 L_f h(x) + L_g h(x) u \ge -\alpha(h(x))  \Leftrightarrow \\
 - L_g h(x)^\top u \le  L_f h(x) + \alpha(h(x))
\end{matrix}
$$
When programming CBF's it is common to define $L_f h(x) + L_g h(x) \rightarrow A$ and $\alpha k \rightarrow b$ and since the system is subsect to $L_f h(x) + L_g h(x) u \ge -\alpha(h(x))$ this means that $Au \ge -b$ , making so that if the rule is kept true there is no need to change the actuation u since this will naturally keep it self inside the set. In case the equation is false it is need the change the actuation in a way to make the equation true this is done via a half plane projection (No I do not know this I just know it works...)
$$
u_{cbf} = u - \frac{A*u - b}{A*A'} * A^\top;
$$
## Barrier to keep agent outside a set

In some problems, the objective is not to keep the system inside a safe set, but instead to ensure it **never enters a forbidden region** (e.g., obstacle avoidance).

---

### Defining the Unsafe Set

Let the **unsafe set** be

$$  
\mathcal{D} = { x : h(x) \ge 0 }  
$$

To keep the agent _outside_ this set, we define the **safe region** as:

$$
\mathcal{C} = { x : h(x) \le 0 }  
$$

So now the safety condition becomes:

$$ 
h(x) \le 0  
$$

---

### CBF Condition for Staying Outside

To ensure forward invariance of the set $\mathcal{C} = { x : h(x) \le 0 }$, we enforce:
$$ 
L_f h(x) + L_g h(x) u \le -\alpha(h(x))  
$$
Compare this carefully with the "stay-inside" condition:    
$$  
L_f h(x) + L_g h(x) u \ge -\alpha(h(x))  
$$
The inequality flips because the sign of the safe region flips.

---

### Optimization Formulation

The quadratic program becomes:

$$ 
\begin{aligned}  
u(x) &= \arg\min_{u \in \mathbb{R}^m} \frac{1}{2} | u - k(x) |^2 \\  
\text{s.t. } & L_f h(x) + L_g h(x) u \le -\alpha(h(x))  
\end{aligned}  
$$

Rewriting in linear constraint form:

$$  
L_g h(x) u \le -L_f h(x) - \alpha(h(x))  
$$

Define:

$$ 
A = L_g h(x), \quad b = -L_f h(x) - \alpha(h(x))  
$$

Then the constraint becomes:

$$  
Au \le b  
$$

---

### Geometric Interpretation

This defines a **half-space constraint** in control space.

- If $Au \le b$  is already satisfied →  
    The nominal control $k(x)$ is safe and no modification is needed.
    
- If violated →  
    The control is **projected onto the admissible half-space**.
    
This is again a **half-plane (half-space) projection**, but now with the opposite inequality sign.

---

### Closed-Form Projection

If the nominal control violates the constraint $Au \le b$ aka being $Au > b$. the projected control is:

$$ 
u_{cbf} =  
u - \frac{A u - b}{A A^\top} A^\top  
$$
Otherwise:

$$
u_{cbf} = u  
$$
---

### Intuition

- Inside-set CBF → prevents $h(x)$ from decreasing below zero.
    
- Outside-set CBF → prevents $h(x)$ from increasing above zero.
    

In obstacle avoidance, a common choice is:

$$ 
h(x) = d(x)^2 - r^2  
$$

where:

- $d(x)$ is the distance to the obstacle center
    
- $r$ is the obstacle radius
    

Then:

- $h(x) \le 0$ → inside obstacle (unsafe)
    
- $h(x) > 0$ → outside obstacle (safe)
    

The CBF ensures the system never crosses into ( h(x) \le 0 ).

---

### Key Insight

Mathematically, "stay inside" and "stay outside" are **the same mechanism with opposite inequality direction**.

You are simply choosing which side of the zero level-set of $h(x)$ is invariant.