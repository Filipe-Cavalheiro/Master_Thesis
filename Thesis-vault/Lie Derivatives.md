In [differential geometry](https://en.wikipedia.org/wiki/Differential_geometry "Differential geometry"), the **Lie derivative** ([/liː/](https://en.wikipedia.org/wiki/Help:IPA/English "Help:IPA/English") [_LEE_](https://en.wikipedia.org/wiki/Help:Pronunciation_respelling_key "Help:Pronunciation respelling key")), named after [Sophus Lie](https://en.wikipedia.org/wiki/Sophus_Lie "Sophus Lie") by [Władysław Ślebodziński](https://en.wikipedia.org/wiki/W%C5%82adys%C5%82aw_%C5%9Alebodzi%C5%84ski "Władysław Ślebodziński"),[[1]](https://en.wikipedia.org/wiki/Lie_derivative#cite_note-1)[[2]](https://en.wikipedia.org/wiki/Lie_derivative#cite_note-2) evaluates the change of a [tensor field](https://en.wikipedia.org/wiki/Tensor_field "Tensor field") (including scalar functions, [vector fields](https://en.wikipedia.org/wiki/Vector_field "Vector field") and [one-forms](https://en.wikipedia.org/wiki/One-form "One-form")), along the [flow](https://en.wikipedia.org/wiki/Flow_\(mathematics\) "Flow (mathematics)") defined by another vector field. This change is coordinate invariant and therefore the Lie derivative is defined on any [differentiable manifold](https://en.wikipedia.org/wiki/Differentiable_manifold "Differentiable manifold").
## Definition (Lie Derivative)

The **Lie derivative** of a function   $V : \mathbb{R}^n \to \mathbb{R}$ along a vector field  $f : \mathbb{R}^n \to \mathbb{R}^n$  is defined as:

$$
\begin{matrix}
\mathcal{L}_f V(x)
= \left( \frac{\partial V}{\partial x}(x) \right)^\top \cdot f(x) \Leftrightarrow \\
\mathcal{L}_f V(x) = \nabla V^\top \cdot f(x)
\end{matrix}
$$

---

## Interpretation

- When calculating the time derivative of \( V(x(t)) \):

$$
\frac{dV(x(t))}{dt}
= \sum_{i=1}^{n} \frac{\partial V(x(t))}{\partial x_i(t)} \dot{x}_i(t)
= \left( \frac{\partial V}{\partial x}(x) \right)^\top \dot{x}
= \mathcal{L}_f V(x)
$$

- Therefore, the **Lie derivative characterizes the time evolution** of the energy function $V$ along the trajectory defined by:
$$
\dot{x} = f(x)
$$
- Lyapunov stability results can be written using the **Lie derivative**, which is especially important for [[Control Barrier Function (CBF)]].