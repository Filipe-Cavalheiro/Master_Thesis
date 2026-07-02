import numpy as np
import os
import matplotlib.pyplot as plt
from math import comb

# Clear console
os.system('cls' if os.name == 'nt' else 'clear')

# Close all existing matplotlib figures
plt.close('all')

def bernstein_approximation(f, n, x, a, b):
    """
    Evaluate the Bernstein polynomial approximation of degree n
    for function f at points x.
    """
    x = np.asarray(x)

    # Map x from [a,b] to t in [0,1]
    t = (x - a) / (b - a)

    B = np.zeros_like(x, dtype=float)

    for k in range(n + 1):
        basis = comb(n, k) * t**k * (1 - t)**(n - k)

        # Sample f at the corresponding point in [a,b]
        xk = a + (b - a) * k / n
        B += f(xk) * basis

    return B

f = np.sin
n = 10

a, b = 0, 10
x = np.linspace(a, b, 1000)

y_exact = f(x)
y_bernstein = bernstein_approximation(f, n, x, a, b)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y_exact, label='sin(x)', linewidth=2)
plt.plot(x, y_bernstein, '--', label=f'Bernstein approximation (n={n})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bernstein Approximation of sin(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
