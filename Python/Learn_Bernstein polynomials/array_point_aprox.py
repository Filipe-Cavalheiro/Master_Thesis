import numpy as np
import os
import matplotlib.pyplot as plt
from math import comb

# Clear console
os.system('cls' if os.name == 'nt' else 'clear')

# Close all existing matplotlib figures
plt.close('all')

def bernstein_from_samples(y, t):
    """
    Bernstein polynomial defined by sample values y.
    y: array of n+1 coefficients/samples
    t: points in [0,1]
    """
    y = np.asarray(y)
    t = np.asarray(t)

    n = len(y) - 1

    B = np.zeros_like(t, dtype=float)

    for k in range(n + 1):
        basis = comb(n, k) * t**k * (1 - t)**(n - k)
        B += y[k] * basis

    return B

n = 20
y = np.array([1, 2.5, 1.2, 4.0, 5.2])
t = np.linspace(0, 1, 500)
curve = bernstein_from_samples(y, t)
sample_x = np.linspace(0, 1, len(y))

plt.plot(t, curve, label='Bernstein curve')
plt.plot(sample_x, y, 'o', label='Data points')
plt.legend()
plt.show()
