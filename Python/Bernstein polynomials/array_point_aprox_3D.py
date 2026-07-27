import numpy as np
import matplotlib.pyplot as plt
from math import comb

# ----------------------------------
# Figure-8 curve in 3D
# ----------------------------------

resolution = 1000
n_points = 50
n_degree = 10

t = np.linspace(0, 2*np.pi, resolution)

# 3D Figure-8
x = 2*np.sin(t)
y = 2*np.sin(t) * np.cos(t)
z = 2*np.abs(np.sin(t)*0.5)

curve = np.column_stack((x, y, z))

# Sample control points
indices = np.linspace(0, resolution - 1, n_points, dtype=int)
X = curve[indices]

def bernstein_basis_matrix(t, n_degree):
    t = np.asarray(t)

    M = np.zeros((len(t), n_degree + 1))

    for k in range(n_degree + 1):
        M[:, k] = comb(n_degree, k) * t**k * (1 - t)**(n_degree - k)

    return M

# Parameter values for control points
t_ctrl = np.linspace(0, 1, n_points)
# Fit Bernstein coefficients
M_ctrl = bernstein_basis_matrix(t_ctrl, n_degree)

# least square approximation for the n_degree polynomial  
c_x, *_ = np.linalg.lstsq(M_ctrl, X[:, 0], rcond=None)
c_y, *_ = np.linalg.lstsq(M_ctrl, X[:, 1], rcond=None)
c_z, *_ = np.linalg.lstsq(M_ctrl, X[:, 2], rcond=None)

# Evaluate fitted curve
t_eval = np.linspace(0, 1, resolution)
M_eval = bernstein_basis_matrix(t_eval, n_degree)

# transform from bernstein to cartesian
curve_x = M_eval @ c_x
curve_y = M_eval @ c_y
curve_z = M_eval @ c_z

# coefficients of first derivative
dc_x = n_degree * np.diff(c_x)
dc_y = n_degree * np.diff(c_y)
dc_z = n_degree * np.diff(c_z)

# Bernstein basis of degree n-1
M1 = bernstein_basis_matrix(t_eval, n_degree - 1)

dx = M1 @ dc_x
dy = M1 @ dc_y
dz = M1 @ dc_z

# coefficients of second derivative
ddc_x = n_degree * (n_degree - 1) * np.diff(c_x, n=2)
ddc_y = n_degree * (n_degree - 1) * np.diff(c_y, n=2)
ddc_z = n_degree * (n_degree - 1) * np.diff(c_z, n=2)

# Bernstein basis of degree n-2
M2 = bernstein_basis_matrix(t_eval, n_degree - 2)

ddx = M2 @ ddc_x
ddy = M2 @ ddc_y
ddz = M2 @ ddc_z

# ----------------------------------
# Plot
# ----------------------------------

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, label="Original curve")
ax.plot(curve_x, curve_y, curve_z,
        label="Bernstein fit", linewidth=2)

ax.scatter(X[:,0], X[:,1], X[:,2],
           color='red', s=40)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

step = 50

ax.quiver(curve_x[::step],
          curve_y[::step],
          curve_z[::step],
          dx[::step],
          dy[::step],
          dz[::step],
          length=0.1,
          normalize=True,
          color='red')

ax.quiver(curve_x[::step],
          curve_y[::step],
          curve_z[::step],
          ddx[::step],
          ddy[::step],
          ddz[::step],
          length=0.1,
          normalize=True,
          color='green')

plt.show()