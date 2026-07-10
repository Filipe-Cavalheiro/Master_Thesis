import numpy as np
from math import comb

def bernstein_basis_matrix(t, n_degree):
    t = np.asarray(t)

    M = np.zeros((len(t), n_degree + 1))

    for k in range(n_degree + 1):
        M[:, k] = comb(n_degree, k) * t**k * (1 - t)**(n_degree - k)

    return M


def fit_bernstein_curve(X, n_degree: int):
    """
    Fit a Bernstein polynomial curve to 3D points.

    Parameters
    ----------
    X : (N,3) array
        Control/sample points.
    n_degree : int
        Degree of Bernstein polynomial.
    resolution : int
        Number of evaluation points.

    Returns
    -------
    coeffs : tuple
        Bernstein coefficients (c_x, c_y, c_z).
    dcoeffs :  
        Bernstein coefficients first derivative (c_x, c_y, c_z).
    ddcoeffs :
        Bernstein coefficients second derivative (c_x, c_y, c_z).
    """

    n_points = len(X)
    t = np.linspace(0, 1, n_points)

    M = bernstein_basis_matrix(t, n_degree)

    # Fit
    c_x, *_ = np.linalg.lstsq(M, X[:,0], rcond=None)
    c_y, *_ = np.linalg.lstsq(M, X[:,1], rcond=None)
    c_z, *_ = np.linalg.lstsq(M, X[:,2], rcond=None)

    coeffs = np.vstack((c_x, c_y, c_z))

    # First-derivative Bernstein coefficients
    dcoeffs = np.array(n_degree * np.diff(coeffs, axis=1))

    # Second-derivative Bernstein coefficients
    ddcoeffs = np.array(n_degree * (n_degree-1) * np.diff(coeffs, n=2, axis=1))

    return coeffs, dcoeffs, ddcoeffs


