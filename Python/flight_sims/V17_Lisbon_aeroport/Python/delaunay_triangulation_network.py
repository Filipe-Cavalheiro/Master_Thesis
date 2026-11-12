import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

def delaunay_triangulation_network(positions, verbose=False):
    """
    Creates a network using Delaunay triangulation.

    Parameters
    ----------
    positions : (N, D) ndarray
        Agent positions.
    verbose : bool, optional
        If True, plots the triangulation (only works for 2D).

    Returns
    -------
    connections : (N, N) ndarray of bool
        Adjacency matrix.
    laplacian : (N, N) ndarray
        Graph Laplacian.
    degree_matrix : (N, N) ndarray
        Degree matrix.
    """
    n_agents = len(positions)

    # remove singleton dimensions
    tol = 1e-9
    position_mask = (np.max(np.abs(positions - positions[0,:]), axis=0) >= tol)
    masked_positions = positions[:, position_mask]

    # Center
    X = masked_positions - masked_positions.mean(axis=0)

    # Normalize each coordinate
    scale = np.std(X, axis=0)
    scale[scale == 0] = 1
    X /= scale

    # Compute numerical rank
    r = np.linalg.matrix_rank(X, tol=1e-5)

    edges = set()
    if r <= 1:
        # collinear agents (e.g. drones in a row at the same height after
        # takeoff): Delaunay needs 2-D data, connect consecutive agents
        # along the line
        t = PCA(n_components=1).fit_transform(X)[:, 0] if r == 1 else np.zeros(n_agents)
        order = np.argsort(t, kind="stable")
        edges = {tuple(sorted((int(a), int(b)))) for a, b in zip(order[:-1], order[1:])}
    else:
        # PCA if necessary
        if r < positions.shape[1]:
            X = PCA(n_components=r).fit_transform(X)
        else:
            X = positions

        dt = Delaunay(X)

        # Extract unique edges
        for simplex in dt.simplices:
            m = len(simplex)
            for i in range(m):
                for j in range(i + 1, m):
                    a, b = sorted((simplex[i], simplex[j]))
                    edges.add((a, b))

    # Build adjacency matrix
    A = np.zeros((n_agents, n_agents), dtype=bool)

    for i, j in edges:
        A[i, j] = True
        A[j, i] = True

    # Degree matrix
    degrees = A.sum(axis=1)
    degree_matrix = np.diag(degrees)

    # Laplacian
    laplacian = degree_matrix - A.astype(int)

    if not np.any(A):
        input("No connections found. Press Enter to continue...")

    return A, laplacian, degree_matrix