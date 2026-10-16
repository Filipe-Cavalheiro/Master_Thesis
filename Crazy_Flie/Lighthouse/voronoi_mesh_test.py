import numpy as np
import heapq
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull, Delaunay

class Param:
    def __init__(self, traj=2):
        # ----------------------------
        # Simulation
        # ----------------------------
        self.Tend = 30
        self.dTi = 0.01
        self.Nsim = int(round(self.Tend / self.dTi)) + 1
        self.t = np.arange(0, self.Tend + self.dTi, self.dTi)

        self.gravity = 9.81
        self.nD = 3
        self.sphere_radius = 2.0

        # ----------------------------
        # Trajectory
        # ----------------------------
        self.ref_target_traj = self._make_traj(traj)
        self.initial_target_pos = self.ref_target_traj["pos"](0)
        

    # ----------------------------
    # Trajectory generator
    # ----------------------------
    def _make_traj(self, traj):
        if traj == 1:
            return {
                "pos": lambda t: np.array([0, 0, 5]),
                "vel": lambda t: np.array([0, 0, 0]),
            }

        if traj == 2:
            w = 0.25
            return {
                "pos": lambda t: np.array([
                    -5 + 15 * np.sin(w * t),
                     15 * np.cos(w * t),
                     5
                ]),
                "vel": lambda t: np.array([
                    15 * w * np.cos(w * t),
                   -15 * w * np.sin(w * t),
                    0
                ]),
            }

        if traj == 3:
            return {
                "pos": lambda t: np.array([
                    5 * np.sin(t),
                    5 * np.sin(2 * t),
                    5
                ]),
                "vel": lambda t: np.array([
                    5 * np.cos(t),
                    10 * np.cos(2 * t),
                    0
                ]),
            }

        if traj == 4:
            return {
                "pos": lambda t: np.array([
                    5 * np.sin(t),
                    5 * np.sin(2 * t),
                    5 + 0.2 * np.cos(2 * t),
                ]),
                "vel": lambda t: np.array([
                    5 * np.cos(t),
                    10 * np.cos(2 * t),
                    -0.4 * np.sin(2 * t),
                ]),
            }

        raise ValueError("Unknown trajectory")

class Voronoi:
    def __init__(self, param: Param):
        self.mesh_pos = np.array(None)
        self.surf_direction = [1, 0, 0]
        self.v_circle = np.array(None)
        self.v_cone = np.array(None)
        self.v_sphere = np.array(None)
        self.vertices = np.array(None)
        self.faces = np.array(None)
        self.sites = np.array(None)
        self.newSite = np.zeros((1, param.nD))

class MeshState:
    """
    Replacement for MATLAB 'persistent' variable.
    """
    ran_u2_eq_1 = False

class MeshVertex:
    def __init__(self, idx, pos):
        self.idx = idx
        self.pos = np.array(pos, dtype=float)

        self.dist = np.inf
        self.origin_site = -1
        self.prev = -1

        self.neigh = []
        self.neighW = []

import numpy as np

def pre_compute_desired_pos(voronoi: Voronoi, param: Param):
    """
    Equivalent of MATLAB function:
    [p_d, swarm_center, voronoi] = pre_compute_desired_pos(voronoi, Param)
    """
    voronoi.v_circle, voronoi.v_cone, voronoi.v_sphere = pre_compute_meshes(
        1001, param
    )

    voronoi.vertices, voronoi.faces, voronoi.mesh_pos = pos_based_mesh_transform(
        param.initial_target_pos,
        voronoi.mesh_pos,
        voronoi.surf_direction,
        voronoi.v_circle,
        voronoi.v_cone,
        voronoi.v_sphere,
        voronoi.faces,
        param
    )

    # get initial desired final positions (p_d)
    N = voronoi.vertices.shape[0]
    voronoi.sites = np.random.permutation(N)[:param.nD]
    p_d = voronoi.vertices[voronoi.sites, :]

    return p_d, voronoi.mesh_pos, voronoi


def pre_compute_meshes(P:int, param: Param):
    """
    Pre-compute circle-like, cone-like, and sphere-like point sets.

    Parameters
    ----------
    P : int
        Number of points
    param : class
        Must contain param.sphere_radius

    Returns
    -------
    v_circle : (P, 3) ndarray
    v_cone   : (P, 3) ndarray
    v_sphere : (P, 3) ndarray
    """

    R = param.sphere_radius

    # --- pre_compute circle ---
    N = np.floor((P - 1) / 2).astype(int)
    golden_angle = np.pi * (3 - np.sqrt(5))

    x0 = np.zeros(P)
    y0 = np.zeros(P)
    z0 = np.zeros(P)

    for k in range(P):
        r = R * np.sqrt((k + 1) / N)
        theta = (k + 1) * golden_angle

        y0[k] = r * np.cos(theta)
        z0[k] = r * np.sin(theta)

    v_circle = np.stack([x0, y0, z0], axis=1)

    # --- pre_compute cone ---
    radial = np.sqrt(v_circle[:, 1]**2 + v_circle[:, 2]**2)
    v_cone = np.column_stack([
        R * (1 - (radial / R)**2),
        v_circle[:, 1],
        v_circle[:, 2]
    ])

    # --- pre_compute sphere ---
    phi_golden = (1 + np.sqrt(5)) / 2

    i = np.arange(P)
    phi = 2 * np.pi * i / phi_golden
    theta = np.arccos(1 - 2 * (i + 0.5) / P)

    x_s = R * np.cos(theta)
    y_s = R * np.sin(theta) * np.sin(phi)
    z_s = R * np.sin(theta) * np.cos(phi)

    v_sphere = np.stack([x_s, y_s, z_s], axis=1)

    return v_circle, v_cone, v_sphere

def pos_based_mesh_transform(
    target_pos,
    mesh_pos,
    surf_direction,
    v_circle,
    v_cone,
    v_sphere,
    faces,
    param
):
    # Distance along surface normal
    dist_surface = np.dot(target_pos, surf_direction)

    sphere_r = param.sphere_radius

    # --- Compute u_target ---
    if dist_surface < -(sphere_r / 2):
        u_target = 0
    elif dist_surface > -(sphere_r / 2) and dist_surface < (sphere_r / 2):
        u_target = 0.5 + (dist_surface / sphere_r)
    elif dist_surface >= (sphere_r / 2):
        u_target = dist_surface / (sphere_r / 2)
    else:
        u_target = 2

    # --- Split stages ---
    if u_target <= 1:
        u1 = u_target
        u2 = 0
        mesh_pos = np.array([0.0, target_pos[1], target_pos[2]])
    else:
        u1 = 1
        u2 = min(u_target - 1, 1)
        mesh_pos = np.array([
            target_pos[0] - sphere_r / 2 + u2 * sphere_r / 2,
            target_pos[1],
            target_pos[2]
        ])

    # --- Circle → cone ---
    v_current = (1 - u1) * v_circle + u1 * v_cone

    # --- Sphere blend ---
    if u2 > 0:
        v_current = (1 - u2) * v_current + u2 * v_sphere

    vertices = v_current + mesh_pos.reshape(1, 3)

    # --- topology update ---
    if u2 == 1 and not MeshState.ran_u2_eq_1:
        hull = ConvexHull(vertices)
        faces = hull.simplices
        MeshState.ran_u2_eq_1 = True

    elif u2 != 1:
        MeshState.ran_u2_eq_1 = False

        if u2 >= 0.2:
            hull = ConvexHull(vertices)
            faces = hull.simplices
        else:
            # NOTE: MATLAB code used only v_circle(:,2:3)
            # Here we approximate same behavior
            tri = Delaunay(v_circle[:, 1:3])  # adjust if needed
            faces = tri.simplices

    return vertices, faces, mesh_pos

def runCVT(state, sites, gfx=None):
    """
    Updates site positions on a mesh using Dijkstra-based Voronoi regions.
    Only the first site index is updated.
    """
    vertices = state.vertices
    faces = state.faces

    V = buildVertexObjects(vertices, faces)
    V = dijkstraMeshHeap(V, sites)
    new_sites = updateSitesCVT(vertices, faces, V, sites)
    return new_sites


def extractVoronoiEdges(V):
    """
    Extract edges between mesh vertices whose Voronoi regions differ.
    """

    edges = []
    N = len(V)

    for i in range(N):
        neigh_i = V[i].neigh

        for j in neigh_i:
            if j <= i:
                continue

            if V[i].origin_site != V[j].origin_site:
                edges.append((i, j))

    return np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)

def updateSitesCVT(vertices, faces, V, sites):

    origin = np.array([v.origin_site for v in V])

    new_sites = sites.copy()

    for i, site in enumerate(sites):

        centroid = np.zeros(3)
        area_sum = 0.0

        for tri in faces:

            # region assignment of triangle vertices
            reg = origin[tri]

            # only use triangles fully inside this Voronoi cell
            if np.all(reg == site):

                p1, p2, p3 = vertices[tri]

                area = 0.5 * np.linalg.norm(
                    np.cross(p2 - p1, p3 - p1)
                )

                tri_centroid = (p1 + p2 + p3) / 3.0

                centroid += area * tri_centroid
                area_sum += area

        if area_sum > 0:

            centroid /= area_sum

            # snap centroid to nearest mesh vertex
            d = np.linalg.norm(vertices - centroid, axis=1)
            new_sites[i] = np.argmin(d)

    return new_sites

def buildVertexObjects(vertices, faces):
    """
    Build MeshVertex array and adjacency from triangle faces.
    """

    N = vertices.shape[0]
    V = [MeshVertex(i, vertices[i]) for i in range(N)]

    # Build unique undirected edges
    E = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])

    E = np.sort(E, axis=1)
    E = np.unique(E, axis=0)

    # Fill adjacency lists
    for a, b in E:
        w = np.linalg.norm(vertices[a] - vertices[b])

        V[a].neigh.append(b)
        V[a].neighW.append(w)

        V[b].neigh.append(a)
        V[b].neighW.append(w)

    return V

def dijkstraMeshHeap(V, sites):
    """
    Multi-source Dijkstra using lazy heap (no decrease-key).
    """

    heap = []

    # initialize sources
    for s in sites:
        V[s].dist = 0.0
        V[s].origin_site = s
        heapq.heappush(heap, (0.0, s))

    while heap:
        current_dist, u = heapq.heappop(heap)

        # skip stale entries
        if current_dist > V[u].dist:
            continue

        Vu = V[u]

        for v, w in zip(Vu.neigh, Vu.neighW):
            alt = current_dist + w

            if alt < V[v].dist:
                V[v].dist = alt
                V[v].prev = u
                V[v].origin_site = Vu.origin_site

                heapq.heappush(heap, (alt, v))

    return V

# ----------------------------
# Main
# ----------------------------
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    param = Param(traj=2)
    voronoi = Voronoi(param)

    p_d, swarm_center, voronoi = pre_compute_desired_pos(voronoi, param)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Voronoi mesh
    ax.plot_trisurf(
        voronoi.vertices[:, 0],
        voronoi.vertices[:, 1],
        voronoi.vertices[:, 2],
        triangles=voronoi.faces,
        color=(0.6, 0.6, 0.8, 0.6),
        edgecolor="k",
        linewidth=0.2,
    )

    # Target position
    target = np.asarray(param.initial_target_pos)
    ax.scatter(
        target[0], target[1], target[2],
        c="red",
        s=100,
        label="Target Position"
    )

    ax.scatter(
        p_d[:, 0],
        p_d[:, 1],
        p_d[:, 2],
        c="green",
        s=60,
        label="Desired Positions"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # animation loop
    for k in range(param.Nsim):
        print("Computing simulation interation: " + str(k))
        t = param.t[k]
        target_pos = param.ref_target_traj["pos"](t)

        # transform mesh
        voronoi.vertices, voronoi.faces, voronoi.mesh_pos = pos_based_mesh_transform(
            target_pos,
            voronoi.mesh_pos,
            voronoi.surf_direction,
            voronoi.v_circle,
            voronoi.v_cone,
            voronoi.v_sphere,
            voronoi.faces,
            param
        )

        if k == 1 or np.mod(k,5) == 0:
            voronoi.sites = runCVT(voronoi, voronoi.sites)

        p_d = voronoi.vertices[voronoi.sites, :]

        # ---- redraw ----
        ax.cla()
        ax.plot_trisurf(
            voronoi.vertices[:, 0],
            voronoi.vertices[:, 1],
            voronoi.vertices[:, 2],
            triangles=voronoi.faces,
            color=(0.6, 0.6, 0.8, 0.6)
        )

        ax.scatter(target_pos[0], target_pos[1], target_pos[2], c="red", s=100)

        ax.scatter(
            p_d[:, 0],
            p_d[:, 1],
            p_d[:, 2],
            c="green",
            s=60,
            label="Desired Positions"
        )

        plt.pause(0.01)

    plt.show()


if __name__ == "__main__":
    main()