import numpy as np
import heapq
from scipy.spatial import ConvexHull, Delaunay
from Structs import Param, Voronoi, MeshVertex, Site
#from get_fibonacci_points import get_fibonacci_points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    R = param.sim.sphere_radius

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

def pre_compute_desired_pos(voronoi: Voronoi, param: Param):
    """
    Pre-compute point sets, mesh position and update voronoi params.

    Parameters
    ----------
    voronoi: class
        mesh information
    param: class
        simulation parameters

    Returns
    -------
    p_d : (Param.nD, 3) ndarray
    voronoi.mesh_pos : (3, 1) ndarray
    voronoi : class
    """
    if param.sim.distribution == "fibonacci":
        p_d = get_fibonacci_points([0,0,0], param)
        voronoi.mesh_pos = param.traj.initial_target_pos

    elif param.sim.distribution == "dijkstra":
        voronoi.v_circle, voronoi.v_cone, voronoi.v_sphere = pre_compute_meshes(
            1001, param
        )
        
        voronoi.vertices, voronoi.faces, voronoi.mesh_pos = pos_based_mesh_transform(
            param.traj.initial_target_pos,
            voronoi,
            param
        )

        # get initial random desired goal positions (p_d)
        N = voronoi.vertices.shape[0]
        voronoi.sites = [
            Site(index=i, weight=1.0)
            for i in np.random.permutation(N)[:param.sim.nD]
        ]
        voronoi.newSite = [
            Site(site.index, site.weight)
            for site in voronoi.sites
        ]
        indices = np.array([site.index for site in voronoi.sites], dtype=int)
        p_d = voronoi.vertices[indices, :].T - voronoi.mesh_pos
    else:
        raise ValueError('param.sim.distribution was not correctly defined.')

    return p_d, voronoi.mesh_pos, voronoi

def pos_based_mesh_transform(
    target_pos: np.ndarray, #3x1
    voronoi: Voronoi,
    param: Param
):
    """
    Update position of the mesh itself as well as its shape

    Parameters
    ----------
    target_pos: np.ndarray
        [3x1] position of the target
    voronoi: class
        mesh information
    param: class
        simulation parameters

    Returns
    -------
    vertices: mesh vertices
    faces: connections between vertices
    mesh_pos: position of the mesh in world frame
    """
    mesh_pos = voronoi.mesh_pos
    surf_direction = voronoi.surf_direction
    v_circle = voronoi.v_circle
    v_cone = voronoi.v_cone
    v_sphere = voronoi.v_sphere
    faces = voronoi.faces

    # Distance along surface normal
    dist_surface =  (target_pos.T @ surf_direction).item()

    sphere_r = param.sim.sphere_radius

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
        mesh_pos = np.array([0.0, target_pos[1,0], target_pos[2,0]]).reshape(-1,1)
    else:
        u1 = 1
        u2 = min(u_target - 1, 1)
        mesh_pos = np.array([
            target_pos[0,0] - sphere_r / 2 + u2 * sphere_r / 2,
            target_pos[1,0],
            target_pos[2,0]
        ]).reshape(-1,1)

    # --- Circle → cone ---
    v_current = (1 - u1) * v_circle + u1 * v_cone

    # --- Sphere blend ---
    if u2 > 0:
        v_current = (1 - u2) * v_current + u2 * v_sphere

    vertices = np.array(v_current + mesh_pos.reshape(1, 3))

    # --- topology update ---
    if u2 == 1 and not voronoi.ran_u2_eq_1:
        hull = ConvexHull(vertices)
        faces = hull.simplices
        voronoi.ran_u2_eq_1 = True

    elif u2 != 1:
        voronoi.ran_u2_eq_1 = False

        if u2 >= 0.2:
            hull = ConvexHull(vertices)
            faces = hull.simplices
        else:
            tri = Delaunay(v_circle[:, 1:3])
            faces = tri.simplices

    return vertices, faces, mesh_pos

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

def dijkstraMeshHeap(V: list[MeshVertex], sites: list[Site]):
    """
    Multi-source Dijkstra using lazy heap (no decrease-key).
    """
    heap: list[tuple[float, int, Site]] = []

    # initialize sources
    for s in sites:
        idx = s.index
        V[idx].dist = 0.0
        V[idx].origin_site = idx
        V[idx].origin_site_weight = s.weight
        heapq.heappush(heap, (0.0, s.index, s))

    while heap:
        current_dist, idx, u = heapq.heappop(heap)

        # skip stale entries
        if current_dist > V[idx].dist:
            continue

        Vu: MeshVertex
        Vu = V[idx]

        for v, w in zip(Vu.neigh, Vu.neighW):
            alternative_W = current_dist + w*Vu.origin_site_weight

            if alternative_W < V[v].dist:
                V[v].dist = alternative_W
                V[v].origin_site = Vu.origin_site
                V[v].origin_site_weight = Vu.origin_site_weight
                heapq.heappush(heap, (alternative_W, v, v))

    return V

import numpy as np

def updateSitesCVT(vertices, faces, V, sites):
    """
    Update site locations by moving each site to the mesh vertex closest
    to the area-weighted centroid of its Voronoi cell.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        Mesh vertices.
    faces : (M, 3) ndarray
        Triangle indices.
    V : list
        Vertex objects with an `origin_site` attribute.
    sites : ndarray
        Current site vertex indices.
    voronoi: Voronoi class
        class with global information about the mesh
    index : int or None, optional
        If None, update all sites.
        Otherwise, update only the specified site index.

    Returns
    -------
    new_site : Site class
        Updated site indices.
    """
    origin = np.array([v.origin_site for v in V])

    agent_site:Site = sites[0]
    new_site = Site(agent_site.index, agent_site.weight) # fallback

    centroid = np.zeros(3)
    area_sum = 0.0

    for tri in faces:
        # Region assignment of triangle vertices
        reg = origin[tri]

        # Only use triangles fully inside this Voronoi cell
        if np.all(reg == agent_site.index):
            p1, p2, p3 = vertices[tri]

            area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            tri_centroid = (p1 + p2 + p3) / 3.0

            centroid += area * tri_centroid
            area_sum += area

    if area_sum > 0:
        centroid /= area_sum

        # Snap centroid to nearest mesh vertex
        d = np.linalg.norm(vertices - centroid, axis=1)
        new_site = Site(np.argmin(d), agent_site.weight)

    return new_site

def extract_voronoi_edges(V):
    edges = []

    N = len(V)

    for i in range(N):
        ni = V[i].neigh

        for j in ni:
            # avoid duplicate edges
            if j <= i:
                continue

            if V[i].origin_site != V[j].origin_site:
                edges.append((i, j))

    return edges

def visualize_cvt(vertices, faces, V: list[MeshVertex], sites: list[Site], fig, ax):
    """
    Show:
      - mesh colored by Dijkstra distance
      - green site markers
      - black Voronoi edges
    """
    ax.cla()

    # --------------------------------------------------
    # distance field
    # --------------------------------------------------

    dist = np.array([v.dist for v in V])
    face_dist = dist[faces].mean(axis=1)

    mesh = Poly3DCollection(
        vertices[faces],
        linewidth=0.0,
        alpha = 0.15,
        edgecolor='none'
    )

    mesh.set_array(face_dist)
    mesh.set_cmap("viridis")
    mesh.set_clim(face_dist.min(), face_dist.max())

    ax.add_collection3d(mesh)

    # --------------------------------------------------
    # sites
    # --------------------------------------------------
    indices = np.array([site.index for site in sites], dtype=int)
    pts = vertices[indices]

    ax.scatter(
        pts[:,0],
        pts[:,1],
        pts[:,2],
        color='lime',
        s=80,
        marker='x',
        linewidth=2
    )

    # --------------------------------------------------
    # Voronoi boundaries
    # --------------------------------------------------
    drawn = set()
    edges = extract_voronoi_edges(V)
    for i, j in edges:
        ax.plot(
            [vertices[i, 0], vertices[j, 0]],
            [vertices[i, 1], vertices[j, 1]],
            [vertices[i, 2], vertices[j, 2]],
            color='k',
            lw=1,
        )

    # --------------------------------------------------

    ax.set_box_aspect([1,1,1])

    ax.set_xlim(vertices[:,0].min(), vertices[:,0].max())
    ax.set_ylim(vertices[:,1].min(), vertices[:,1].max())
    ax.set_zlim(vertices[:,2].min(), vertices[:,2].max())

    ax.set_axis_off()

    plt.draw()
    plt.pause(0.001)

    return fig, ax

fig = None
ax = None

def runCVT(state, sites):
    """
    Updates site positions on a mesh using Dijkstra-based Voronoi regions.
    """
    
    vertices = state.vertices
    faces = state.faces

    V = buildVertexObjects(vertices, faces)
    V = dijkstraMeshHeap(V, sites)

    if True:
        global fig, ax
        if fig is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')

        fig, ax = visualize_cvt(
            vertices,
            faces,
            V,
            sites,
            fig,
            ax
        )

    new_site = updateSitesCVT(vertices, faces, V, sites)
    return new_site