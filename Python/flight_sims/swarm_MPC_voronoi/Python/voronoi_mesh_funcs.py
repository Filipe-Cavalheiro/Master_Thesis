import numpy as np
import heapq
from scipy.spatial import ConvexHull, Delaunay
from Structs_sim import Param, Voronoi, MeshVertex
from Structs_agent import DroneState
from get_fibonacci_points import get_fibonacci_points
from helper_funcs import spherical_gaussian

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
    # Height R at the centre down to 0 at the disc rim, so the cone lip lies
    # on the wall and no vertex goes behind it (the disc rim is past R)
    radial = np.sqrt(v_circle[:, 1]**2 + v_circle[:, 2]**2)
    v_cone = np.column_stack([
        R * (1 - (radial / radial.max())**2),
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

    if getattr(param.sim, "sphere_from_circle", False):
        # Sphere vertex k from circle vertex k (Lambert azimuthal equal-area
        # map: the disk centre goes to the front pole, its rim to the back
        # pole, same angle around the axis). The circle -> cone -> sphere
        # blend then keeps every site at its angle around the target instead
        # of moving it to an unrelated Fibonacci sphere point.
        r = np.sqrt(v_circle[:, 1]**2 + v_circle[:, 2]**2)
        alpha = 2 * np.arcsin(np.clip(r / r.max(), 0.0, 1.0))
        safe = np.where(r > 0, r, 1.0)
        v_sphere = np.column_stack([
            R * np.cos(alpha),
            R * np.sin(alpha) * v_circle[:, 1] / safe,
            R * np.sin(alpha) * v_circle[:, 2] / safe,
        ])

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
            param.sim.mesh_points, param
        )
        
        voronoi.vertices, voronoi.faces, voronoi.mesh_pos = pos_based_mesh_transform(
            param.traj.initial_target_pos,
            voronoi,
            param
        )

        # get initial desired final positions (p_d)
        N = voronoi.vertices.shape[0]
        voronoi.sites = random_safe_sites(voronoi.vertices, param.sim.nD, param)
        if getattr(param.sim, "converge_initial_formation", False):
            # CVT on the initial mesh before the start, as the flight code
            # (Formation.converge), so the drones start on spread slots
            # (random sites can be a few cm apart)
            voronoi.weights = np.ones((N, 1))
            for _ in range(30):
                sites = voronoi.sites
                voronoi.newSites = sites.copy()      # occupied vertices, updated drone by drone (as main)
                for i in range(len(sites)):
                    voronoi.newSites[i] = runCVT(voronoi, [sites[i], *np.delete(sites, i)], None, param)
                if np.array_equal(voronoi.newSites, sites):
                    break
                voronoi.sites = voronoi.newSites.copy()
        voronoi.newSites = voronoi.sites.copy()
        p_d = (voronoi.vertices[voronoi.sites, :]).T - param.traj.initial_target_pos

        if param.sim.measurement_dimension == 1:
            voronoi.weights = np.ones((voronoi.vertices.shape[0],1))
    else:
        raise ValueError('param.sim.distribution was not correctly defined.')

    return p_d, voronoi.mesh_pos, voronoi

def outside_envelope(delta, param: Param):
    """
    True for the offsets delta (m, 3) outside the inter-agent safety envelope
    (sphere of radius obstacle_avoidance_dim), as in the MPC collision constraint.
    """
    return np.linalg.norm(delta, axis=1) >= param.sim.obstacle_avoidance_dim


def random_safe_sites(vertices, n, param: Param):
    """
    n random mesh vertices, pairwise outside the safety envelope: the
    vertices are visited in random order and each is kept if it is outside
    the envelope of those already kept (random sequential adsorption), so
    the drones start at random on the surface but already separated, as the
    collision avoidance assumes.
    """
    sites = []
    for v in np.random.permutation(vertices.shape[0]):
        if not sites or outside_envelope(vertices[sites] - vertices[v], param).all():
            sites.append(v)
            if len(sites) == n:
                return np.array(sites)
    raise ValueError(f"Cannot place {n} drones on the initial mesh outside the safety envelope")


def mesh_blend_parameter(dist_surface, sphere_r):
    """
    Blend parameter u of the formation mesh from the distance of the target
    past the wall (dist_surface, after the red-border correction):
    u = 0 disc on the wall, 0 < u <= 1 disc -> cone, 1 < u < 2 cone -> sphere,
    u >= 2 sphere.
    """
    if dist_surface < -(sphere_r / 2):
        return 0
    elif dist_surface >= -(sphere_r / 2) and dist_surface < (sphere_r / 2):
        return 0.5 + (dist_surface / sphere_r)
    elif dist_surface >= (sphere_r / 2):
        return dist_surface / (sphere_r / 2)
    return 2


def formation_type(target_pos, airspace, sphere_r):
    """
    Formation of the mesh around target_pos, as pos_based_mesh_transform builds
    it: "wall" (flat disc on the wall), "cone" as soon as it is not a flat disc,
    "sphere" as soon as it is not a pure cone, and "red border" (disc on the
    red border) as soon as the target is within sphere_r / 2 of the red border.
    """
    _, dist_surface, past_red, _ = airspace.wall_frame(np.asarray(target_pos, dtype=float).reshape(3, 1))
    if past_red > 0:
        dist_surface = dist_surface - past_red
    if past_red + sphere_r / 2 > 0:
        return "red border"
    u = mesh_blend_parameter(dist_surface, sphere_r)
    if u == 0:
        return "wall"
    return "cone" if u <= 1 else "sphere"


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
    airspace = voronoi.airspace
    v_circle = voronoi.v_circle
    v_cone = voronoi.v_cone
    v_sphere = voronoi.v_sphere
    faces = voronoi.faces

    # Wall frame from the red zone: the wall is the surface `wall_offset`
    # metres outside the red border, normal to the closest red border point.
    # dist_surface is how far the target is past the wall (towards red).
    n_in, dist_surface, past_red, boundary = airspace.wall_frame(target_pos)
    up = np.array([0.0, 0.0, 1.0])
    # local mesh frame -> world: local x along n_in, y lateral, z up
    R = np.column_stack([n_in, np.cross(up, n_in), up])

    # The formation may not follow the target into the red zone: past the red
    # border the mesh is built around the closest border point instead, so the
    # encirclement extends only up to the red edge.
    target_eff = target_pos.reshape(3).astype(float)
    if past_red > 0:
        target_eff = target_eff - past_red * n_in
        dist_surface = dist_surface - past_red

    sphere_r = param.sim.sphere_radius

    u_target = mesh_blend_parameter(dist_surface, sphere_r)

    # --- Split stages ---
    if u_target <= 1:
        u1 = u_target
        u2 = 0
        # on the wall, in front of the target
        mesh_pos = (target_eff - dist_surface * n_in).reshape(-1, 1)
    else:
        u1 = 1
        u2 = min(u_target - 1, 1)
        mesh_pos = (target_eff - (1 - u2) * (sphere_r / 2) * n_in).reshape(-1, 1)

    # --- Circle → cone ---
    v_current = (1 - u1) * v_circle + u1 * v_cone

    # --- Sphere blend ---
    if u2 > 0:
        v_current = (1 - u2) * v_current + u2 * v_sphere

    vertices = v_current @ R.T + mesh_pos.reshape(1, 3)

    # --- Red border: sphere -> circle on the red/orange border surface ---
    # Mirrors the wall transition: the blend starts r/2 before the red border
    # and the formation is a flat circle on the border r/2 inside it.
    u3 = np.clip((past_red + sphere_r / 2) / sphere_r, 0.0, 1.0)
    if u3 > 0:
        border_pos = (boundary - airspace.red_margin * n_in).reshape(-1, 1)
        circle_on_border = v_circle @ R.T + border_pos.reshape(1, 3)
        sphere_vertices = vertices
        vertices = (1 - u3) * vertices + u3 * circle_on_border
        mesh_pos = (1 - u3) * mesh_pos + u3 * border_pos

    # --- topology update (on the unconstrained shape) ---
    if u3 >= 0.8:
        # (nearly) flat circle on the red border
        voronoi.ran_u2_eq_1 = False
        faces = Delaunay(v_circle[:, 1:3]).simplices

    elif u3 > 0:
        # sphere -> circle blend: keep the sphere topology
        voronoi.ran_u2_eq_1 = False
        faces = ConvexHull(sphere_vertices).simplices

    elif u2 == 1 and not voronoi.ran_u2_eq_1:
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

    # --- airspace: no vertex inside the red zone, orange altitude ceiling ---
    vertices = airspace.project(vertices)

    # --- ground / obstacles: no vertex closer than env_clearance above the terrain ---
    # The setpoint safety filter (safe_setpoint) keeps every drone env_clearance away from
    # the ground, so a goal below that height is unreachable: a flat disc low on the wall
    # would leave its bottom agents parked at the clearance height, short of their goals.
    floor = param.environment.height(vertices[:, :2]) + param.sim.env_clearance
    vertices[:, 2] = np.maximum(vertices[:, 2], floor)

    return vertices, faces, mesh_pos

def buildVertexObjects(vertices, weights, faces):
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
        w = np.linalg.norm(vertices[a] - vertices[b]) * ((weights[a,0] + weights[b,0]) / 2)

        V[a].neigh.append(b)
        V[a].neighW.append(w)

        V[b].neigh.append(a)
        V[b].neighW.append(w)

    return V

def dijkstraMeshHeap(V: list[MeshVertex], sites: list[int]):
    """
    Multi-source Dijkstra using lazy heap (no decrease-key).
    """
    # heap sort based on dist then  index if same dist
    heap: list[tuple[float, int]] = []

    # initialize sources
    for idx in sites:
        V[idx].dist = 0.0
        V[idx].origin_site = idx
        #V[idx].origin_site_weight = s.weight
        heapq.heappush(heap, (0.0, idx))

    while heap:
        current_dist, idx = heapq.heappop(heap)

        # skip stale entries
        if current_dist > V[idx].dist:
            continue

        Vu: MeshVertex
        Vu = V[idx]

        for v, w in zip(Vu.neigh, Vu.neighW):
            alternative_W = current_dist + w

            if alternative_W < V[v].dist:
                V[v].dist = alternative_W
                V[v].origin_site = Vu.origin_site
                heapq.heappush(heap, (alternative_W, v))
    return V

def nearest_free_vertex(vertices, centroid, occupied):
    d = np.linalg.norm(vertices - centroid, axis=1)
    order = np.argsort(d)

    for idx in order:
        if idx not in occupied:
            return idx

    raise RuntimeError("No free mesh vertices available.")

def updateSitesCVT(vertices, faces, V, neighbour_sites, voronoi: Voronoi):
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
    new_sites : ndarray
        Updated site indices.
    """
    origin = np.array([v.origin_site for v in V])

    site = neighbour_sites[0]
    new_site = site # fallback

    centroid = np.zeros(3)
    area_sum = 0.0

    for tri in faces:
        # Region assignment of triangle vertices
        reg = origin[tri]

        # Only use triangles fully inside this Voronoi cell
        if np.all(reg == site):
            p1, p2, p3 = vertices[tri]

            area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            tri_centroid = (p1 + p2 + p3) / 3.0

            centroid += area * tri_centroid
            area_sum += area

    occupied = set()

    for index in voronoi.newSites:
        occupied.add(index)

    occupied.remove(site)
    
    if area_sum > 0:
        centroid /= area_sum
        # d = np.linalg.norm(vertices - centroid, axis=1)
        # new_site = np.argmin(d)
        new_site = nearest_free_vertex(vertices, centroid, occupied)

    return new_site

def runCVT(voronoi: Voronoi, sites: list[int], agent: DroneState, param: Param):
    """
    Updates site positions on a mesh using Dijkstra-based Voronoi regions.
    """
    vertices = voronoi.vertices
    faces = voronoi.faces

    if param.sim.measurement_dimension == 2:
        if voronoi.ran_u2_eq_1 == True:
            tmp = vertices - voronoi.mesh_pos.T
            norms = np.linalg.norm(tmp, axis=1, keepdims=True)
            unit_sphere_vertices = tmp / norms

            # Negate the velocity direction because the spherical Gaussian assigns
            # larger weights to points that are closer to the reference direction.
            # In our formulation, larger weights correspond to vertices that are
            # farther from the desired motion direction (i.e., more dispersed).
            # By flipping the reference direction, the weighting is inverted so that
            # vertices closer to the target's velocity direction receive higher
            # importance.
            unit_sphere_vel = (
                -agent.estimated.target_velocity
                / np.linalg.norm(agent.estimated.target_velocity)
            )
            #Layman's terms: closer goal points in the vel direction

            voronoi.weights = spherical_gaussian(unit_sphere_vel, unit_sphere_vertices, param.sim.weights_sigma)
        else:
            voronoi.weights = np.ones((voronoi.vertices.shape[0],1))

    V = buildVertexObjects(vertices, voronoi.weights, faces)
    V = dijkstraMeshHeap(V, sites)
    new_site = updateSitesCVT(vertices, faces, V, sites, voronoi)
    return new_site