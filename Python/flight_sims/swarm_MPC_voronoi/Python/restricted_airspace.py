import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# World frame shared with the Gazebo world (Gazebo/scripts/generate_world.py):
# ENU, origin at the Lisbon airport reference point (LPPT ARP), sea level.
LAT0, LON0 = 38.7813, -9.1359
M_PER_DEG_LAT = 110950.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(LAT0))

# [m] minimum length scale of the joint coordinate s, so the wall frame does
# not spin when the target passes right next to a joint vertex
JOINT_MIN_SCALE = 5.0


def lonlat_to_enu(lonlat):
    lonlat = np.asarray(lonlat, dtype=float)
    return np.column_stack([
        (lonlat[:, 0] - LON0) * M_PER_DEG_LON,
        (lonlat[:, 1] - LAT0) * M_PER_DEG_LAT,
    ])


def read_kmz(path):
    """
    Read every Placemark polygon of a KMZ file.

    Returns
    -------
    zones : list of (name, [polygon (N, 2) lon/lat arrays])
        A placemark with a MultiGeometry yields several polygons.
    """
    root = ET.fromstring(zipfile.ZipFile(path).read("doc.kml"))
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    zones = []
    for pm in root.iter(f"{ns}Placemark"):
        name = (pm.findtext(f"{ns}name") or "").strip()
        polys = []
        for c in pm.iter(f"{ns}outerBoundaryIs"):
            coords = c.find(f".//{ns}coordinates").text.split()
            ll = np.array([[float(v) for v in p.split(",")[:2]] for p in coords])
            if np.allclose(ll[0], ll[-1]):
                ll = ll[:-1]
            polys.append(ll)
        if polys:
            zones.append((name, polys))
    return zones


class Polygon2D:
    """Simple polygon in the horizontal plane (ENU metres)."""

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)
        self.a = self.vertices
        self.b = np.roll(self.vertices, -1, axis=0)
        # unit edge directions and inward normals (left of the edge if CCW)
        e = self.b - self.a
        self.edge_dir = e / np.linalg.norm(e, axis=1, keepdims=True)
        area = 0.5 * np.sum(self.a[:, 0] * self.b[:, 1] - self.b[:, 0] * self.a[:, 1])
        left = np.column_stack([-self.edge_dir[:, 1], self.edge_dir[:, 0]])
        self.inward_normals = left if area > 0 else -left

    def contains(self, P):
        """Even-odd point in polygon test. P: (n, 2) -> (n,) bool."""
        x, y = P[:, 0:1], P[:, 1:2]
        ax, ay = self.a[:, 0], self.a[:, 1]
        bx, by = self.b[:, 0], self.b[:, 1]
        crosses = (ay > y) != (by > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = ax + (y - ay) * (bx - ax) / (by - ay)
        return np.sum(crosses & (x < x_int), axis=1) % 2 == 1

    def edge_distances(self, P):
        """Closest point on every edge and its distance. P: (n, 2) -> ((n, m, 2), (n, m))."""
        ab = self.b - self.a                                     # (m, 2)
        ap = P[:, None, :] - self.a[None]                        # (n, m, 2)
        t = np.clip(np.sum(ap * ab, axis=2) / np.sum(ab * ab, axis=1), 0.0, 1.0)
        c = self.a[None] + t[..., None] * ab[None]               # (n, m, 2)
        return c, np.linalg.norm(P[:, None, :] - c, axis=2)

    def closest_boundary(self, P):
        """Closest point on the polygon boundary. P: (n, 2) -> ((n, 2), (n,))."""
        c, d = self.edge_distances(P)
        k = np.argmin(d, axis=1)
        idx = np.arange(len(P))
        return c[idx, k], d[idx, k]

    def signed_distance(self, P):
        """Horizontal distance to the boundary, positive outside, negative inside."""
        _, d = self.closest_boundary(P)
        return np.where(self.contains(P), -d, d)

    def outward_normal(self, P):
        """
        Unit horizontal direction pointing away from the polygon (towards the
        outside) at the boundary point closest to each P. Also returns that
        boundary point and the signed distance of P.
        """
        c, d = self.closest_boundary(P)
        inside = self.contains(P)
        n = P - c
        n = np.where(inside[:, None], -n, n)
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        # P exactly on the boundary: fall back to the edge normal
        on_edge = norm[:, 0] < 1e-9
        if np.any(on_edge):
            n[on_edge] = self._edge_normal(c[on_edge])
            norm[on_edge] = 1.0
        return n / norm, c, np.where(inside, -d, d)

    def edge_features(self, p):
        """
        Per-edge signed distance of p (2,) and its gradient (outward unit
        vector), using each edge's own closest point.

        Returns (sd (m,), grad (m, 2), d (m,) unsigned).
        """
        ab = self.b - self.a
        ap = p - self.a
        t = np.clip(np.sum(ap * ab, axis=1) / np.sum(ab * ab, axis=1), 0.0, 1.0)
        diff = ap - t[:, None] * ab
        d = np.linalg.norm(diff, axis=1)
        sign = -1.0 if self.contains(p.reshape(1, 2))[0] else 1.0
        grad = sign * diff / np.maximum(d, 1e-12)[:, None]
        return sign * d, grad, d

    def trace(self, p, u, level, max_iter=60, tol=1e-4):
        """
        Signed distance t along p + t*u (u unit) to the first point where the
        signed distance equals `level`, searching towards the level set
        (t > 0 if sd(p) > level and u points towards the polygon).

        Uses Lipschitz-safe steps (|grad sd| = 1, never overshoots the first
        crossing) accelerated by Newton steps that are only accepted if they
        do not overshoot; a Newton overshoot is resolved by bisection.
        """
        def h(tau):                       # > 0 before the crossing, decreasing
            n_out, _, sd = self.outward_normal((p + s * tau * u).reshape(1, 2))
            return s * (sd[0] - level), float(n_out[0] @ u)

        n_out, _, sd = self.outward_normal(p.reshape(1, 2))
        s = 1.0 if sd[0] > level else -1.0
        tau = 0.0
        val, slope = s * (sd[0] - level), float(n_out[0] @ u)
        for _ in range(max_iter):
            if abs(val) < tol:
                break
            if slope < -0.05:
                tau_n = tau - val / slope                    # Newton candidate
                val_n, slope_n = h(tau_n)
                if val_n > -tol:
                    tau, val, slope = tau_n, val_n, slope_n
                    continue
                lo, hi = tau, tau_n                          # overshoot: bisection
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if h(mid)[0] > 0:
                        lo = mid
                    else:
                        hi = mid
                tau = 0.5 * (lo + hi)
                break
            tau += val                                       # safe step
            val, slope = h(tau)
        return s * tau

    def _edge_normal(self, C):
        ab = self.b - self.a
        ac = C[:, None, :] - self.a[None]
        t = np.clip(np.sum(ac * ab, axis=2) / np.sum(ab * ab, axis=1), 0.0, 1.0)
        k = np.argmin(np.linalg.norm(self.a[None] + t[..., None] * ab[None] - C[:, None, :], axis=2), axis=1)
        e = ab[k] / np.linalg.norm(ab[k], axis=1, keepdims=True)
        n = np.column_stack([e[:, 1], -e[:, 0]])                # right of the edge
        probe = C + 1e-3 * n
        return np.where(self.contains(probe)[:, None], -n, n)


class RestrictedAirspace:
    """
    Lisbon airport RPA restricted airspace (KMZ) used by the formation.

    - red (Área Proibida): no formation drone may fly inside it. The deterrent
      "wall" is placed `wall_offset` metres outside its border, and drones may
      fly in the band between the wall and the red border.
    - orange (Área 1): drones must stay below `orange_max_agl` metres above the
      ground.
    """
    enabled = True
    kind = "lisbon"
    joint_min_scale = JOINT_MIN_SCALE   # [m] (scaled with the scene: the arena slices)

    def __init__(self, sim):
        zones = read_kmz(Path(sim.airspace_kmz))
        red = [p for name, polys in zones if name.endswith(sim.red_zone_name) for p in polys]
        if len(red) != 1:
            raise ValueError(f"expected one '{sim.red_zone_name}' polygon in {sim.airspace_kmz}, found {len(red)}")
        self.red = Polygon2D(lonlat_to_enu(red[0]))
        self.orange = [
            Polygon2D(lonlat_to_enu(p))
            for name, polys in zones if name.endswith(sim.orange_zone_name) for p in polys
        ]
        self.red_zone_name = sim.red_zone_name
        self.orange_zone_name = sim.orange_zone_name
        # every KMZ zone (for plotting): [(name, [polygon (N, 2) ENU])]
        self.zones = [(name, [lonlat_to_enu(p) for p in polys]) for name, polys in zones]
        self.wall_offset = sim.wall_offset
        self.wall_joint_blend = sim.wall_joint_blend
        m = len(self.red.vertices)
        # edge i goes from vertex i to vertex i+1: edges i and i+1 meet at vertex i+1
        self.adjacent_edges = {frozenset((i, (i + 1) % m)) for i in range(m)}
        self.red_margin = sim.red_margin
        self.orange_max_agl = sim.orange_max_agl

    @staticmethod
    def ground_height(xy):
        """Terrain height under each (x, y). Flat ground at sea level for now."""
        return np.zeros(len(xy))

    def wall_normal(self, p):
        """
        Horizontal unit normal into the red zone for the wall frame at p (2,).

        Away from the joints between red border edges it is the exact normal
        of the nearest edge. When the two nearest edges a, b meet at a joint,
        the normal is turned between their (constant) inward normals through
        the joint's bisector, as a function of

            s = (d_b - d_a) / (d_a + d_b),   d_i = distance to edge line i

        (in a notch s = tan(phi)/tan(alpha): the angular position across the
        notch, so the turn is scale-free). It is the bisector for s = 0 and
        the nearest edge normal once |s| >= wall_joint_blend. Line distances
        and constant normals keep it continuous on both sides of the joint,
        including when the target passes through the joint into the red zone.
        """
        sd, grad, d = self.red.edge_features(p)
        a, b = (int(i) for i in np.argsort(d)[:2])
        if frozenset((a, b)) not in self.adjacent_edges:
            return -grad[a]
        n_a, n_b = self.red.inward_normals[a], self.red.inward_normals[b]
        bis = n_a + n_b
        if np.linalg.norm(bis) < 1e-9:
            return -grad[a]
        bis /= np.linalg.norm(bis)
        d_a = abs((p - self.red.a[a]) @ n_a)                 # distances to the edge lines
        d_b = abs((p - self.red.a[b]) @ n_b)
        s = (d_b - d_a) / max(d_a + d_b, self.joint_min_scale)      # > 0: closer to edge a
        f = np.clip(s / self.wall_joint_blend, -1.0, 1.0)
        # rotate the bisector towards n_a (f > 0) or n_b (f < 0); n_a and n_b
        # are symmetric about the bisector
        half = np.arctan2(bis[0] * n_a[1] - bis[1] * n_a[0], bis @ n_a)
        c, sn = np.cos(f * half), np.sin(f * half)
        return np.array([c * bis[0] - sn * bis[1], sn * bis[0] + c * bis[1]])

    def wall_frame(self, pos):
        """
        Local frame of the wall for a point (normally the target).

        The normal comes from wall_normal (turns smoothly at the joints between
        red border edges). Before the wall, the wall point is the exact first
        crossing of the wall along that normal line, so the mesh is always
        placed on the wall (at a joint's bisector, on the corner where the two
        walls meet). Past the wall, dist_surface is wall_offset minus the
        distance to red (it only drives the cone/sphere blend; tracing back
        along a tilted normal can run parallel to the wall in a notch).
        Outside red, the red border point is the crossing of the red border
        along the normal; inside red it is the closest red border point (both
        agree when the target crosses the border).

        Returns
        -------
        n_in : (3,) horizontal unit vector pointing into the red zone
        dist_surface : distance travelled past the wall towards the red zone,
            along n_in (< 0 before the wall)
        past_red : distance past the red border along n_in (> 0 inside red)
        boundary : (3,) red border point on the normal line, at pos height
        """
        p = np.asarray(pos, dtype=float).reshape(3)
        n_in = self.wall_normal(p[:2])
        sd = self.red.signed_distance(p[:2].reshape(1, 2))[0]
        if sd > self.wall_offset:
            dist_surface = -self.red.trace(p[:2], n_in, self.wall_offset)
        else:
            dist_surface = self.wall_offset - sd
        if sd >= 0:
            t_red = self.red.trace(p[:2], n_in, 0.0)
            border, past_red = p[:2] + t_red * n_in, -t_red
        else:
            border, past_red = self.red.closest_boundary(p[:2].reshape(1, 2))[0][0], -sd
        boundary = np.array([border[0], border[1], p[2]])
        return np.array([n_in[0], n_in[1], 0.0]), dist_surface, past_red, boundary

    def red_signed_distance(self, P):
        P = np.atleast_2d(P)
        return self.red.signed_distance(P[:, :2])

    def project(self, P):
        """
        Project points (n, 3) onto the allowed airspace: out of the red zone
        (at least `red_margin` from its border) and below the orange ceiling.
        """
        P = np.array(P, dtype=float, copy=True)
        m = self.red_margin
        n_out, c, sd = self.red.outward_normal(P[:, :2])
        bad = np.flatnonzero(sd < m)
        if len(bad):
            orig = P[bad, :2].copy()
            P[bad, :2] = c[bad] + m * n_out[bad]
            # In a notch (joint narrower than 2*margin) pushing a point out of
            # one edge's margin can put it inside the other edge's margin: the
            # closest feasible point is then the intersection of the two
            # margin lines (tip of the feasible wedge).
            sd2 = self.red.signed_distance(P[bad, :2])
            k = np.flatnonzero(sd2 < m - 1e-6)
            if len(k):
                _, d = self.red.edge_distances(orig[k])
                ea, eb = np.argsort(d, axis=1)[:, :2].T
                na, nb = self.red.inward_normals[ea], self.red.inward_normals[eb]
                ra = np.sum(na * self.red.a[ea], axis=1) - m
                rb = np.sum(nb * self.red.a[eb], axis=1) - m
                det = na[:, 0] * nb[:, 1] - na[:, 1] * nb[:, 0]
                ok = np.abs(det) > 1e-9
                x = (ra * nb[:, 1] - rb * na[:, 1]) / np.where(ok, det, 1.0)
                y = (na[:, 0] * rb - nb[:, 0] * ra) / np.where(ok, det, 1.0)
                P[bad[k[ok]], 0] = x[ok]
                P[bad[k[ok]], 1] = y[ok]

        if self.orange:
            in_orange = np.zeros(len(P), dtype=bool)
            for poly in self.orange:
                in_orange |= poly.contains(P[:, :2])
            if np.any(in_orange):
                ceiling = self.ground_height(P[in_orange, :2]) + self.orange_max_agl
                P[in_orange, 2] = np.minimum(P[in_orange, 2], ceiling)
        return P


class NoAirspace:
    """
    Empty world (figures only): no zones, nothing is projected.
    As before the KMZ, the formation reacts to a single plane wall at x = 0
    facing +x (circle on the wall -> cone -> sphere as the target crosses it).
    Same interface as RestrictedAirspace.
    """
    enabled = False
    kind = "none"

    def __init__(self, sim):
        self.red = None
        self.orange = []
        self.zones = []
        self.red_zone_name = sim.red_zone_name
        self.orange_zone_name = sim.orange_zone_name
        self.wall_offset = 0.0
        self.red_margin = sim.red_margin

    def wall_frame(self, pos):
        """Wall frame of RestrictedAirspace.wall_frame for the x = 0 wall (never past red)."""
        p = np.asarray(pos, dtype=float).reshape(3)
        return np.array([1.0, 0.0, 0.0]), float(p[0]), -np.inf, None

    def project(self, P):
        return np.array(P, dtype=float, copy=True)
