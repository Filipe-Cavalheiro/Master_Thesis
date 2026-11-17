"""
Base map and zone data of the Motivation airspace slide.

Top view of the Lisbon OSM meshes of the Gazebo world (buildings, roads, rail, water), drawn
dark so the restricted airspace stands out, plus the ANAC zones of Lisbon airport (LPPT) and
a few public infrastructure points, all in the ENU frame of the simulations (metres, origin at
the airport reference point).

Writes
  assets/images/lisbon_map.png   base map, extent EXTENT (no zones, no labels)
  assets/data/lisbon_zones.npz   extent, zone polygons (red, orange) and infrastructure points

usage (repository environment, ~1 min):
  .venv/bin/python Presentation/tools/make_lisbon_map.py
"""
import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "Python", "flight_sims", "swarm_MPC_voronoi", "Python")
MESHES = os.path.join(ROOT, "Gazebo", "models", "lisbon_terrain")
OUT_PNG = os.path.join(HERE, "..", "assets", "images", "lisbon_map.png")
OUT_NPZ = os.path.join(HERE, "..", "assets", "data", "lisbon_zones.npz")
sys.path.insert(0, SIM)
os.chdir(SIM)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from restricted_airspace import RestrictedAirspace, lonlat_to_enu  # noqa: E402
from Structs_sim import SimulationConfig  # noqa: E402

EXTENT = (-6700.0, 5700.0, -10500.0, 10500.0)     # E min, E max, N min, N max [m]
PX_PER_M = 0.12                                   # 1490 x 2520 px

# (label, lat, lon): public infrastructure under the airport zones
POIS = [
    ("Hospital de Santa Maria", 38.7488, -9.1606),
    ("Hospital Pulido Valente", 38.7662, -9.1567),
    ("University of Lisbon", 38.7527, -9.1580),
    ("Estádio José Alvalade", 38.7612, -9.1609),
    ("Estádio da Luz", 38.7527, -9.1847),
    ("Gare do Oriente", 38.7678, -9.0991),
]

# layer: (colour, minimum upward normal component)
LAYERS = [
    ("water", "#0E2338"),
    ("trails", "#1A1A1A"),
    ("roads", "#3A3D42"),
    ("rail", "#4A4038"),
    ("buildings", "#6E7074"),
]


def mesh_pose():
    sdf = open(os.path.join(MESHES, "model.sdf")).read()
    pose = [float(v) for v in re.search(r"<pose>([^<]+)</pose>", sdf).group(1).split()]
    scale = float(re.search(r"<scale>([\d.]+)", sdf).group(1))
    return scale, np.array(pose[:2]), pose[5]


def top_triangles(name, scale, trans, yaw):
    data = open(os.path.join(MESHES, "meshes", f"map_{name}.stl"), "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    tri = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    v = tri["v"].astype(float)
    e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    nz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    keep = (nz > 0) & (v[:, :, 2].min(axis=1) > 0)      # upward faces, not the flat base
    xy = v[keep, :, :2] * scale
    c, s = np.cos(yaw), np.sin(yaw)
    xy = xy @ np.array([[c, s], [-s, c]]) + trans
    x0, x1, y0, y1 = EXTENT
    inside = (xy[:, :, 0].max(1) > x0) & (xy[:, :, 0].min(1) < x1) & (xy[:, :, 1].max(1) > y0) & (xy[:, :, 1].min(1) < y1)
    return xy[inside]


def clip_to_extent(P):
    """Sutherland-Hodgman clip of polygon P (N, 2) to the map rectangle."""
    x0, x1, y0, y1 = EXTENT
    for axis, lim, keep_above in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        out = []
        for a, b in zip(P, np.roll(P, -1, axis=0)):
            ina, inb = (a[axis] >= lim) == keep_above, (b[axis] >= lim) == keep_above
            if ina:
                out.append(a)
            if ina != inb:
                out.append(a + (lim - a[axis]) / (b[axis] - a[axis]) * (b - a))
        P = np.array(out)
    return P


def main():
    scale, trans, yaw = mesh_pose()
    x0, x1, y0, y1 = EXTENT
    w, h = (x1 - x0) * PX_PER_M, (y1 - y0) * PX_PER_M
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#101010")
    fig.patch.set_facecolor("#101010")
    for name, col in LAYERS:
        tri = top_triangles(name, scale, trans, yaw)
        print(name, len(tri))
        if name == "buildings":
            roofs = tri
        ax.add_collection(PolyCollection(tri, facecolors=col, edgecolors=col, linewidths=0.15, antialiased=True))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT_PNG, dpi=100, facecolor=fig.get_facecolor())
    print("wrote", OUT_PNG)

    air = RestrictedAirspace(SimulationConfig())
    orange = [p for name, polys in air.zones if name.endswith(air.orange_zone_name) for p in polys]
    from matplotlib.path import Path as MPath
    cen = roofs.mean(axis=1)
    e1, e2 = roofs[:, 1] - roofs[:, 0], roofs[:, 2] - roofs[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    in_any = air.red.contains(cen) | np.any([MPath(p).contains_points(cen) for p in orange], axis=0)
    print(f"building footprint under the zones: {area[in_any].sum() / 1e6:.1f} km2 "
          f"(red {area[air.red.contains(cen)].sum() / 1e6:.1f} km2)")
    poi_xy = lonlat_to_enu(np.array([[lon, lat] for _, lat, lon in POIS]))
    in_red = air.red.contains(poi_xy)
    np.savez_compressed(
        OUT_NPZ,
        extent=np.array(EXTENT),
        red=clip_to_extent(air.red.vertices),
        poi_names=np.array([p[0] for p in POIS]),
        poi_xy=poi_xy,
        **{f"orange{k}": clip_to_extent(p) for k, p in enumerate(orange)},
    )
    in_orange = np.any([MPath(p).contains_points(poi_xy) for p in orange], axis=0)
    for (name, *_), xy, r, o in zip(POIS, poi_xy, in_red, in_orange):
        print(f"{name:28s} E {xy[0]:7.0f} N {xy[1]:7.0f}  red {r}  orange {o}")
    print("wrote", OUT_NPZ)


if __name__ == "__main__":
    main()
