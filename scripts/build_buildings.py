# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mapbox-earcut", "pillow", "scipy", "shapely", "tifffile", "imagecodecs", "pyproj"]
# ///
"""Turn OpenStreetMap footprints into buildings with roofs fitted to the Environment Agency lidar, as one triangle
soup for the renderer.

For every footprint the lidar surface minus the terrain inside it is fitted with a flat roof, a gable along either
axis of its minimum rotated rectangle, or a hipped roof, and the best fit wins. The roof is the lower envelope of
those planes over the footprint, and the walls run up to it, so gable ends come out as pentagons.

Output (data/build/):
  buildings.bin   float32 per vertex: x, y, z, nx, ny, nz, u, v, kind, building index, atlas u, atlas v (-1: none)
  roof_atlas.png  a 1 px placeholder: there is no open aerial photo to texture roofs from
  buildings.json  per building: id, attributes, wall and roof colors
  footprints.json per building: footprint ring in local meters, for the ground and tree steps

Local coordinates: x east, y up, z south, origin at the area center, y=0 at the building's ground level.
"""

import hashlib
import json
from collections import defaultdict

import mapbox_earcut as earcut
import numpy as np
from PIL import Image, ImageDraw

from shapely.geometry import Polygon

from config import BUILD, CX, CY, LIDAR_RES, XMAX, XMIN, YMAX, YMIN, write_area
from lidar import surface_and_terrain

KIND = {"GroundSurface": 0, "WallSurface": 1, "RoofSurface": 2}
COLS = 12

# Roof vertex blocks per building index (emit() keeps them for roof texturing; unused without an aerial photo).
ROOF_BLOCKS = defaultdict(list)

# OpenStreetMap building tags to the renderer's kinds (render/scene.js TYPES).
KINDS = {
    "church": "church", "chapel": "church", "cathedral": "church", "school": "school", "greenhouse": "greenhouse",
    "glasshouse": "greenhouse", "silo": "silo", "storage_tank": "tank", "barn": "barn", "farm_auxiliary": "barn",
    "agricultural": "barn", "cowshed": "barn", "stable": "barn", "warehouse": "barn", "industrial": "barn",
    "hangar": "barn", "garage": "shed", "garages": "shed", "shed": "shed", "outbuilding": "shed", "carport": "canopy",
    "hut": "shed", "roof": "canopy", "kiosk": "shed", "service": "shed", "boathouse": "shed",
}
# Roofs of an English village: clay tiles, slate and brown concrete tiles on houses; sheet metal on farm buildings.
HOUSE_ROOFS = [((150, 70, 52), 5), ((108, 56, 46), 2), ((66, 68, 74), 3), ((120, 92, 70), 2)]
SHED_ROOFS = [((120, 122, 124), 3), ((98, 114, 98), 2), ((88, 90, 96), 2), ((140, 64, 50), 1)]
DEFAULT_HEIGHT = {"shed": 2.6, "canopy": 3.0, "greenhouse": 3.0, "barn": 6.0, "house": 7.5}


def ring_area(xy):
    return 0.5 * (np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))


def extrude(rings_xy, z0, z1):
    """Prism from a footprint (rings in RD x/y, first exterior). Yields (kind, rings in 3D)."""
    out = []
    for i, r in enumerate(rings_xy):
        r = np.asarray(r, dtype=np.float64)[:, :2]
        if np.allclose(r[0], r[-1]):
            r = r[:-1]
        # Exterior counter-clockwise, holes clockwise, so wall normals point outward.
        if (ring_area(r) > 0) != (i == 0):
            r = r[::-1]
        for a, b in zip(r, np.roll(r, -1, axis=0)):
            if np.linalg.norm(b - a) < 0.05:
                continue
            out.append((1, [np.array([[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]])]))
    top = []
    for i, r in enumerate(rings_xy):
        r = np.asarray(r, dtype=np.float64)[:, :2]
        if np.allclose(r[0], r[-1]):
            r = r[:-1]
        if (ring_area(r) > 0) != (i == 0):
            r = r[::-1]
        top.append(np.column_stack([r, np.full(len(r), z1)]))
    out.append((2, top))
    return out


def long_axis(xy):
    """Direction (radians, RD x/y) of the longest footprint edge."""
    xy = np.asarray(xy)[:, :2]
    d = np.roll(xy, -1, axis=0) - xy
    i = int(np.argmax(np.hypot(d[:, 0], d[:, 1])))
    return float(np.arctan2(d[i, 1], d[i, 0]))


def newell_normal(pts):
    n = np.zeros(3)
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else None


def triangulate(rings):
    """rings: list of (N,3) arrays in RD meters, first is the exterior. Returns (M,3,3) triangles."""
    n = newell_normal(rings[0])
    if n is None:
        return np.zeros((0, 3, 3)), None
    drop = int(np.argmax(np.abs(n)))
    keep = [i for i in range(3) if i != drop]
    pts = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    idx = earcut.triangulate_float64(pts[:, keep], ends)
    tris = pts[idx].reshape(-1, 3, 3)
    # earcut does not keep the winding; flip triangles that face away from the surface normal.
    tn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = (tn @ n) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris, n


def stable_rand(key, salt=""):
    return int(hashlib.md5((key + salt).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def emit(verts, surfaces, bi, ground, center_u=False):
    """surfaces: iterable of (kind, rings in RD meters with absolute z). Appends vertices; returns roof triangles.

    center_u puts u=0 in the middle of each wall, so towers can draw a clock or louvres centered on a face."""
    roof_tris = []
    for kind, rings in surfaces:
        if kind == 0:
            continue  # the floor is never visible
        tris, n = triangulate(rings)
        if n is None or len(tris) == 0:
            continue
        if kind == 2:
            roof_tris.append(tris)
        # Wall pattern coordinates: u runs along the wall, v is height above ground.
        if kind == 1:
            h = np.array([n[1], -n[0]])
            h = h / (np.linalg.norm(h) or 1)
            u = tris[..., :2] @ h
            if center_u:
                u = u - 0.5 * (u.min() + u.max())
            uv = np.stack([u, tris[..., 2] - ground], axis=-1)
        else:
            uv = tris[..., :2] - [CX, CY]
        local = np.stack([tris[..., 0] - CX, tris[..., 2] - ground, CY - tris[..., 1]], axis=-1)
        normal = [n[0], n[2], -n[1]]
        block = np.concatenate(
            [local, np.broadcast_to(normal, local.shape), uv, np.full(local.shape[:2] + (1,), kind), np.full(local.shape[:2] + (1,), bi),
             np.full(local.shape[:2] + (2,), -1.0)],
            axis=-1,
        ).reshape(-1, COLS)
        verts.append(block)
        if kind == 2:
            ROOF_BLOCKS[bi].append((block, tris))
    return roof_tris


def ccw(ring):
    ring = np.asarray(ring, dtype=np.float64)
    return ring if ring_area(ring) > 0 else ring[::-1]


def rect_ring(cx, cy, w, d, ang):
    c, s = np.cos(ang), np.sin(ang)
    pts = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    return ccw([(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts])


def ngon_ring(cx, cy, r, n, ang):
    return ccw([(cx + r * np.cos(ang + 2 * np.pi * (i + 0.5) / n), cy + r * np.sin(ang + 2 * np.pi * (i + 0.5) / n)) for i in range(n)])


def pyramid(ring, z0, apex):
    """Roof triangles from a footprint ring up to a point above its center."""
    ring = ccw(ring)
    cx, cy = ring.mean(0)
    return [(2, [np.array([[a[0], a[1], z0], [b[0], b[1], z0], [cx, cy, apex]])]) for a, b in zip(ring, np.roll(ring, -1, axis=0))]


def frustum(bottom, top, z0, z1):
    """Walls between two rings with the same vertex count, for tapered masts."""
    bottom, top = ccw(bottom), ccw(top)
    out = []
    for i in range(len(bottom)):
        j = (i + 1) % len(bottom)
        out.append((1, [np.array([[*bottom[i], z0], [*bottom[j], z0], [*top[j], z1], [*top[i], z1]])]))
    out.append((2, [np.column_stack([top, np.full(len(top), z1)])]))
    return out


def part(meta, pid, kind, axis, height, wall, roof, param=0.0):
    meta.append({"id": f"{pid}-{kind}", "kind": kind, "axis": round(float(axis), 4), "param": param, "height": height,
                 "wall_fixed": list(wall), "roof_fixed": list(roof), "rand": 0.5, "roof_rgb": None})
    return len(meta) - 1


def add_landmark_parts(verts, meta, pid, lm, ground, axis, roof_tris):
    """Towers, roof turrets and crosses that 3D BAG smooths away."""
    wall, roof = lm.get("wall", (150, 80, 62)), lm.get("roof", (70, 72, 80))
    if "tower" in lm:
        t = lm["tower"]
        ring = rect_ring(t["x"], t["y"], t["w"], t["w"], axis)
        bi = part(meta, pid, "tower", axis, t["top"], wall, roof, param=t["body"])
        emit(verts, extrude([ring], ground, ground + t["body"]), bi, ground, center_u=True)
        emit(verts, pyramid(ring, ground + t["body"], ground + t["top"]), bi, ground)
        if t.get("cross"):
            bi = part(meta, pid, "cross", axis, t["top"] + 2, (196, 170, 96), (196, 170, 96))
            top = ground + t["top"]
            emit(verts, extrude([rect_ring(t["x"], t["y"], 0.3, 0.3, axis)], top - 0.4, top + 2.2), bi, ground)
            emit(verts, extrude([rect_ring(t["x"], t["y"], 1.4, 0.3, axis)], top + 1.2, top + 1.5), bi, ground)
    if "turret" in lm:
        t = lm["turret"]
        ring = ngon_ring(t["x"], t["y"], t["r"], 8, axis)
        bi = part(meta, pid, "turret", axis, t["top"], (240, 238, 230), (62, 64, 72), param=t["body"])
        emit(verts, extrude([ring], ground + t["base"], ground + t["body"]), bi, ground, center_u=True)
        emit(verts, pyramid(ring, ground + t["body"], ground + t["top"]), bi, ground)
    if lm.get("bell") and roof_tris:
        # A small white bell turret on the highest point of the roof, as on the school.
        pts = np.concatenate(roof_tris).reshape(-1, 3)
        x, y, zmax = pts[np.argmax(pts[:, 2])]
        ring = rect_ring(x, y, 1.5, 1.5, axis)
        bi = part(meta, pid, "turret", axis, zmax - ground + 3.4, (240, 238, 230), (70, 74, 82), param=zmax - ground + 2.0)
        emit(verts, extrude([ring], zmax - 0.4, zmax + 2.0), bi, ground, center_u=True)
        emit(verts, pyramid(ring, zmax + 2.0, zmax + 3.4), bi, ground)


def add_masts(verts, meta, masts):
    for m in masts:
        bot = rect_ring(m["x"], m["y"], m["base"], m["base"], 0.3)
        top = rect_ring(m["x"], m["y"], m["tip"], m["tip"], 0.3)
        bi = part(meta, "mast", "mast", 0.3, m["height"], (184, 188, 192), (150, 154, 158), param=m["height"])
        emit(verts, frustum(bot, top, 0.0, m["height"]), bi, 0.0, center_u=True)




def greenhouse_height(year):
    return 3.0


def pick(options, r):
    total = sum(w for _, w in options)
    for c, w in options:
        r -= w / total
        if r <= 0:
            return list(c)
    return list(options[0][0])


class Lidar:
    """Height above the terrain (lidar surface minus terrain) at LIDAR_RES, row 0 north."""

    def __init__(self):
        dsm, dtm = surface_and_terrain()
        self.ndsm = np.nan_to_num(dsm - dtm, nan=0.0)

    def samples(self, poly):
        """Pixel centers inside the footprint, shrunk half a meter so walls and eaves don't bleed in: (xy, height)."""
        inner = poly.buffer(-0.5)
        if inner.is_empty or inner.area < 4:
            inner = poly
        x0, y0, x1, y1 = inner.bounds
        c0, c1 = int((x0 - XMIN) / LIDAR_RES), int((x1 - XMIN) / LIDAR_RES) + 1
        r0, r1 = int((YMAX - y1) / LIDAR_RES), int((YMAX - y0) / LIDAR_RES) + 1
        h, w = self.ndsm.shape
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, w), min(r1, h)
        if c1 <= c0 or r1 <= r0:
            return np.zeros((0, 2)), np.zeros(0)
        gy, gx = np.mgrid[r0:r1, c0:c1]
        px = XMIN + (gx + 0.5) * LIDAR_RES
        py = YMAX - (gy + 0.5) * LIDAR_RES
        img = Image.new("1", (c1 - c0, r1 - r0), 0)
        ring = [((x - XMIN) / LIDAR_RES - c0 - 0.5, (YMAX - y) / LIDAR_RES - r0 - 0.5) for x, y in inner.exterior.coords]
        ImageDraw.Draw(img).polygon(ring, fill=1)
        m = np.array(img, dtype=bool)
        return np.column_stack([px[m], py[m]]), self.ndsm[r0:r1, c0:c1][m]


def frame(poly):
    """Center, long-axis unit vector, half length and half width of the minimum rotated rectangle."""
    r = np.array(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = r[1] - r[0], r[2] - r[1]
    if np.linalg.norm(e1) < np.linalg.norm(e2):
        e1, e2 = e2, e1
    u = e1 / (np.linalg.norm(e1) or 1)
    return r.mean(0), u, np.linalg.norm(e1) / 2, np.linalg.norm(e2) / 2


def fit_roof(xy, h, poly):
    """Pick the roof model that fits the lidar best. Returns (planes, eave, ridge, shape) where planes are
    (a, b, c) with z = a + b x + c y in grid meters, or a flat height when planes is None."""
    if len(h) < 6:
        return None, float(np.percentile(h, 80)) if len(h) else None, None, "flat"
    c, u, L, W = frame(poly)
    v_dir = np.array([-u[1], u[0]])
    du, dv = (xy - c) @ u, (xy - c) @ v_dir
    flat_h = float(np.percentile(h, 70))
    flat_rmse = float(np.sqrt(np.mean((h - flat_h) ** 2)))
    best = (flat_rmse * 0.85, None, flat_h, None, "flat")  # pitched has to fit clearly better
    for shape in ("gable", "cross_gable", "hip"):
        if shape == "gable":
            feat = np.abs(dv)
        elif shape == "cross_gable":
            feat = np.abs(du)
        else:
            feat = np.maximum(np.abs(dv), np.abs(du) - max(L - W, 0))
        keep = np.ones(len(h), bool)
        for _ in range(3):  # refit without outliers: trees over the roof, chimneys, dormers
            A = np.column_stack([np.ones(keep.sum()), -feat[keep]])
            (r, slope), *_ = np.linalg.lstsq(A, h[keep], rcond=None)
            res = h - (r - slope * feat)
            keep = np.abs(res) < max(0.8, 2.5 * np.std(res[keep]))
            if keep.sum() < 6:
                break
        rmse = float(np.sqrt(np.mean(res[keep] ** 2))) if keep.sum() else 99
        half = W if shape != "cross_gable" else L
        eave = r - slope * half
        if not (0.25 < slope < 1.7 and eave > 1.8 and r - eave > 1.0):
            continue
        if rmse < best[0]:
            best = (rmse, (shape, r, slope, c, u, v_dir, L, W), eave, r, shape)
    _, model, eave, ridge, shape = best
    if model is None:
        return None, flat_h, None, "flat"
    shape, r, s, c, u, v, L, W = model
    # Planes as z = a + b x + c y. A gable is two planes falling away from the ridge; a hip adds two at the ends.
    planes = []
    def plane(direction, offset):
        # z = r - s * (direction . (p - c) - offset)
        return (r + s * float(direction @ c) + s * offset, -s * direction[0], -s * direction[1])
    if shape in ("gable", "hip"):
        planes += [plane(v, 0), plane(-v, 0)]
    if shape == "cross_gable":
        planes += [plane(u, 0), plane(-u, 0)]
    if shape == "hip":
        planes += [plane(u, max(L - W, 0)), plane(-u, max(L - W, 0))]
    return planes, eave, ridge, shape


def zof(plane, x, y):
    return plane[0] + plane[1] * x + plane[2] * y


def roof_surfaces(rings, planes):
    """Walls up to the lower envelope of the planes and one roof face per plane, as emit() surfaces."""
    poly = Polygon(rings[0], rings[1:]).buffer(0)
    env = lambda x, y: min(zof(p, x, y) for p in planes)
    out = []
    for i, r in enumerate(rings):
        r = np.asarray(r, dtype=np.float64)[:, :2]
        if np.allclose(r[0], r[-1]):
            r = r[:-1]
        if (ring_area(r) > 0) != (i == 0):
            r = r[::-1]
        for a, b in zip(r, np.roll(r, -1, axis=0)):
            if np.linalg.norm(b - a) < 0.05:
                continue
            # Split the edge where the topmost plane changes, so every wall piece is planar.
            ts = {0.0, 1.0}
            for j in range(len(planes)):
                for k in range(j + 1, len(planes)):
                    fa = zof(planes[j], *a) - zof(planes[k], *a)
                    fb = zof(planes[j], *b) - zof(planes[k], *b)
                    if fa * fb < 0:
                        ts.add(fa / (fa - fb))
            ts = sorted(ts)
            for t0, t1 in zip(ts, ts[1:]):
                p, q = a + (b - a) * t0, a + (b - a) * t1
                if np.linalg.norm(q - p) < 0.02:
                    continue
                out.append((1, [np.array([[p[0], p[1], 0.0], [q[0], q[1], 0.0], [q[0], q[1], env(*q)], [p[0], p[1], env(*p)]])]))
    big = 1e4
    for k, pk in enumerate(planes):
        region = poly
        for j, pj in enumerate(planes):
            if j == k:
                continue
            # Where plane k lies below plane j: (ak - aj) + (bk - bj) x + (ck - cj) y <= 0.
            region = region.intersection(halfplane(pk[0] - pj[0], pk[1] - pj[1], pk[2] - pj[2], poly.centroid, big))
            if region.is_empty:
                break
        for part in getattr(region, "geoms", [region]):
            if part.geom_type != "Polygon" or part.area < 0.05:
                continue
            rr = [np.array(part.exterior.coords)[:-1], *[np.array(h.coords)[:-1] for h in part.interiors]]
            rr = [ccw(rr[0]), *[x if ring_area(x) < 0 else x[::-1] for x in rr[1:]]]
            out.append((2, [np.column_stack([x, [zof(pk, *p) for p in x]]) for x in rr]))
    return out


def halfplane(a, b, c, center, big):
    """The polygon {a + b x + c y <= 0} within a big square around center."""
    cx, cy = center.x, center.y
    sq = [(cx - big, cy - big), (cx + big, cy - big), (cx + big, cy + big), (cx - big, cy + big)]
    f = lambda p: a + b * p[0] + c * p[1]
    out = []
    for p, q in zip(sq, sq[1:] + sq[:1]):
        fp, fq = f(p), f(q)
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return Polygon(out) if len(out) >= 3 else Polygon()


def main():
    write_area()
    landmarks = json.loads((BUILD / "landmarks.json").read_text()) if (BUILD / "landmarks.json").exists() else {}
    named = landmarks.get("buildings", {})
    lidar = Lidar()
    verts, meta, footprints = [], [], []
    shapes = defaultdict(int)
    for b in json.loads((BUILD / "osm_buildings.json").read_text()):
        t = b["tags"]
        rings = [np.array(r, dtype=np.float64)[:, :2] for r in b["rings"]]
        poly = Polygon(rings[0], rings[1:]).buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon" or poly.area < 2:
            continue
        cx, cy = poly.centroid.x, poly.centroid.y
        if not (XMIN <= cx <= XMAX and YMIN <= cy <= YMAX):
            continue
        rings = [np.array(poly.exterior.coords)[:-1], *[np.array(r.coords)[:-1] for r in poly.interiors]]
        kind = KINDS.get(t.get("building"), "house")
        if t.get("amenity") == "place_of_worship":
            kind = "church"
        lm = named.get(b["id"])
        xy, h = lidar.samples(poly)
        planes, eave, ridge, shape = fit_roof(xy, h, poly) if kind not in ("greenhouse", "canopy") else (None, None, None, "flat")
        if lm and "ruin" in lm:
            # A roofless shell: the outline with a hole a wall's thickness in, so you look down inside.
            height, shape = lm["ruin"]["height"], "ruin"
            inner = poly.buffer(-lm["ruin"]["wall"], join_style="mitre")
            holes = [np.array(g.exterior.coords)[:-1] for g in getattr(inner, "geoms", [inner]) if not g.is_empty]
            roof_tris = emit(verts, extrude([rings[0], *holes], 0.0, height), len(meta), 0.0)
        elif planes is None:
            height = eave if eave and eave > 1.8 else DEFAULT_HEIGHT.get(kind, 4.0)
            if kind == "greenhouse":
                height = DEFAULT_HEIGHT["greenhouse"]
            roof_tris = emit(verts, extrude(rings, 0.0, height), len(meta), 0.0)
        else:
            height = ridge
            roof_tris = emit(verts, roof_surfaces(rings, planes), len(meta), 0.0)
        shapes[shape] += 1
        if kind == "house" and poly.area > 400 and height < 10:
            kind = "barn"
        footprints.append([[round(x - CX, 2), round(CY - y, 2)] for x, y in rings[0]])
        rand = stable_rand(b["id"])
        if lm:
            kind = lm["style"]
        meta.append({
            "id": b["id"], "kind": kind, "axis": round(long_axis(rings[0]), 4), "glass": kind == "greenhouse",
            "year": None, "floors": None, "roof": shape, "area": round(poly.area, 1), "height": round(float(height), 2),
            "roof_rgb": pick(SHED_ROOFS if kind in ("barn", "shed", "canopy") else HOUSE_ROOFS, stable_rand(b["id"], "roof")),
            "rand": round(rand, 4),
        })
        if t.get("roof:colour", "").startswith("#") and len(t["roof:colour"]) == 7:
            meta[-1]["roof_fixed"] = [int(t["roof:colour"][i : i + 2], 16) for i in (1, 3, 5)]
        if lm:
            meta[-1].update({"name": lm["name"], "wall_fixed": lm.get("wall"), "roof_fixed": lm.get("roof"), "frame_fixed": lm.get("frame")})
            add_landmark_parts(verts, meta, b["id"], lm, 0.0, long_axis(rings[0]), roof_tris)
    add_masts(verts, meta, landmarks.get("masts", []))
    arr = np.concatenate(verts).astype(np.float32)
    arr.tofile(BUILD / "buildings.bin")
    Image.new("RGB", (1, 1), (128, 128, 128)).save(BUILD / "roof_atlas.png")
    (BUILD / "buildings.json").write_text(json.dumps(meta))
    (BUILD / "footprints.json").write_text(json.dumps(footprints))
    from collections import Counter

    print(f"{len(meta)} buildings, {len(arr) // 3} triangles, {Counter(m['kind'] for m in meta)}, roofs {dict(shapes)}")


if __name__ == "__main__":
    main()
