# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "shapely", "pyproj"]
# ///
"""OpenStreetMap (data/raw/osm_all.json) to what the build steps need, in grid meters.

Output (data/build/):
  osm_buildings.json   per building: id, tags, rings (first the exterior)
  surfaces.json        ground polygons: style, prio, road layer (road, cycle, foot or none), bridge, id
  osm_trees.json       single trees (natural=tree) as [x, y, height or null]
  osm_names.json       streets and named places for the search box: [x, y, label]

OpenStreetMap maps most roads as lines, so carriageways, pavements, paths, waterways and hedges are buffered into
polygons by a width per kind.
"""

import json
import re
from collections import defaultdict

import numpy as np
from shapely.geometry import LineString, Polygon, box, mapping
from shapely.ops import polygonize, unary_union

from config import BUILD, RAW, XMAX, XMIN, YMAX, YMIN
from geo import wgs_to_grid

MARGIN = 60
EXTENT = box(XMIN - MARGIN, YMIN - MARGIN, XMAX + MARGIN, YMAX + MARGIN)

# Full carriageway width in meters per highway kind, and its traffic layer.
ROADS = {
    "motorway": (11.0, "road"), "motorway_link": (6.0, "road"), "trunk": (9.0, "road"), "trunk_link": (6.0, "road"),
    "primary": (7.5, "road"), "primary_link": (6.0, "road"), "secondary": (7.0, "road"), "secondary_link": (5.5, "road"),
    "tertiary": (6.5, "road"), "tertiary_link": (5.0, "road"), "unclassified": (5.0, "road"), "residential": (5.5, "road"),
    "living_street": (4.5, "road"), "service": (3.5, "road"), "road": (5.0, "road"), "track": (3.0, "road"),
    "pedestrian": (4.0, "foot"), "footway": (1.6, "foot"), "path": (1.2, "foot"), "bridleway": (2.0, "foot"),
    "steps": (1.6, "foot"), "cycleway": (2.2, "cycle"),
}
SERVICE_WIDTH = {"driveway": 3.0, "parking_aisle": 5.0, "alley": 3.0}
WATERWAYS = {"river": 8.0, "canal": 10.0, "stream": 2.5, "brook": 2.0, "ditch": 1.5, "drain": 1.5}
SURFACE = {
    "asphalt": "asphalt", "paved": "asphalt", "concrete": "concrete", "concrete:plates": "concrete", "paving_stones": "tiles",
    "sett": "clinkers", "cobblestone": "clinkers", "bricks": "clinkers", "gravel": "gravel", "fine_gravel": "gravel",
    "compacted": "gravel", "pebblestone": "gravel", "unpaved": "dirt", "dirt": "dirt", "ground": "dirt", "earth": "dirt",
    "mud": "dirt", "grass": "grass", "sand": "sand", "wood": "bridge", "grass_paver": "grass_pavers",
}

LANDUSE = {
    "farmland": "field", "meadow": "meadow", "grass": "grass", "residential": "garden", "farmyard": "yard",
    "industrial": "concrete", "commercial": "concrete", "retail": "concrete", "allotments": "orchard", "orchard": "orchard",
    "vineyard": "orchard", "recreation_ground": "grass", "village_green": "grass", "cemetery": "grass", "forest": "forest",
    "construction": "dirt", "brownfield": "dirt", "greenfield": "meadow", "landfill": "dirt", "quarry": "sand",
    "reservoir": "water", "basin": "water", "railway": "gravel", "plant_nursery": "orchard", "flowerbed": "shrub",
}
NATURAL = {"wood": "forest", "scrub": "shrub", "heath": "heath", "wetland": "marsh", "grassland": "meadow", "water": "water",
           "sand": "sand", "bare_rock": "gravel", "shingle": "gravel", "tree_row": "shrub"}
LEISURE = {"park": "grass", "garden": "garden", "pitch": "grass", "playground": "sand", "swimming_pool": "water",
           "golf_course": "grass", "common": "meadow", "sports_centre": "grass", "nature_reserve": None}
AMENITY = {"parking": "asphalt", "school": "garden", "grave_yard": "grass"}


def to_grid(geom):
    return [wgs_to_grid(p["lat"], p["lon"]) for p in geom]


def ring_area(xy):
    a = np.asarray(xy)
    return 0.5 * abs(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1)))


def area_style(t):
    if t.get("building") or t.get("building:part"):
        return None
    for key, table in (("amenity", AMENITY), ("leisure", LEISURE), ("natural", NATURAL), ("landuse", LANDUSE)):
        if key in t and table.get(t[key]):
            return table[t[key]]
    if t.get("water"):
        return "water"
    if t.get("man_made") in ("wastewater_plant", "works"):
        return "concrete"
    if t.get("man_made") == "bridge":
        return "bridge"
    return None


def relation_polygons(rel):
    """Assemble a multipolygon relation from its member ways."""
    outer, inner = [], []
    for m in rel.get("members", []):
        if m.get("type") != "way" or not m.get("geometry"):
            continue
        line = LineString(to_grid(m["geometry"]))
        (inner if m.get("role") == "inner" else outer).append(line)
    polys = list(polygonize(unary_union(outer)))
    holes = list(polygonize(unary_union(inner))) if inner else []
    shape = unary_union(polys)
    if holes:
        shape = shape.difference(unary_union(holes))
    return shape


def side_buffer(line, width, side):
    """A strip of `width` meters along one side of a line (left: +1, right: -1), starting at its edge."""
    a = line.buffer(width, single_sided=True) if side > 0 else line.buffer(-width, single_sided=True)
    return a


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    els = json.load(open(RAW / "osm_all.json"))["elements"]
    buildings, surfaces, trees, names = [], [], [], []
    street_pts = defaultdict(list)

    def add(shape, style, prio, layer=None, bridge=False, oid=""):
        shape = shape.intersection(EXTENT)
        for part in getattr(shape, "geoms", [shape]):
            if part.geom_type != "Polygon" or part.is_empty or part.area < 0.5:
                continue
            surfaces.append({"style": style, "prio": prio, "layer": layer, "bridge": bridge, "id": oid,
                             "rings": [[[round(x, 2), round(y, 2)] for x, y in r.coords] for r in [part.exterior, *part.interiors]]})

    for el in els:
        t = el.get("tags", {})
        oid = f"{el['type']}/{el['id']}"
        if el["type"] == "node":
            x, y = wgs_to_grid(el["lat"], el["lon"])
            if not (XMIN <= x <= XMAX and YMIN <= y <= YMAX):
                continue
            if t.get("natural") == "tree":
                h = t.get("height")
                try:
                    h = float(h.split()[0]) if h else None
                except ValueError:
                    h = None
                trees.append([round(x, 2), round(y, 2), h])
            if t.get("name") and (t.get("amenity") or t.get("shop") or t.get("tourism") or t.get("historic") or t.get("place")
                                  or t.get("leisure") or t.get("man_made")):
                names.append([round(x, 1), round(y, 1), t["name"]])
            if t.get("addr:housenumber") and t.get("addr:street"):
                names.append([round(x, 1), round(y, 1), f"{t['addr:housenumber']} {t['addr:street']}"])
            continue

        if el["type"] == "relation":
            if t.get("type") != "multipolygon":
                continue
            shape = relation_polygons(el)
            if shape.is_empty:
                continue
            if t.get("building"):
                for part in getattr(shape, "geoms", [shape]):
                    buildings.append({"id": oid, "tags": t, "rings": [list(part.exterior.coords), *map(lambda r: list(r.coords), part.interiors)]})
                continue
            style = area_style(t)
            if style:
                add(shape, style, 0.3 if style == "water" else 0)
            continue

        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        pts = to_grid(geom)
        closed = len(pts) > 3 and np.allclose(pts[0], pts[-1])
        line = LineString(pts)
        if not line.intersects(EXTENT):
            continue

        if t.get("building") and closed:
            buildings.append({"id": oid, "tags": t, "rings": [pts]})
            continue
        hw = t.get("highway")
        if hw in ROADS and t.get("area") != "yes":
            width, layer = ROADS[hw]
            if hw == "service":
                width = SERVICE_WIDTH.get(t.get("service"), width)
            try:
                width = float(t.get("width", "").split()[0])
            except (ValueError, IndexError):
                pass
            if t.get("footway") == "sidewalk" or t.get("highway") == "footway":
                layer = "foot"
            surf = t.get("surface")
            style = SURFACE.get(surf) if surf else None
            if style is None:
                style = {"track": "dirt", "path": "dirt", "bridleway": "dirt"}.get(hw, "asphalt")
            if style == "grass":
                continue  # a mown strip in a field reads as the field itself
            bridge = t.get("bridge") not in (None, "no")
            if t.get("tunnel") not in (None, "no") or t.get("layer", "0").startswith("-"):
                continue
            prio = 2 if bridge else 0.5 + (0.01 if layer == "road" else 0)
            add(line.buffer(width / 2, cap_style="flat" if bridge else "round"), "bridge" if bridge else style, prio, layer, bridge, oid)
            # Pavements tagged on the road itself.
            sw = t.get("sidewalk") or t.get("sidewalk:both") and "both"
            sides = {"both": (1, -1), "left": (1,), "right": (-1,), "yes": (1, -1)}.get(sw or "", ())
            for side in sides:
                strip = side_buffer(line.offset_curve(side * width / 2), 1.8, side)
                add(strip, "tiles", 0.5, "foot", False, oid + f"/sidewalk{side}")
            if t.get("name") and layer == "road":
                street_pts[t["name"]] += pts
            continue
        if t.get("waterway") in WATERWAYS and t.get("tunnel") in (None, "no"):
            add(line.buffer(WATERWAYS[t["waterway"]] / 2), "water", 0.4, oid=oid)
            continue
        if t.get("barrier") == "hedge":
            add(line.buffer(0.8), "shrub", 0.45, oid=oid)
            continue
        if t.get("natural") == "tree_row":
            add(line.buffer(1.5), "shrub", 0.45, oid=oid)
            continue
        if t.get("railway") in ("rail", "light_rail", "disused", "abandoned") and t.get("tunnel") in (None, "no"):
            add(line.buffer(3.0), "gravel", 0.42, oid=oid)
            continue
        if closed:
            style = area_style(t)
            if hw in ("pedestrian", "footway", "service") and t.get("area") == "yes":
                style = SURFACE.get(t.get("surface"), "tiles")
            if style:
                poly = Polygon(pts).buffer(0)
                layer = None
                prio = 0.3 if style == "water" else 0
                if style in ("asphalt", "tiles", "concrete") and (t.get("amenity") == "parking" or hw):
                    prio, layer = 0.48, "road" if t.get("amenity") == "parking" else "foot"
                add(poly, style, prio, layer, oid=oid)
            if t.get("name") and (t.get("amenity") or t.get("leisure") or t.get("landuse") or t.get("natural") or t.get("historic")
                                  or t.get("tourism")):
                c = Polygon(pts).representative_point()
                names.append([round(c.x, 1), round(c.y, 1), t["name"]])

    # One search entry per street, at the middle of its longest stretch inside the area.
    for name, pts in street_pts.items():
        inside = [p for p in pts if XMIN <= p[0] <= XMAX and YMIN <= p[1] <= YMAX]
        if inside:
            a = np.array(inside)
            c = a[np.argmin(np.linalg.norm(a - a.mean(0), axis=1))]
            names.append([round(float(c[0]), 1), round(float(c[1]), 1), name])
    for b in buildings:
        t = b["tags"]
        if t.get("name") or (t.get("addr:housenumber") and t.get("addr:street")) or t.get("addr:housename"):
            c = Polygon(b["rings"][0]).representative_point()
            label = t.get("name") or t.get("addr:housename") or f"{t['addr:housenumber']} {t['addr:street']}"
            names.append([round(c.x, 1), round(c.y, 1), label])

    (BUILD / "osm_buildings.json").write_text(json.dumps(buildings))
    (BUILD / "surfaces.json").write_text(json.dumps(surfaces))
    (BUILD / "osm_trees.json").write_text(json.dumps(trees))
    # Placeholders and generic labels in OpenStreetMap are no use in a search box.
    junk = re.compile(r"^(\?+|crop field|field|welcome to .*)$", re.I)
    seen, uniq = set(), []
    for n in names:
        if junk.match(n[2].strip()):
            continue
        if n[2] not in seen:
            seen.add(n[2])
            uniq.append(n)
    (BUILD / "osm_names.json").write_text(json.dumps(uniq))
    from collections import Counter

    print(f"{len(buildings)} buildings, {len(surfaces)} surfaces {dict(Counter(s['style'] for s in surfaces))}, "
          f"{len(trees)} trees, {len(uniq)} names")


if __name__ == "__main__":
    main()
