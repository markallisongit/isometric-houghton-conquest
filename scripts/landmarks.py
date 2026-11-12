# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyproj"]
# ///
"""Named places of Houghton Conquest: style overrides and extra geometry for buildings, plus the viewer's place list.

Sources: the National Heritage List for England (Historic England, Open Government Licence), Wikipedia, and
OpenStreetMap; tower and wall heights measured in the Environment Agency lidar. Texts are written for this map.

Output: data/build/landmarks.json
"""

import json

import numpy as np

from config import BUILD
from geo import wgs_to_grid

NHLE = "https://historicengland.org.uk/listing/the-list/list-entry/"
WIKI = "https://en.wikipedia.org/wiki/"

# Buildings found by a point (`at`, grid meters) inside their OpenStreetMap footprint.
BUILDINGS = [
    {
        "name": "Church of All Saints",
        "kind": "Parish church · 14th century · Grade I",
        "text": "The parish church, built in the 14th century, with a battlemented west tower that holds six bells. Over the north door is a 15th-century wall painting of St Christopher, and the east window was put in in 1880 by John William Burgon, Dean of Chichester. In October 2018 thieves stripped the lead from the whole roof.",
        "more": NHLE + "1113919",
        "at": (504360, 241429),
        "style": "church",
        "wall": (164, 134, 100),
        "roof": (112, 116, 122),
        # Tower from the lidar: a block of about 7 m square at the west end, flat at 20 m above the churchyard.
        "tower": {"x": 504344.5, "y": 241425.5, "w": 6.6, "body": 20.0, "top": 20.4},
    },
    {
        "name": "Houghton House",
        "kind": "Ruin · 1615 · Grade I · English Heritage",
        "text": "Built from about 1615 for Mary Sidney, Countess of Pembroke, poet and patron, possibly to designs by Inigo Jones; James I came to see it in 1621. It is said to be John Bunyan's House Beautiful in The Pilgrim's Progress. In 1794 the Duke of Bedford had it stripped and the roof taken off. The shell, on the ridge above the valley, is free to visit.",
        "more": NHLE + "1321465",
        "web": "https://www.english-heritage.org.uk/visit/places/houghton-house/",
        "at": (503922, 239484),
        "style": "house",
        "wall": (196, 160, 110),
        "roof": (196, 160, 110),
        # No roof: the walls stand about 11 m high in the lidar, so the house is a shell, open to the sky.
        "ruin": {"height": 11.0, "wall": 1.2},
    },
    {
        "name": "The Old Rectory",
        "kind": "Grade II",
        "text": "The old rectory of the parish. Rectors of Houghton Conquest included the antiquarian Thomas Archer, from 1589 to 1631, and the theologian Henry John Rose, from 1837 to 1873.",
        "more": NHLE + "1321739",
        "at": (504522, 240893),
    },
    {
        "name": "Village Farmhouse",
        "kind": "Farmhouse · Grade II",
        "more": NHLE + "1321738",
        "at": (504743, 241624),
    },
    {
        "name": "Home Farmhouse",
        "kind": "Farmhouse · Grade II",
        "more": NHLE + "1113920",
        "at": (504545, 241598),
    },
    {
        "name": "Hill Farmhouse",
        "kind": "Farmhouse · Grade II",
        "text": "A listed farmhouse at Chapel End, with a listed range of barns about ten metres to the south-east.",
        "more": NHLE + "1321737",
        "at": (505407, 241895),
    },
    {
        "name": "Lodge Piece",
        "kind": "Gardener's house of Houghton House · Grade II",
        "more": NHLE + "1114418",
        "at": (503929, 238949),
    },
    {
        "name": "Holly Walk Cottage",
        "kind": "Grade II",
        "more": NHLE + "1114426",
        "at": (503304, 238620),
    },
    {
        "name": "16 and 17 How End Road",
        "kind": "Cottages at How End · Grade II",
        "more": NHLE + "1113921",
        "at": (503388, 240358),
    },
    {
        "name": "Houghton Conquest Stores & Post Office",
        "kind": "Village shop and post office",
        "text": "The village shop is in a house of the 1850s that, according to the Victoria County History, was built partly from an old manor house, most likely Conquestbury, the seat of the Conquest family, who gave the village the second half of its name.",
        "address": "35 High Street",
        "at": (504571.9, 241600.8),
    },
    {
        "name": "Houghton Conquest Methodist Church",
        "kind": "Methodist church",
        "at": (504455, 241346),
    },
    {
        "name": "Houghton Conquest Primary School",
        "kind": "School",
        "web": "https://hcschool.org.uk/",
        "at": (504647, 241584),
        "style": "school",
    },
]

# Places without a building of their own.
POINTS = [
    {
        "name": "War memorial",
        "kind": "War memorial · Grade II",
        "more": NHLE + "1431372",
        "at": (504400, 241441),
    },
    {
        "name": "The Knife and Cleaver",
        "kind": "Pub",
        "address": "The Grove",
        "at": (504405.6, 241393.4),
    },
    {
        "name": "Royal Oak",
        "kind": "Pub",
        "address": "High Street",
        "at": (504688.1, 241603.9),
    },
    {
        "name": "Bury Farm",
        "kind": "Site of Conquestbury manor",
        "text": "The Conquest family held a manor here from the 13th to the 18th century. Their manor house, Conquestbury, stood at the south-east end of the village, where Bury Farm is now, beside London Lane, and was left to fall down when the family moved away.",
        "more": WIKI + "Houghton_Conquest",
        "at": (505165, 240606),
    },
    {
        "name": "King's Wood and Glebe Meadows",
        "kind": "Site of Special Scientific Interest · Local Nature Reserve",
        "text": "36 hectares of ash and maple wood on heavy clay, rare in lowland England, and old hay meadows with ponds and hedges. Wood anemone and wood melick show the wood is ancient; in the meadows grow cowslips, great burnet and adder's-tongue fern. The rare plants were found by a local boy, Peter Sollars. A footpath runs in from Rectory Lane.",
        "more": WIKI + "Kings_Wood_and_Glebe_Meadows,_Houghton_Conquest",
        "at": (504531, 239447),
    },
    {
        "name": "The Devil's Jumps",
        "kind": "Standing stone",
        "text": "Local legend says that if you run around the stone holding your breath, the Devil will appear.",
        "at": (505487.1, 238505),
    },
    {
        "name": "Wartime pillbox",
        "kind": "Second World War",
        "at": (503213.2, 238734.0),
    },
    {
        "name": "Midland Main Line",
        "kind": "Railway · London St Pancras to Bedford and the north",
        "at": (503150, 241200),
    },
]

CATEGORIES = [
    ("listed", "Listed buildings", [
        "Church of All Saints", "Houghton House", "The Old Rectory", "Village Farmhouse", "Home Farmhouse", "Hill Farmhouse",
        "Lodge Piece", "Holly Walk Cottage", "16 and 17 How End Road", "War memorial",
    ]),
    ("village", "Village life", [
        "Houghton Conquest Stores & Post Office", "The Knife and Cleaver", "Royal Oak", "Houghton Conquest Primary School",
        "Houghton Conquest Methodist Church",
    ]),
    ("land", "Woods and legends", ["King's Wood and Glebe Meadows", "The Devil's Jumps", "Bury Farm", "Wartime pillbox"]),
]

# What the viewer shows when a place is clicked; the rest of an entry is for the render.
INFO = ("note", "kind", "text", "address", "web", "more", "photo", "stop")

# Lattice communication masts on the ridge by Houghton House (OpenStreetMap, with their tagged heights).
MASTS = [
    {"at": (504013.0, 239085.4), "height": 12.0, "base": 1.6, "tip": 0.5},
    {"at": (504184.5, 239184.0), "height": 15.0, "base": 1.8, "tip": 0.5},
    {"at": (504001.2, 239178.7), "height": 12.0, "base": 1.6, "tip": 0.5},
]


def point_in_ring(x, y, ring):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def main():
    rings = [(b["id"], np.array(b["rings"][0])[:, :2]) for b in json.loads((BUILD / "osm_buildings.json").read_text())]
    out = {"buildings": {}, "places": [], "masts": [], "hotspots": [], "categories": []}
    for b in BUILDINGS:
        x, y = b["at"] if "at" in b else wgs_to_grid(*b["latlon"])
        pid = next((rid for rid, ring in rings if point_in_ring(x, y, ring)), None)
        if pid is None:
            dist, rid = min((np.hypot(*(ring.mean(0) - (x, y))), rid) for rid, ring in rings)
            pid = rid if dist <= 20 else None
        height = b["tower"]["top"] if "tower" in b else b["ruin"]["height"] if "ruin" in b else 8.0
        out["places"].append({"name": b["name"], "note": b.get("kind"), "x": x, "y": y, "h": round(height, 1)})
        if pid and "style" in b:
            out["buildings"][pid] = {k: v for k, v in b.items() if k not in ("at", "latlon", *INFO)}
        if pid:
            out["hotspots"].append({"name": b["name"], "pid": pid, "x": x, "y": y, **{k: b[k] for k in INFO if k in b}})
        else:
            out["hotspots"].append({"name": b["name"], "x": x, "y": y, **{k: b[k] for k in INFO if k in b}})
        print(f"{b['name']}: {pid or 'no building'}")
    for p in POINTS:
        x, y = p["at"] if "at" in p else wgs_to_grid(*p["latlon"])
        out["places"].append({"name": p["name"], "note": p.get("kind"), "x": x, "y": y, "h": 2.0})
        out["hotspots"].append({"name": p["name"], "x": x, "y": y, **{k: p[k] for k in INFO if k in p}})
    for m in MASTS:
        x, y = m["at"]
        out["masts"].append({**{k: v for k, v in m.items() if k != "at"}, "x": x, "y": y})
    names = {h["name"] for h in out["hotspots"]}
    for key, label, members in CATEGORIES:
        missing = [n for n in members if n not in names]
        assert not missing, f"{label}: no hotspot for {missing}"
        out["categories"].append({"key": key, "label": label, "names": members})
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "landmarks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
