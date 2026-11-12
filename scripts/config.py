"""Shared settings: the area we render, in British National Grid (OSGB36, EPSG:27700) meters."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
BUILD = ROOT / "data" / "build"

NAME = "Houghton Conquest"
CRS = "EPSG:27700"

# Houghton Conquest, Bedfordshire, with Houghton House on the Greensand Ridge to the south: 3 x 4 km.
XMIN, YMIN, XMAX, YMAX = 503000, 238500, 506000, 242500
CX, CY = (XMIN + XMAX) / 2, (YMIN + YMAX) / 2
# The middle of the village (the High Street by the shop): where walks through a series start.
VILLAGE = (504555, 241553)

# Downloads and ground textures come in square chunks of this many meters.
CHUNK = 500
NX, NY = int((XMAX - XMIN) / CHUNK), int((YMAX - YMIN) / CHUNK)

# Ground raster resolution in meters per pixel. The Environment Agency lidar is 1 m, so 0.5 m is plenty (and keeps
# the ground raster within memory).
GROUND_RES = 0.5
# Lidar (canopy height model) resolution in meters per pixel.
LIDAR_RES = 1.0

# The compass directions the map is rendered from (render/run.mjs --look); the first is the one the viewer opens on.
VIEWS = (45, 135, 225, 315)


def write_area():
    """The renderer and the viewer read the area from here, with a polynomial for grid <-> WGS84 (scripts/geo.py)."""
    from geo import js_fit

    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "area.json").write_text(
        json.dumps({"xmin": XMIN, "ymin": YMIN, "xmax": XMAX, "ymax": YMAX, "cx": CX, "cy": CY, "chunk": CHUNK, "nx": NX, "ny": NY,
                    "name": NAME, "village": VILLAGE, "geo": js_fit()})
    )
